from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from datasets import Dataset as HFDataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings


FILE_BLOCK_PATTERN = re.compile(
    r"\[\[FILE:(?P<path>[^\]|]+)(?:\s*\|[^\]]*)?\]\]\s*(?P<body>.*?)(?=(?:\n\n==========\n\n|\n\n----------\n\n|\Z))",
    re.DOTALL,
)
SOURCE_PATTERNS = [
    re.compile(r"来源文件\s*:\s*`([^`]+)`", re.IGNORECASE),
    re.compile(r"source\s*file\s*:\s*`([^`]+)`", re.IGNORECASE),
]
BACKTICK_MD_PATTERN = re.compile(r"`([^`]+?\.md)`", re.IGNORECASE)
RAGAS_MAX_WORKERS = max(1, min(int(os.getenv("RAGAS_MAX_WORKERS", "4")), 16))
RAGAS_TIMEOUT_SEC = max(30, min(int(os.getenv("RAGAS_TIMEOUT_SEC", "180")), 600))


def normalize_text(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = lowered.replace("，", ",").replace("：", ":")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def normalize_path(path: str) -> str:
    return (path or "").strip().replace("\\", "/").lower()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    item = str(value).strip()
    return [item] if item else []


def dedup_list(items: list[str], *, path_mode: bool = False) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item:
            continue
        key = normalize_path(item) if path_mode else normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = math.ceil((p / 100.0) * len(sorted_values)) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return sorted_values[rank]


def extract_cited_files(answer: str) -> list[str]:
    files: list[str] = []

    for pattern in SOURCE_PATTERNS:
        for match in pattern.finditer(answer or ""):
            path = match.group(1).strip()
            if path:
                files.append(path)

    if not files:
        for match in BACKTICK_MD_PATTERN.finditer(answer or ""):
            path = match.group(1).strip()
            if path:
                files.append(path)

    return dedup_list(files, path_mode=True)


def extract_contexts_from_tool_result(content: str, max_chars: int = 1200) -> tuple[list[str], list[str]]:
    contexts: list[str] = []
    file_paths: list[str] = []

    for match in FILE_BLOCK_PATTERN.finditer(content or ""):
        file_path = normalize_whitespace(match.group("path"))
        body = normalize_whitespace(match.group("body"))
        if file_path:
            file_paths.append(file_path)
        if not body:
            continue
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + " ..."
        contexts.append(body)

    return dedup_list(contexts), dedup_list(file_paths, path_mode=True)


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"评测集不存在: {dataset_path}")

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_questions = payload
    elif isinstance(payload, dict):
        raw_questions = payload.get("问题列表") or payload.get("questions") or []
    else:
        raw_questions = []

    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("评测集格式错误：缺少非空的 `问题列表`。")

    normalized: list[dict[str, Any]] = []
    invalid_items: list[str] = []
    for idx, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            invalid_items.append(f"第 {idx} 题不是对象")
            continue

        qid = str(item.get("id") or f"Q{idx:03d}")
        question = str(item.get("问题") or item.get("question") or item.get("user_input") or "").strip()
        reference = str(
            item.get("参考答案")
            or item.get("reference")
            or item.get("ground_truth")
            or item.get("reference_answer")
            or ""
        ).strip()
        reference_contexts = coerce_str_list(item.get("参考上下文") or item.get("reference_contexts"))
        expected_files = coerce_str_list(item.get("期望证据文件") or item.get("expected_files"))

        if not question:
            invalid_items.append(f"{qid} 缺少 `问题`")
            continue
        if not reference:
            invalid_items.append(f"{qid} 缺少 `参考答案`（RAGAS 评测必需）")
            continue

        normalized.append(
            {
                "id": qid,
                "问题": question,
                "参考答案": reference,
                "参考上下文": reference_contexts,
                "期望证据文件": expected_files,
            }
        )

    if invalid_items:
        preview = "\n- ".join(invalid_items[:8])
        raise ValueError(
            "评测集存在不可用条目，请修复后再运行（当前只支持 RAGAS 数据格式）。\n"
            f"- {preview}"
        )
    return normalized


def stream_chat_once(
    base_url: str,
    question: str,
    provider: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat"
    payload: dict[str, Any] = {"messages": [{"role": "user", "content": question}]}
    if provider:
        payload["provider"] = provider

    answer_chunks: list[str] = []
    tool_call_names: list[str] = []
    tool_call_id_to_name: dict[str, str] = {}
    retrieved_contexts: list[str] = []
    retrieved_files: list[str] = []
    usage_total_input = 0
    usage_total_output = 0
    usage_total_tokens = 0
    tool_call_rounds = 0

    started_at = time.perf_counter()
    with httpx.Client(timeout=timeout_sec) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else raw_line
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if not data:
                    continue

                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "content":
                    answer_chunks.append(event.get("content", ""))
                    continue

                if event_type == "tool_calls":
                    calls = event.get("tool_calls") or []
                    if calls:
                        tool_call_rounds += 1
                    for call in calls:
                        fn = (call.get("function") or {}).get("name")
                        call_id = str(call.get("id") or "").strip()
                        if fn:
                            tool_call_names.append(fn)
                        if call_id and fn:
                            tool_call_id_to_name[call_id] = str(fn)
                    continue

                if event_type == "tool_results":
                    results = event.get("results") or []
                    for tool_result in results:
                        tool_call_id = str(tool_result.get("tool_call_id") or "").strip()
                        tool_name = tool_call_id_to_name.get(tool_call_id, "")
                        if not tool_name:
                            meta = tool_result.get("meta", {})
                            if isinstance(meta, dict):
                                tool_name = str(meta.get("tool_name") or "").strip()
                        content = str(tool_result.get("content") or "")
                        if not content:
                            continue
                        if tool_name in {"retrieve_sections", "retrieve_files"}:
                            ctxs, files = extract_contexts_from_tool_result(content)
                            retrieved_contexts.extend(ctxs)
                            retrieved_files.extend(files)
                    continue

                if event_type == "usage":
                    usage = event.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
                    usage_total_input += prompt_tokens
                    usage_total_output += completion_tokens
                    usage_total_tokens += total_tokens
                    continue

                if event_type == "done":
                    break

    latency_ms = (time.perf_counter() - started_at) * 1000.0
    return {
        "answer": "".join(answer_chunks).strip(),
        "tool_call_rounds": tool_call_rounds,
        "tool_call_names": tool_call_names,
        "retrieved_contexts": dedup_list(retrieved_contexts),
        "retrieved_files": dedup_list(retrieved_files, path_mode=True),
        "usage_total_input": usage_total_input,
        "usage_total_output": usage_total_output,
        "usage_total_tokens": usage_total_tokens,
        "latency_ms": round(latency_ms, 2),
    }


def build_ragas_wrappers(provider_name: str | None) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper, dict[str, str]]:
    judge_provider = (provider_name or settings.api_provider).strip()
    judge_cfg = settings.get_provider_config(judge_provider)

    ragas_embedding_provider = os.getenv("RAGAS_EMBEDDING_PROVIDER", "").strip()
    if not ragas_embedding_provider:
        ragas_embedding_provider = (
            "ragas_embedding"
            if settings.get_provider_config("ragas_embedding").get("model")
            else judge_provider
        )
    embedding_cfg = settings.get_provider_config(ragas_embedding_provider)

    judge_api_key = str(judge_cfg.get("api_key") or "").strip()
    judge_base_url = str(judge_cfg.get("base_url") or "").strip() or None
    judge_model = str(judge_cfg.get("model") or "").strip()
    judge_headers = judge_cfg.get("headers") if isinstance(judge_cfg.get("headers"), dict) else None

    if not judge_api_key:
        raise ValueError(f"缺少 {judge_provider.upper()}_API_KEY，无法执行 RAGAS 评分。")
    if not judge_model:
        raise ValueError(f"缺少 {judge_provider.upper()}_MODEL，无法执行 RAGAS 评分。")

    embedding_api_key = str(embedding_cfg.get("api_key") or "").strip()
    embedding_base_url = str(embedding_cfg.get("base_url") or "").strip() or judge_base_url
    embedding_headers = (
        embedding_cfg.get("headers") if isinstance(embedding_cfg.get("headers"), dict) else None
    )
    embedding_model = (
        os.getenv(f"{ragas_embedding_provider.upper()}_EMBEDDING_MODEL")
        or os.getenv("RAGAS_EMBEDDING_MODEL")
        or os.getenv(f"{ragas_embedding_provider.upper()}_MODEL")
        or str(embedding_cfg.get("model") or "").strip()
        or "text-embedding-3-small"
    ).strip()

    if not embedding_api_key:
        raise ValueError(
            f"缺少 {ragas_embedding_provider.upper()}_API_KEY，无法执行 RAGAS embedding 评分。"
        )
    if not embedding_model:
        raise ValueError("未设置 embedding model，无法执行 RAGAS 评分。")

    llm = ChatOpenAI(
        model=judge_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
        default_headers=judge_headers,
        timeout=RAGAS_TIMEOUT_SEC,
        max_retries=2,
        temperature=0.0,
        use_responses_api=False,
    )
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=embedding_api_key,
        base_url=embedding_base_url,
        default_headers=embedding_headers,
        timeout=RAGAS_TIMEOUT_SEC,
        max_retries=2,
    )

    return (
        LangchainLLMWrapper(llm),
        LangchainEmbeddingsWrapper(embeddings),
        {
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "embedding_provider": ragas_embedding_provider,
            "embedding_model": embedding_model,
            "judge_base_url": judge_base_url or "",
            "embedding_base_url": embedding_base_url or "",
        },
    )


def apply_ragas_scores(
    results: list[dict[str, Any]],
    judge_provider: str | None,
    log_callback: Any | None = None,
) -> dict[str, str]:
    scored_indices: list[int] = []
    ragas_rows: list[dict[str, Any]] = []

    for idx, item in enumerate(results):
        if item.get("错误"):
            continue
        user_input = str(item.get("问题") or "").strip()
        response = str(item.get("答案") or "").strip()
        reference = str(item.get("参考答案") or "").strip()
        retrieved_contexts = coerce_str_list(item.get("检索上下文"))
        if not user_input or not response or not reference:
            continue

        if not retrieved_contexts:
            retrieved_contexts = [""]

        ragas_rows.append(
            {
                "user_input": user_input,
                "response": response,
                "reference": reference,
                "retrieved_contexts": retrieved_contexts,
            }
        )
        scored_indices.append(idx)

    if not ragas_rows:
        for item in results:
            if not item.get("错误"):
                item["错误"] = "无可用样本：答案或参考答案为空，无法执行 RAGAS 评分。"
            item["忠诚度"] = None
            item["上下文召回率"] = None
            item["答案准确度"] = None
        return {}

    if callable(log_callback):
        log_callback(f"[RAGAS] 待评分题数: {len(ragas_rows)}")

    llm_wrapper, emb_wrapper, ragas_meta = build_ragas_wrappers(judge_provider)
    metrics = [
        copy.deepcopy(Faithfulness()),
        copy.deepcopy(ContextRecall()),
        copy.deepcopy(AnswerCorrectness()),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        run_config = RunConfig(
            timeout=RAGAS_TIMEOUT_SEC,
            max_retries=3,
            max_wait=60,
            max_workers=RAGAS_MAX_WORKERS,
        )
        ragas_result = evaluate(
            dataset=HFDataset.from_list(ragas_rows),
            metrics=metrics,
            llm=llm_wrapper,
            embeddings=emb_wrapper,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=False,
        )

    per_row_scores = ragas_result.scores
    for local_idx, score_dict in enumerate(per_row_scores):
        result_idx = scored_indices[local_idx]
        results[result_idx]["忠诚度"] = to_float(score_dict.get("faithfulness"))
        results[result_idx]["上下文召回率"] = to_float(score_dict.get("context_recall"))
        results[result_idx]["答案准确度"] = to_float(score_dict.get("answer_correctness"))

    for idx, item in enumerate(results):
        if idx in scored_indices:
            continue
        item.setdefault("忠诚度", None)
        item.setdefault("上下文召回率", None)
        item.setdefault("答案准确度", None)

    return ragas_meta


def run_eval(
    questions: list[dict[str, Any]],
    base_url: str,
    provider: str | None,
    timeout_sec: float,
    progress_callback: Any | None = None,
    log_callback: Any | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(questions)

    for idx, item in enumerate(questions, start=1):
        qid = item.get("id", f"Q{idx:03d}")
        question = str(item.get("问题", "")).strip()

        log_line = f"[{idx:02d}/{total:02d}] {qid} - {question}"
        if callable(log_callback):
            log_callback(log_line)
        else:
            print(log_line)

        if callable(progress_callback):
            progress_callback({"current": idx, "total": total, "id": qid, "question": question})

        response: dict[str, Any]
        error = ""
        try:
            response = stream_chat_once(
                base_url=base_url,
                question=question,
                provider=provider,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            response = {
                "answer": "",
                "tool_call_rounds": 0,
                "tool_call_names": [],
                "retrieved_contexts": [],
                "retrieved_files": [],
                "usage_total_input": 0,
                "usage_total_output": 0,
                "usage_total_tokens": 0,
                "latency_ms": 0.0,
            }
            error = str(exc)

        answer = response["answer"]
        cited_files = extract_cited_files(answer)

        results.append(
            {
                "id": qid,
                "问题": question,
                "参考答案": str(item.get("参考答案") or ""),
                "参考上下文": coerce_str_list(item.get("参考上下文")),
                "期望证据文件": coerce_str_list(item.get("期望证据文件")),
                "答案": answer,
                "错误": error,
                "引用证据文件": cited_files,
                "检索证据文件": response["retrieved_files"],
                "检索上下文": response["retrieved_contexts"],
                "忠诚度": None,
                "上下文召回率": None,
                "答案准确度": None,
                "tool_call_rounds": response["tool_call_rounds"],
                "tool_call_names": response["tool_call_names"],
                "usage_total_input": response["usage_total_input"],
                "usage_total_output": response["usage_total_output"],
                "usage_total_tokens": response["usage_total_tokens"],
                "latency_ms": response["latency_ms"],
            }
        )

    try:
        ragas_meta = apply_ragas_scores(results, judge_provider=provider, log_callback=log_callback)
        for item in results:
            item["RAGAS评分器"] = ragas_meta
    except Exception as exc:
        err = f"RAGAS 评分失败: {exc}"
        if callable(log_callback):
            log_callback(f"[ERROR] {err}")
        for item in results:
            item["忠诚度"] = None
            item["上下文召回率"] = None
            item["答案准确度"] = None
            if not item.get("错误"):
                item["错误"] = err

    return results


def summarize_results(
    results: list[dict[str, Any]],
    run_name: str,
    base_url: str,
    provider: str | None,
    dataset_path: Path,
) -> dict[str, Any]:
    total = len(results)
    faithfulness_values = [v for v in (to_float(r.get("忠诚度")) for r in results) if v is not None]
    context_recall_values = [v for v in (to_float(r.get("上下文召回率")) for r in results) if v is not None]
    answer_correctness_values = [v for v in (to_float(r.get("答案准确度")) for r in results) if v is not None]

    token_values = [int(r["usage_total_tokens"]) for r in results if int(r.get("usage_total_tokens", 0)) > 0]
    latency_values = [float(r["latency_ms"]) for r in results if float(r.get("latency_ms", 0.0)) >= 0]
    input_values = [int(r["usage_total_input"]) for r in results if int(r.get("usage_total_input", 0)) > 0]
    output_values = [int(r["usage_total_output"]) for r in results if int(r.get("usage_total_output", 0)) > 0]
    no_context_count = sum(1 for r in results if not coerce_str_list(r.get("检索上下文")))

    summary = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "provider": provider or "",
        "dataset_path": str(dataset_path),
        "题目总数": total,
        "指标": {
            "忠诚度": round(sum(faithfulness_values) / len(faithfulness_values), 4) if faithfulness_values else 0.0,
            "上下文召回率": round(sum(context_recall_values) / len(context_recall_values), 4)
            if context_recall_values
            else 0.0,
            "答案准确度": round(sum(answer_correctness_values) / len(answer_correctness_values), 4)
            if answer_correctness_values
            else 0.0,
            "平均token": round(sum(token_values) / len(token_values), 2) if token_values else 0.0,
            "平均延迟_ms": round(sum(latency_values) / len(latency_values), 2) if latency_values else 0.0,
        },
        "附加统计": {
            "平均输入token": round(sum(input_values) / len(input_values), 2) if input_values else 0.0,
            "平均输出token": round(sum(output_values) / len(output_values), 2) if output_values else 0.0,
            "token可用题数": len(token_values),
            "RAGAS可用题数": len(answer_correctness_values),
            "无检索上下文题数": no_context_count,
            "延迟P50_ms": round(percentile(latency_values, 50), 2) if latency_values else 0.0,
            "延迟P95_ms": round(percentile(latency_values, 95), 2) if latency_values else 0.0,
            "失败题数": sum(1 for r in results if r.get("错误")),
        },
    }
    return summary


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, float]:
    curr_metrics = current.get("指标", {})
    prev_metrics = previous.get("指标", {})

    delta: dict[str, float] = {}
    for key in ["忠诚度", "上下文召回率", "答案准确度", "平均token", "平均延迟_ms"]:
        delta[key] = float(curr_metrics.get(key, 0.0)) - float(prev_metrics.get(key, 0.0))
    return delta


def to_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_markdown_report(
    summary: dict[str, Any],
    detail_path: Path,
    summary_path: Path,
    compare_summary: dict[str, Any] | None = None,
) -> str:
    metrics = summary["指标"]
    extra = summary["附加统计"]

    lines: list[str] = []
    lines.append(f"# RAGAS 评测结果 - {summary['run_name']}")
    lines.append("")
    lines.append(f"- 生成时间: `{summary['created_at']}`")
    lines.append(f"- 题目数量: `{summary['题目总数']}`")
    lines.append(f"- 接口地址: `{summary['base_url']}`")
    lines.append(f"- Provider: `{summary.get('provider') or '默认'}`")
    lines.append("")
    lines.append("## 核心指标（RAGAS + 性能）")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| 忠诚度 (Faithfulness) | {to_percent(float(metrics['忠诚度']))} |")
    lines.append(f"| 上下文召回率 (Context Recall) | {to_percent(float(metrics['上下文召回率']))} |")
    lines.append(f"| 答案准确度 (Answer Correctness) | {to_percent(float(metrics['答案准确度']))} |")
    lines.append(f"| 平均 token | {float(metrics['平均token']):.2f} |")
    lines.append(f"| 平均延迟 | {float(metrics['平均延迟_ms']):.2f} ms |")
    lines.append("")
    lines.append("## 补充统计")
    lines.append("")
    lines.append(f"- 平均输入 token: `{float(extra['平均输入token']):.2f}`")
    lines.append(f"- 平均输出 token: `{float(extra['平均输出token']):.2f}`")
    lines.append(f"- token 可用题数: `{extra['token可用题数']}`")
    lines.append(f"- RAGAS 可用题数: `{extra['RAGAS可用题数']}`")
    lines.append(f"- 无检索上下文题数: `{extra['无检索上下文题数']}`")
    lines.append(f"- 延迟 P50: `{float(extra['延迟P50_ms']):.2f} ms`")
    lines.append(f"- 延迟 P95: `{float(extra['延迟P95_ms']):.2f} ms`")
    lines.append(f"- 失败题数: `{extra['失败题数']}`")

    if compare_summary:
        delta = build_delta(summary, compare_summary)
        lines.append("")
        lines.append("## 与上一轮对比")
        lines.append("")
        lines.append(f"- 对比基线: `{compare_summary.get('run_name', 'unknown')}`")
        lines.append("")
        lines.append("| 指标 | 当前 | 基线 | 差值(当前-基线) |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| 忠诚度 | {to_percent(float(metrics['忠诚度']))} | "
            f"{to_percent(float(compare_summary['指标'].get('忠诚度', 0.0)))} | "
            f"{to_percent(float(delta['忠诚度']))} |"
        )
        lines.append(
            f"| 上下文召回率 | {to_percent(float(metrics['上下文召回率']))} | "
            f"{to_percent(float(compare_summary['指标'].get('上下文召回率', 0.0)))} | "
            f"{to_percent(float(delta['上下文召回率']))} |"
        )
        lines.append(
            f"| 答案准确度 | {to_percent(float(metrics['答案准确度']))} | "
            f"{to_percent(float(compare_summary['指标'].get('答案准确度', 0.0)))} | "
            f"{to_percent(float(delta['答案准确度']))} |"
        )
        lines.append(
            f"| 平均 token | {float(metrics['平均token']):.2f} | "
            f"{float(compare_summary['指标'].get('平均token', 0.0)):.2f} | "
            f"{float(delta['平均token']):+.2f} |"
        )
        lines.append(
            f"| 平均延迟(ms) | {float(metrics['平均延迟_ms']):.2f} | "
            f"{float(compare_summary['指标'].get('平均延迟_ms', 0.0)):.2f} | "
            f"{float(delta['平均延迟_ms']):+.2f} |"
        )

    lines.append("")
    lines.append("## 结果文件")
    lines.append("")
    lines.append(f"- 汇总: `{summary_path}`")
    lines.append(f"- 明细: `{detail_path}`")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir / "ragas评测集_40题.json"
    default_output_dir = script_dir / "结果"

    parser = argparse.ArgumentParser(description="运行 Deep RAG 的 RAGAS 评测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="后端服务地址")
    parser.add_argument("--dataset", default=str(default_dataset), help="评测集 JSON 文件路径")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="结果输出目录")
    parser.add_argument("--run-name", default="", help="本次评测名称，例如 baseline 或 optimized")
    parser.add_argument("--provider", default="", help="可选，覆盖 provider（同时用于问答与RAGAS评分）")
    parser.add_argument("--timeout", type=float, default=180.0, help="单题请求超时（秒）")
    parser.add_argument("--max-questions", type=int, default=0, help="仅跑前 N 题，0 表示全量")
    parser.add_argument("--compare", default="", help="可选，上一轮汇总 JSON 路径，用于出对比")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_dataset(dataset_path)
    if args.max_questions > 0:
        questions = questions[: args.max_questions]

    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    provider = args.provider.strip() or None

    print("开始 RAGAS 评测...")
    print(f"题目数: {len(questions)}")
    print(f"接口: {args.base_url}")
    print(f"数据集: {dataset_path}")
    print("")

    results = run_eval(
        questions=questions,
        base_url=args.base_url,
        provider=provider,
        timeout_sec=args.timeout,
    )
    summary = summarize_results(
        results=results,
        run_name=run_name,
        base_url=args.base_url,
        provider=provider,
        dataset_path=dataset_path,
    )

    compare_summary = None
    if args.compare:
        compare_path = Path(args.compare).resolve()
        if compare_path.exists():
            compare_summary = load_summary(compare_path)
        else:
            print(f"[WARN] 对比文件不存在，已跳过: {compare_path}")

    detail_path = output_dir / f"{run_name}_明细.json"
    summary_path = output_dir / f"{run_name}_汇总.json"
    report_path = output_dir / f"{run_name}_报告.md"

    detail_payload = {
        "run_name": run_name,
        "created_at": summary["created_at"],
        "题目总数": len(results),
        "results": results,
    }
    detail_path.write_text(json.dumps(detail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_content = build_markdown_report(
        summary=summary,
        detail_path=detail_path,
        summary_path=summary_path,
        compare_summary=compare_summary,
    )
    report_path.write_text(report_content, encoding="utf-8")

    print("")
    print("评测完成。")
    print(f"忠诚度: {to_percent(float(summary['指标']['忠诚度']))}")
    print(f"上下文召回率: {to_percent(float(summary['指标']['上下文召回率']))}")
    print(f"答案准确度: {to_percent(float(summary['指标']['答案准确度']))}")
    print(f"平均 token: {float(summary['指标']['平均token']):.2f}")
    print(f"平均延迟: {float(summary['指标']['平均延迟_ms']):.2f} ms")
    print(f"汇总文件: {summary_path}")
    print(f"明细文件: {detail_path}")
    print(f"报告文件: {report_path}")


if __name__ == "__main__":
    main()
