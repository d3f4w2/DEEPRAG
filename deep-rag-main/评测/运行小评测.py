from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


SOURCE_PATTERNS = [
    re.compile(r"来源文件\s*:\s*`([^`]+)`", re.IGNORECASE),
    re.compile(r"source\s*file\s*:\s*`([^`]+)`", re.IGNORECASE),
]
BACKTICK_MD_PATTERN = re.compile(r"`([^`]+?\.md)`", re.IGNORECASE)


def normalize_text(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = lowered.replace("，", ",").replace("：", ":")
    lowered = re.sub(r"\s+", "", lowered)
    return lowered


def normalize_path(path: str) -> str:
    return (path or "").strip().replace("\\", "/").lower()


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

    dedup: list[str] = []
    seen: set[str] = set()
    for item in files:
        key = normalize_path(item)
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"评测集不存在: {dataset_path}")

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = data.get("问题列表")
    if not isinstance(questions, list) or not questions:
        raise ValueError("评测集格式错误：缺少非空的 `问题列表`。")
    return questions


def score_answer(answer: str, keyword_groups: list[Any]) -> tuple[bool, list[dict[str, Any]]]:
    answer_norm = normalize_text(answer)
    group_results: list[dict[str, Any]] = []

    for raw_group in keyword_groups:
        group = raw_group if isinstance(raw_group, list) else [raw_group]
        group = [str(item) for item in group if str(item).strip()]

        hit = False
        hit_keyword = ""
        for keyword in group:
            if normalize_text(keyword) in answer_norm:
                hit = True
                hit_keyword = keyword
                break

        group_results.append(
            {
                "关键词候选": group,
                "命中": hit,
                "命中关键词": hit_keyword,
            }
        )

    if not group_results:
        return False, []

    is_correct = all(item["命中"] for item in group_results)
    return is_correct, group_results


def score_evidence(expected_files: list[str], cited_files: list[str]) -> tuple[bool, list[str]]:
    expected_norm = {normalize_path(p) for p in expected_files if normalize_path(p)}
    cited_norm = {normalize_path(p) for p in cited_files if normalize_path(p)}

    if not expected_norm:
        return bool(cited_norm), []

    hit_norm = sorted(expected_norm & cited_norm)
    return bool(hit_norm), hit_norm


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = math.ceil((p / 100.0) * len(sorted_values)) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return sorted_values[rank]


def stream_chat_once(
    base_url: str,
    question: str,
    provider: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat"
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": question}],
    }
    if provider:
        payload["provider"] = provider

    answer_chunks: list[str] = []
    tool_call_names: list[str] = []
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
                elif event_type == "tool_calls":
                    calls = event.get("tool_calls") or []
                    if calls:
                        tool_call_rounds += 1
                    for call in calls:
                        fn = (call.get("function") or {}).get("name")
                        if fn:
                            tool_call_names.append(fn)
                elif event_type == "usage":
                    usage = event.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
                    usage_total_input += prompt_tokens
                    usage_total_output += completion_tokens
                    usage_total_tokens += total_tokens
                elif event_type == "done":
                    break

    latency_ms = (time.perf_counter() - started_at) * 1000.0

    return {
        "answer": "".join(answer_chunks).strip(),
        "tool_call_rounds": tool_call_rounds,
        "tool_call_names": tool_call_names,
        "usage_total_input": usage_total_input,
        "usage_total_output": usage_total_output,
        "usage_total_tokens": usage_total_tokens,
        "latency_ms": round(latency_ms, 2),
    }


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
        keyword_groups = item.get("判分关键词组") or []
        expected_files = item.get("期望证据文件") or []

        log_line = f"[{idx:02d}/{total:02d}] {qid} - {question}"
        if callable(log_callback):
            log_callback(log_line)
        else:
            print(log_line)

        if callable(progress_callback):
            progress_callback(
                {
                    "current": idx,
                    "total": total,
                    "id": qid,
                    "question": question,
                }
            )

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
                "usage_total_input": 0,
                "usage_total_output": 0,
                "usage_total_tokens": 0,
                "latency_ms": 0.0,
            }
            error = str(exc)

        answer = response["answer"]
        cited_files = extract_cited_files(answer)
        is_correct, keyword_group_results = score_answer(answer, keyword_groups)
        evidence_hit, evidence_hit_files = score_evidence(expected_files, cited_files)

        result = {
            "id": qid,
            "问题": question,
            "答案": answer,
            "错误": error,
            "判分关键词组结果": keyword_group_results,
            "准确": is_correct and not error,
            "期望证据文件": expected_files,
            "引用证据文件": cited_files,
            "证据命中": evidence_hit and not error,
            "命中证据文件(归一化)": evidence_hit_files,
            "tool_call_rounds": response["tool_call_rounds"],
            "tool_call_names": response["tool_call_names"],
            "usage_total_input": response["usage_total_input"],
            "usage_total_output": response["usage_total_output"],
            "usage_total_tokens": response["usage_total_tokens"],
            "latency_ms": response["latency_ms"],
        }
        results.append(result)

    return results


def summarize_results(
    results: list[dict[str, Any]],
    run_name: str,
    base_url: str,
    provider: str | None,
    dataset_path: Path,
) -> dict[str, Any]:
    total = len(results)
    correct_count = sum(1 for r in results if r["准确"])
    evidence_hit_count = sum(1 for r in results if r["证据命中"])

    token_values = [int(r["usage_total_tokens"]) for r in results if int(r["usage_total_tokens"]) > 0]
    latency_values = [float(r["latency_ms"]) for r in results]
    input_values = [int(r["usage_total_input"]) for r in results if int(r["usage_total_input"]) > 0]
    output_values = [int(r["usage_total_output"]) for r in results if int(r["usage_total_output"]) > 0]

    summary = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "provider": provider or "",
        "dataset_path": str(dataset_path),
        "题目总数": total,
        "指标": {
            "准确率": round(correct_count / total, 4) if total else 0.0,
            "证据命中率": round(evidence_hit_count / total, 4) if total else 0.0,
            "平均token": round(sum(token_values) / len(token_values), 2) if token_values else 0.0,
            "平均延迟_ms": round(sum(latency_values) / len(latency_values), 2) if latency_values else 0.0,
        },
        "附加统计": {
            "平均输入token": round(sum(input_values) / len(input_values), 2) if input_values else 0.0,
            "平均输出token": round(sum(output_values) / len(output_values), 2) if output_values else 0.0,
            "token可用题数": len(token_values),
            "延迟P50_ms": round(percentile(latency_values, 50), 2) if latency_values else 0.0,
            "延迟P95_ms": round(percentile(latency_values, 95), 2) if latency_values else 0.0,
            "失败题数": sum(1 for r in results if r["错误"]),
        },
    }
    return summary


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, float]:
    curr_metrics = current.get("指标", {})
    prev_metrics = previous.get("指标", {})

    delta = {}
    for key in ["准确率", "证据命中率", "平均token", "平均延迟_ms"]:
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
    lines.append(f"# 小评测结果 - {summary['run_name']}")
    lines.append("")
    lines.append(f"- 生成时间: `{summary['created_at']}`")
    lines.append(f"- 题目数量: `{summary['题目总数']}`")
    lines.append(f"- 接口地址: `{summary['base_url']}`")
    lines.append(f"- Provider: `{summary.get('provider') or '默认'}`")
    lines.append("")
    lines.append("## 四项核心指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| 准确率 | {to_percent(float(metrics['准确率']))} |")
    lines.append(f"| 证据命中率 | {to_percent(float(metrics['证据命中率']))} |")
    lines.append(f"| 平均 token | {float(metrics['平均token']):.2f} |")
    lines.append(f"| 平均延迟 | {float(metrics['平均延迟_ms']):.2f} ms |")
    lines.append("")
    lines.append("## 补充统计")
    lines.append("")
    lines.append(f"- 平均输入 token: `{float(extra['平均输入token']):.2f}`")
    lines.append(f"- 平均输出 token: `{float(extra['平均输出token']):.2f}`")
    lines.append(f"- token 可用题数: `{extra['token可用题数']}`")
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
            f"| 准确率 | {to_percent(float(metrics['准确率']))} | "
            f"{to_percent(float(compare_summary['指标']['准确率']))} | "
            f"{to_percent(float(delta['准确率']))} |"
        )
        lines.append(
            f"| 证据命中率 | {to_percent(float(metrics['证据命中率']))} | "
            f"{to_percent(float(compare_summary['指标']['证据命中率']))} | "
            f"{to_percent(float(delta['证据命中率']))} |"
        )
        lines.append(
            f"| 平均 token | {float(metrics['平均token']):.2f} | "
            f"{float(compare_summary['指标']['平均token']):.2f} | "
            f"{float(delta['平均token']):+.2f} |"
        )
        lines.append(
            f"| 平均延迟(ms) | {float(metrics['平均延迟_ms']):.2f} | "
            f"{float(compare_summary['指标']['平均延迟_ms']):.2f} | "
            f"{float(delta['平均延迟_ms']):+.2f} |"
        )

    lines.append("")
    lines.append("## 结果文件")
    lines.append("")
    lines.append(f"- 汇总: `{summary_path}`")
    lines.append(f"- 明细: `{detail_path}`")
    lines.append("")
    lines.append("> 建议面试展示时优先贴“与上一轮对比”表格。")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir / "小评测集_40题.json"
    default_output_dir = script_dir / "结果"

    parser = argparse.ArgumentParser(description="运行 Deep RAG 小评测（40题）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="后端服务地址")
    parser.add_argument("--dataset", default=str(default_dataset), help="评测集 JSON 文件路径")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="结果输出目录")
    parser.add_argument("--run-name", default="", help="本次评测名称，例如 baseline 或 optimized")
    parser.add_argument("--provider", default="", help="可选，覆盖 provider")
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

    print("开始评测...")
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
    print(f"准确率: {to_percent(float(summary['指标']['准确率']))}")
    print(f"证据命中率: {to_percent(float(summary['指标']['证据命中率']))}")
    print(f"平均 token: {float(summary['指标']['平均token']):.2f}")
    print(f"平均延迟: {float(summary['指标']['平均延迟_ms']):.2f} ms")
    print(f"汇总文件: {summary_path}")
    print(f"明细文件: {detail_path}")
    print(f"报告文件: {report_path}")


if __name__ == "__main__":
    main()
