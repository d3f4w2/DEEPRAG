from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from datetime import datetime, timezone
import json
import os
import mimetypes
from pathlib import Path
import signal
import re
from typing import Any, AsyncIterator, Dict, List, Optional
from dotenv import load_dotenv, find_dotenv

from backend.budget_control import (
    accumulate_usage_tokens,
    build_budget_event,
    check_budget_guard,
    create_budget_state,
)
from backend.config import settings
from backend.models import (
    ChatRequest, FileRetrievalRequest, FileRetrievalResponse,
    KnowledgeBaseInfo, HealthResponse, EvalStartRequest,
    VoiceDraftRequest, VoiceDraftResponse, VoiceIngestRequest, VoiceIngestResponse,
    ImageDraftResponse, ImageIngestResponse
)
from backend.knowledge_base import knowledge_base
from backend.llm_provider import LLMProvider
from backend.evaluation_manager import evaluation_manager
from backend.image_processing import (
    inspect_image_bytes,
    parse_tags_text,
    run_paddle_ocr,
    run_vision_analysis,
)
from backend.prompts import (
    create_react_system_prompt,
    create_retrieve_sections_tool,
    create_search_paths_tool,
    create_system_prompt,
    process_tool_calls,
)
from backend.react_handler import handle_react_mode

app = FastAPI(
    title="Deep RAG",
    version="1.0.0",
    description="A Deep RAG system that teaches AI to truly understand your knowledge base"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_ROUTE_PLAN = {
    "top_k": 6,
    "max_sections_per_file": 2,
    "min_file_paths": 4,
}
DEFAULT_QUESTION_ROUTE = {
    "route": "contextual_followup",
    "history_mode": "full",
    "max_iterations": 10,
    "reason": "default",
}
STAGE_ORDER = {
    "plan": 10,
    "execute": 20,
    "critic": 30,
}
STAGE_LABELS = {
    "plan": "Plan",
    "execute": "Execute",
    "critic": "Critic",
}
AUTO_RETRIEVAL_TOP_K_BOOST = 2
ENABLE_PLAN_EXECUTE_STAGES = os.getenv("ENABLE_PLAN_EXECUTE_STAGES", "false").strip().lower() in {"1", "true", "yes", "on"}
ENFORCE_CRITIC_RETRIEVAL = os.getenv("ENFORCE_CRITIC_RETRIEVAL", "true").strip().lower() in {"1", "true", "yes", "on"}
MAX_RETRIEVAL_ROUNDS = max(1, min(int(os.getenv("MAX_RETRIEVAL_ROUNDS", "4")), 20))
MAX_TOOL_CALL_ROUNDS = max(1, min(int(os.getenv("MAX_TOOL_CALL_ROUNDS", "10")), 40))
MAX_AUTO_RETRIEVAL_ROUNDS = max(0, min(int(os.getenv("MAX_AUTO_RETRIEVAL_ROUNDS", "3")), 10))
MAX_FINAL_EVIDENCE_ITEMS = 3
MAX_EVIDENCE_SNIPPET_LENGTH = 280
VOICE_NOTES_DIR = "Voice-Notes"
IMAGE_NOTES_DIR = "Image-Notes"
IMAGE_ASSETS_DIR = "assets"
MAX_IMAGE_UPLOAD_BYTES = int(os.getenv("MAX_IMAGE_UPLOAD_BYTES", str(10 * 1024 * 1024)))
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FALLBACK_NO_EVIDENCE_ANSWER = (
    "我无法给出有证据支撑的回答。已自动继续检索，但仍未找到可引用片段。"
)
FALLBACK_INSUFFICIENT_EVIDENCE_ANSWER = (
    "我无法给出有证据支撑的回答。检索轮次已用尽，证据仍不足以支撑结论。"
)
CRITIC_MIN_EVIDENCE_ITEMS = 2
CRITIC_MIN_DISTINCT_FILES = 1
CRITIC_MIN_QUERY_TERM_HITS = 1
CRITIC_ACCEPT_CONFIDENCE = 0.72
EVIDENCE_SECTION_PATTERN = re.compile(r"(?is)\n#{2,3}\s*(证据|evidence)\s*\n.*$")
EVIDENCE_BLOCK_PATTERN = re.compile(
    r"\[\[FILE:(?P<path>.+?)\s*\|\s*SECTION:(?P<section>\d+)\s*\|\s*SCORE:(?P<score>-?\d+(?:\.\d+)?)\s*\|\s*HEADING:(?P<heading>.*?)\]\]\s*(?P<body>.*?)(?=(?:\n\n==========\n\n|\n\n----------\n\n|\Z))",
    re.DOTALL,
)
IMAGE_NOTE_LINE_PATTERN = re.compile(r"(?im)^\s*-\s*image\s*:\s*`?([^\n`]+)`?\s*$")
UNCERTAIN_ANSWER_KEYWORDS = (
    "i don't know",
    "i dont know",
    "not sure",
    "insufficient",
    "不知道",
    "不确定",
    "无法判断",
    "暂无信息",
    "不清楚",
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _clip_text(text: str, max_len: int = 180) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3].rstrip() + "..."


def _normalize_stage_metrics(metrics: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    safe_metrics: List[Dict[str, str]] = []
    for item in metrics or []:
        if not isinstance(item, dict):
            continue
        label = _normalize_whitespace(str(item.get("label", "")))
        value = _normalize_whitespace(str(item.get("value", "")))
        if not label or not value:
            continue
        tone = _normalize_whitespace(str(item.get("tone", ""))).lower() or "neutral"
        safe_metrics.append(
            {
                "label": label,
                "value": value,
                "tone": tone,
            }
        )
    return safe_metrics


def _build_reasoning_stage_payload(
    *,
    stage_key: str,
    status: str,
    title: str,
    summary: str = "",
    mode: str = "llm",
    badge: str = "",
    metrics: Optional[List[Dict[str, Any]]] = None,
    order: Optional[int] = None,
) -> Dict[str, Any]:
    normalized_key = _normalize_whitespace(stage_key).lower() or "stage"
    normalized_status = _normalize_whitespace(status).lower() or "running"
    if normalized_status not in {"pending", "running", "completed", "failed"}:
        normalized_status = "running"
    normalized_mode = _normalize_whitespace(mode).lower() or "llm"
    stage_label = STAGE_LABELS.get(normalized_key, normalized_key.upper())
    stage_order = (
        int(order)
        if isinstance(order, int)
        else int(STAGE_ORDER.get(normalized_key, 99))
    )

    return {
        "type": "reasoning_stage",
        "stage_key": normalized_key,
        "stage_label": stage_label,
        "title": _clip_text(title, 100) or f"{stage_label} update",
        "summary": _clip_text(summary, 128),
        "status": normalized_status,
        "mode": normalized_mode,
        "badge": _normalize_whitespace(badge) or normalized_mode.upper(),
        "order": stage_order,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "metrics": _normalize_stage_metrics(metrics),
    }


def _tool_display_name(tool_name: str) -> str:
    normalized = _normalize_whitespace(tool_name).lower()
    if normalized == "search_paths":
        return "路径检索"
    if normalized == "retrieve_sections":
        return "片段检索"
    return normalized or "工具执行"


def _build_plan_stage_summary(user_query: str) -> str:
    query = _clip_text(user_query, 64)
    if not query:
        return "已基于问题意图生成检索路线，并准备执行。"
    return f"围绕问题“{query}”生成检索路线，优先覆盖相关文件并控制检索预算。"


def _compose_answer_for_delivery(candidate_answer: str, evidence_pool: List[Dict[str, str]]) -> str:
    if evidence_pool:
        return _format_answer_with_evidence(candidate_answer, evidence_pool)
    clean_answer = _strip_existing_evidence_section(candidate_answer or "").strip()
    if clean_answer:
        return clean_answer
    return FALLBACK_NO_EVIDENCE_ANSWER


def _build_critic_stage_payload(critic_payload: Dict[str, Any]) -> Dict[str, Any]:
    decision = _normalize_whitespace(str(critic_payload.get("decision", "revise"))).lower()
    if decision not in {"accept", "revise", "refuse"}:
        decision = "revise"

    evidence_items = int(critic_payload.get("evidence_items", 0) or 0)
    has_some_evidence = evidence_items > 0
    retrieval_rounds = int(critic_payload.get("retrieval_rounds", 0) or 0)
    max_retrieval_rounds = int(critic_payload.get("max_retrieval_rounds", MAX_RETRIEVAL_ROUNDS) or MAX_RETRIEVAL_ROUNDS)
    tool_call_rounds = int(critic_payload.get("tool_call_rounds", 0) or 0)
    max_tool_call_rounds = int(critic_payload.get("max_tool_call_rounds", MAX_TOOL_CALL_ROUNDS) or MAX_TOOL_CALL_ROUNDS)
    retry_count_total = int(critic_payload.get("retry_count_total", 0) or 0)
    retry_exhausted_count = int(critic_payload.get("retry_exhausted_count", 0) or 0)
    cache_hit_count = int(critic_payload.get("cache_hit_count", 0) or 0)

    if decision == "accept":
        status = "completed"
        title = "证据通过，答案可交付"
        badge = "ACCEPT"
    elif decision == "refuse":
        if has_some_evidence:
            status = "running"
            title = "证据存在争议，建议复核"
            badge = "REVIEW"
        else:
            status = "failed"
            title = "证据不足，触发拒答保护"
            badge = "REFUSE"
    else:
        status = "running"
        title = "证据仍需补强，继续检索"
        badge = "REVISE"

    fit_score = float(critic_payload.get("confidence", 0.0) or 0.0)
    fit_percent = round(fit_score * 100)
    if fit_score >= 0.75:
        fit_level = "高"
        fit_tone = "good"
    elif fit_score >= 0.55:
        fit_level = "中"
        fit_tone = "warn"
    else:
        fit_level = "低"
        fit_tone = "risk"

    summary = _normalize_whitespace(str(critic_payload.get("reason", "")))
    summary_prefix = "注：证据分用于衡量检索充分性，不等于答案正确率。"
    summary = f"{summary_prefix} {summary}".strip()
    if decision == "refuse" and has_some_evidence:
        summary = (
            "该结论属于保守判定（证据匹配度偏弱，不代表答案必错）。 "
            + summary
        ).strip()

    return _build_reasoning_stage_payload(
        stage_key="critic",
        status=status,
        title=title,
        summary=summary,
        mode=str(critic_payload.get("mode", "llm")),
        badge=badge,
        metrics=[
            {
                "label": "检索轮次",
                "value": f"{retrieval_rounds}/{max_retrieval_rounds}",
                "tone": "warn" if retrieval_rounds >= max_retrieval_rounds else "neutral",
            },
            {
                "label": "工具轮次",
                "value": f"{tool_call_rounds}/{max_tool_call_rounds}",
                "tone": "warn" if tool_call_rounds >= max_tool_call_rounds else "neutral",
            },
            {
                "label": "失败重试",
                "value": f"{retry_count_total} (耗尽 {retry_exhausted_count})",
                "tone": "risk" if retry_exhausted_count > 0 else ("warn" if retry_count_total > 0 else "neutral"),
            },
            {
                "label": "缓存命中",
                "value": str(cache_hit_count),
                "tone": "good" if cache_hit_count > 0 else "neutral",
            },
            {
                "label": "证据等级",
                "value": fit_level,
                "tone": fit_tone,
            },
            {
                "label": "匹配参考",
                "value": f"{fit_percent}% (非准确率)",
            },
            {
                "label": "证据",
                "value": f"{evidence_items} / {int(critic_payload.get('distinct_files', 0) or 0)}",
            },
            {
                "label": "命中",
                "value": f"{int(critic_payload.get('query_term_hits', 0) or 0)}/{int(critic_payload.get('query_terms', 0) or 0)}",
            },
        ],
    )


def _extract_query_signals(user_query: str) -> List[List[str]]:
    query = (user_query or "").strip().lower()
    if not query:
        return []

    signals: List[List[str]] = []

    for term in re.findall(r"[a-z0-9]{3,}", query):
        signals.append([term])

    for span in re.findall(r"[\u4e00-\u9fff]{2,}", query):
        variants: List[str] = [span]
        max_n = min(4, len(span))
        for n in range(max_n, 1, -1):
            for i in range(0, len(span) - n + 1):
                variants.append(span[i : i + n])

        compact: List[str] = []
        seen_variant = set()
        for variant in variants:
            token = (variant or "").strip()
            if not token or token in seen_variant:
                continue
            seen_variant.add(token)
            compact.append(token)
        if compact:
            signals.append(compact[:6])

    deduped: List[List[str]] = []
    seen_signal = set()
    for signal in signals:
        key = "||".join(signal)
        if key in seen_signal:
            continue
        seen_signal.add(key)
        deduped.append(signal)
    return deduped[:12]


def _is_uncertain_answer(answer: str) -> bool:
    text = _normalize_whitespace(answer).lower()
    if not text:
        return True
    return any(keyword in text for keyword in UNCERTAIN_ANSWER_KEYWORDS)


def _evaluate_retrieval_critic_metrics(
    *,
    user_query: str,
    candidate_answer: str,
    evidence_pool: List[Dict[str, str]],
) -> Dict[str, Any]:
    evidence_items = [
        item
        for item in evidence_pool
        if item.get("file_path", "").strip() and item.get("snippet", "").strip()
    ]
    evidence_count = len(evidence_items)
    distinct_files = len({item["file_path"].strip() for item in evidence_items})
    query_signals = _extract_query_signals(user_query)

    snippets_lower = [str(item.get("snippet", "")).strip().lower() for item in evidence_items]
    combined_text = " ".join(snippets_lower)
    query_term_hits = sum(
        1 for variants in query_signals if any(term in combined_text for term in variants)
    )
    matched_evidence_items = (
        sum(
            1
            for snippet in snippets_lower
            if any(any(term in snippet for term in variants) for variants in query_signals)
        )
        if query_signals
        else evidence_count
    )

    hit_ratio = 1.0 if not query_signals else query_term_hits / max(len(query_signals), 1)
    snippet_match_ratio = (
        matched_evidence_items / max(evidence_count, 1)
        if evidence_count
        else 0.0
    )
    file_diversity_ratio = (
        min(distinct_files / (1.0 if evidence_count <= 2 else 2.0), 1.0)
        if evidence_count
        else 0.0
    )
    uncertain_answer = _is_uncertain_answer(candidate_answer)
    confidence = hit_ratio * 0.50 + snippet_match_ratio * 0.30 + file_diversity_ratio * 0.20
    if uncertain_answer:
        confidence *= 0.85

    return {
        "evidence_items": evidence_count,
        "distinct_files": distinct_files,
        "total_snippet_chars": sum(len(snippet) for snippet in snippets_lower),
        "query_term_hits": int(query_term_hits),
        "query_terms": len(query_signals),
        "matched_evidence_items": int(matched_evidence_items),
        "uncertain_answer": bool(uncertain_answer),
        "confidence": round(max(0.0, min(confidence, 1.0)), 2),
    }


def _build_rule_based_critic_decision(metrics: Dict[str, Any]) -> Dict[str, Any]:
    evidence_items = int(metrics.get("evidence_items", 0) or 0)
    distinct_files = int(metrics.get("distinct_files", 0) or 0)
    query_terms = int(metrics.get("query_terms", 0) or 0)
    query_term_hits = int(metrics.get("query_term_hits", 0) or 0)
    confidence = float(metrics.get("confidence", 0.0) or 0.0)
    uncertain_answer = bool(metrics.get("uncertain_answer", False))
    required_hits = min(CRITIC_MIN_QUERY_TERM_HITS, query_terms) if query_terms > 0 else 0

    adaptive_accept_confidence = float(CRITIC_ACCEPT_CONFIDENCE)
    if not uncertain_answer:
        # Dense multi-file evidence should not be rejected by an overly strict
        # global threshold in simple/list questions.
        if evidence_items >= 6 and distinct_files >= 3 and query_term_hits >= 1:
            adaptive_accept_confidence = min(adaptive_accept_confidence, 0.58)
        elif evidence_items >= 4 and distinct_files >= 2 and query_term_hits >= 1:
            adaptive_accept_confidence = min(adaptive_accept_confidence, 0.62)
        elif evidence_items >= 2 and query_term_hits >= 1:
            adaptive_accept_confidence = min(adaptive_accept_confidence, 0.66)

    blockers: List[str] = []
    if evidence_items < CRITIC_MIN_EVIDENCE_ITEMS:
        blockers.append(f"证据条数不足({evidence_items}/{CRITIC_MIN_EVIDENCE_ITEMS})")
    if distinct_files < CRITIC_MIN_DISTINCT_FILES:
        blockers.append(f"来源文件不足({distinct_files}/{CRITIC_MIN_DISTINCT_FILES})")
    if query_terms > 0:
        if query_term_hits < required_hits:
            blockers.append(f"问题关键词命中不足({query_term_hits}/{required_hits})")
    if confidence < adaptive_accept_confidence:
        blockers.append(
            f"证据分偏低({round(confidence * 100)}%<{round(adaptive_accept_confidence * 100)}%)"
        )

    if not blockers:
        return {
            "decision": "accept",
            "stop": True,
            "reason": "证据池满足停机门槛，可停止检索。",
            "evidence_sufficient": True,
        }

    if evidence_items <= 0:
        reason = "证据池为空，需继续检索。"
        decision = "refuse"
    else:
        reason = "证据仍不足：" + "；".join(blockers)
        decision = "revise"

    return {
        "decision": decision,
        "stop": False,
        "reason": reason,
        "evidence_sufficient": False,
    }


def _apply_critic_rule_guardrail(critic_payload: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(critic_payload or {})
    rule = _build_rule_based_critic_decision(enriched)

    llm_decision = _normalize_whitespace(str(enriched.get("decision", ""))).lower()
    llm_stop = bool(enriched.get("stop", False))
    rule_decision = str(rule.get("decision", "revise"))
    rule_stop = bool(rule.get("stop", False))
    rule_reason = _normalize_whitespace(str(rule.get("reason", "")))
    llm_reason = _normalize_whitespace(str(enriched.get("reason", "")))

    if llm_decision != rule_decision or llm_stop != rule_stop:
        mode = _normalize_whitespace(str(enriched.get("mode", "llm"))) or "llm"
        enriched["mode"] = f"{mode}+rule"
        if rule_reason:
            if llm_reason:
                enriched["reason"] = f"{llm_reason}; 规则判定: {rule_reason}"
            else:
                enriched["reason"] = rule_reason
    elif not llm_reason and rule_reason:
        enriched["reason"] = rule_reason

    enriched["decision"] = rule_decision
    enriched["stop"] = rule_stop
    enriched["evidence_sufficient"] = bool(rule.get("evidence_sufficient", False))
    enriched["rule_reason"] = rule_reason
    return enriched


def _extract_candidate_paths(search_output: str) -> List[str]:
    parsed = _extract_json_object(search_output or "")
    candidates = parsed.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    paths: List[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        if path:
            paths.append(path)
    return paths


def _extract_evidence_entries(section_output: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for match in EVIDENCE_BLOCK_PATTERN.finditer(section_output or ""):
        file_path = _normalize_whitespace(match.group("path"))
        snippet = _normalize_whitespace(match.group("body"))
        if not file_path or not snippet:
            continue
        if len(snippet) > MAX_EVIDENCE_SNIPPET_LENGTH:
            snippet = snippet[:MAX_EVIDENCE_SNIPPET_LENGTH].rstrip() + "..."
        snippet = snippet.replace('"', "'")
        image_path = _extract_image_path_from_note(file_path)
        evidence_item = {"file_path": file_path, "snippet": snippet}
        if image_path:
            evidence_item["image_path"] = image_path
        entries.append(evidence_item)
    return entries


def _extract_image_path_from_note(note_rel_path: str) -> Optional[str]:
    rel_path = (note_rel_path or "").strip().replace("\\", "/").lstrip("/")
    if not rel_path.lower().startswith("image-notes/"):
        return None
    if not rel_path.lower().endswith(".md"):
        return None

    kb_root = Path(settings.knowledge_base_chunks).resolve()
    try:
        note_path = (kb_root / rel_path).resolve()
        note_path.relative_to(kb_root)
    except Exception:
        return None
    if not note_path.is_file():
        return None

    try:
        content = note_path.read_text(encoding="utf-8")
    except Exception:
        return None
    match = IMAGE_NOTE_LINE_PATTERN.search(content)
    if not match:
        return None

    image_rel = match.group(1).strip().replace("\\", "/").lstrip("/")
    if not image_rel:
        return None
    try:
        image_path = (kb_root / image_rel).resolve()
        image_path.relative_to(kb_root)
    except Exception:
        return None
    if not image_path.is_file():
        return None
    return image_rel


def _merge_evidence_pool(
    current_pool: List[Dict[str, str]], new_entries: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    merged = list(current_pool)
    seen = {(item.get("file_path", ""), item.get("snippet", "")) for item in merged}
    for entry in new_entries:
        key = (entry.get("file_path", ""), entry.get("snippet", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
    return merged


def _strip_existing_evidence_section(answer: str) -> str:
    if not answer:
        return ""
    text = answer.strip()
    text = EVIDENCE_SECTION_PATTERN.sub("", text)
    return text.strip()


def _format_answer_with_evidence(answer: str, evidence_pool: List[Dict[str, str]]) -> str:
    if not evidence_pool:
        return FALLBACK_NO_EVIDENCE_ANSWER

    clean_answer = _strip_existing_evidence_section(answer)
    if not clean_answer:
        clean_answer = "基于检索到的知识库证据，结论如下。"

    lines = [clean_answer, "", "### 证据"]
    for idx, item in enumerate(evidence_pool[:MAX_FINAL_EVIDENCE_ITEMS], start=1):
        file_path = item.get("file_path", "").strip() or "unknown"
        snippet = item.get("snippet", "").strip() or "(空片段)"
        image_path = item.get("image_path", "").strip()
        lines.append(f"{idx}. 来源文件: `{file_path}`")
        if image_path:
            lines.append(f"   原始图片: `{image_path}`")
        lines.append(f"   证据片段: \"{snippet}\"")

    return "\n".join(lines).strip()


def _format_budget_guard_answer(reason: str, evidence_pool: List[Dict[str, str]]) -> str:
    normalized_reason = _normalize_whitespace(reason) or "budget guard triggered"
    if evidence_pool:
        return _format_answer_with_evidence(
            f"已触发预算保护（{normalized_reason}）。以下为当前预算内可支持的阶段性结论。",
            evidence_pool,
        )
    return (
        f"已触发预算保护（{normalized_reason}），且当前证据不足。"
        "请适度提高 token/时延上限后重试。"
    )


def _format_insufficient_evidence_answer(stop_reason: str) -> str:
    reason = _normalize_whitespace(stop_reason)
    if not reason:
        return FALLBACK_INSUFFICIENT_EVIDENCE_ANSWER
    return (
        "我无法给出有证据支撑的回答。"
        f"{reason}，证据仍不足以支持结论。"
    )


def _build_retrieval_critic_payload(
    *,
    stop: bool,
    decision: str,
    reason: str,
    mode: str,
    user_query: str,
    candidate_answer: str,
    evidence_pool: List[Dict[str, str]],
) -> Dict[str, Any]:
    metrics = _evaluate_retrieval_critic_metrics(
        user_query=user_query,
        candidate_answer=candidate_answer,
        evidence_pool=evidence_pool,
    )

    return {
        "type": "retrieval_critic",
        "decision": decision,
        "stop": bool(stop),
        "mode": mode,
        "reason": _normalize_whitespace(reason),
        "confidence": float(metrics.get("confidence", 0.0)),
        "evidence_items": int(metrics.get("evidence_items", 0)),
        "distinct_files": int(metrics.get("distinct_files", 0)),
        "matched_evidence_items": int(metrics.get("matched_evidence_items", 0)),
        "total_snippet_chars": int(metrics.get("total_snippet_chars", 0)),
        "uncertain_answer": bool(metrics.get("uncertain_answer", False)),
        "query_term_hits": int(metrics.get("query_term_hits", 0)),
        "query_terms": int(metrics.get("query_terms", 0)),
    }


def _build_tool_call(call_id: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        iv = int(value)
    except Exception:
        iv = default
    return max(low, min(iv, high))


def _guess_question_route(user_query: str) -> Dict[str, Any]:
    query = _normalize_whitespace(user_query)
    if not query:
        return dict(DEFAULT_QUESTION_ROUTE)

    lower_query = query.lower()
    followup_markers = (
        "上面",
        "前面",
        "刚才",
        "上一轮",
        "继续",
        "接着",
    )
    deep_markers = (
        "对比",
        "比较",
        "分析",
        "总结",
        "方案",
        "优缺点",
        "原因",
        "架构",
        "趋势",
        "tradeoff",
        "compare",
        "analysis",
        "strategy",
    )

    is_followup = any(marker in query for marker in followup_markers) or bool(
        re.search(r"\b(previous|above|earlier|follow[- ]?up|former|latter)\b", lower_query)
    )
    is_deep = any(marker in query for marker in deep_markers) or len(query) >= 90

    if is_deep:
        return {
            "route": "deep_analysis",
            "history_mode": "full",
            "max_iterations": 12,
            "reason": "heuristic_deep",
        }
    if is_followup:
        return {
            "route": "contextual_followup",
            "history_mode": "full",
            "max_iterations": 10,
            "reason": "heuristic_followup",
        }
    return {
        "route": "simple_fact",
        "history_mode": "latest_only",
        "max_iterations": 6,
        "reason": "heuristic_simple",
    }


def _normalize_question_route_payload(
    payload: Dict[str, Any],
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    route = _normalize_whitespace(str(payload.get("route", fallback.get("route", "")))).lower()
    if route not in {"simple_fact", "contextual_followup", "deep_analysis"}:
        route = str(fallback.get("route", "contextual_followup"))

    history_mode = _normalize_whitespace(
        str(payload.get("history_mode", fallback.get("history_mode", "")))
    ).lower()
    if history_mode not in {"latest_only", "full"}:
        history_mode = "latest_only" if route == "simple_fact" else "full"

    default_max_iterations = int(fallback.get("max_iterations", 10) or 10)
    max_iterations = _coerce_int(
        payload.get("max_iterations"),
        default_max_iterations,
        4,
        16,
    )
    if route == "simple_fact":
        max_iterations = min(max_iterations, 8)
    elif route == "deep_analysis":
        max_iterations = max(max_iterations, 10)

    reason = _clip_text(str(payload.get("reason", fallback.get("reason", ""))), 80)
    if not reason:
        reason = str(fallback.get("reason", "default"))

    return {
        "route": route,
        "history_mode": history_mode,
        "max_iterations": max_iterations,
        "reason": reason,
    }


def _select_messages_by_history_mode(
    history_messages: List[Dict[str, str]],
    history_mode: str,
) -> List[Dict[str, str]]:
    if history_mode != "latest_only":
        return list(history_messages or [])

    latest_user: Optional[Dict[str, str]] = None
    for message in reversed(history_messages or []):
        role = _normalize_whitespace(str(message.get("role", ""))).lower()
        content = str(message.get("content", ""))
        if role == "user" and _normalize_whitespace(content):
            latest_user = {"role": "user", "content": content}
            break

    if latest_user:
        return [latest_user]
    if history_messages:
        last_item = history_messages[-1]
        return [
            {
                "role": str(last_item.get("role", "user")),
                "content": str(last_item.get("content", "")),
            }
        ]
    return []


def _safe_parse_tool_args(arguments: str) -> Dict[str, Any]:
    return _extract_json_object(arguments or "")


def _extract_tool_runtime_stats(tool_results: List[Dict[str, Any]]) -> Dict[str, int]:
    cache_hits = 0
    retry_total = 0
    retry_exhausted = 0
    for result in tool_results or []:
        if not isinstance(result, dict):
            continue
        meta = result.get("meta", {})
        if not isinstance(meta, dict):
            continue
        if bool(meta.get("cache_hit", False)):
            cache_hits += 1
        retry_total += int(meta.get("retry_count", 0) or 0)
        if bool(meta.get("retry_exhausted", False)):
            retry_exhausted += 1
    return {
        "cache_hits": cache_hits,
        "retry_total": retry_total,
        "retry_exhausted": retry_exhausted,
    }


def _append_runtime_metrics(
    critic_payload: Dict[str, Any],
    *,
    retrieval_rounds: int,
    tool_call_rounds: int,
    auto_retrieval_rounds: int,
    retry_total: int,
    retry_exhausted: int,
    cache_hits: int,
    stop_reason: str = "",
) -> Dict[str, Any]:
    enriched = dict(critic_payload or {})
    enriched.update(
        {
            "retrieval_rounds": max(0, int(retrieval_rounds)),
            "max_retrieval_rounds": int(MAX_RETRIEVAL_ROUNDS),
            "tool_call_rounds": max(0, int(tool_call_rounds)),
            "max_tool_call_rounds": int(MAX_TOOL_CALL_ROUNDS),
            "auto_retrieval_rounds": max(0, int(auto_retrieval_rounds)),
            "max_auto_retrieval_rounds": int(MAX_AUTO_RETRIEVAL_ROUNDS),
            "retry_count_total": max(0, int(retry_total)),
            "retry_exhausted_count": max(0, int(retry_exhausted)),
            "cache_hit_count": max(0, int(cache_hits)),
        }
    )
    reason = _normalize_whitespace(stop_reason)
    if reason:
        current_reason = _normalize_whitespace(str(enriched.get("reason", "")))
        if current_reason:
            enriched["reason"] = f"{current_reason}; {reason}"
        else:
            enriched["reason"] = reason
    return enriched


async def _llm_run_retrieval_critic(
    provider: LLMProvider,
    *,
    user_query: str,
    candidate_answer: str,
    evidence_pool: List[Dict[str, str]],
) -> Dict[str, Any]:
    evidence_preview: List[Dict[str, str]] = []
    for item in evidence_pool[:4]:
        evidence_preview.append(
            {
                "file_path": str(item.get("file_path", "")),
                "snippet": str(item.get("snippet", ""))[:220],
            }
        )

    critic_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict retrieval critic for RAG. "
                "Return strict JSON only: "
                "{\"decision\":\"accept|revise|refuse\",\"stop\":bool,\"reason\":string}. "
                "Set decision=accept and stop=true only when evidence is sufficient and directly supports the answer. "
                "If evidence is weak/incomplete, set revise + stop=false. "
                "If no usable evidence exists, set refuse + stop=false."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": user_query,
                    "candidate_answer": candidate_answer,
                    "evidence": evidence_preview,
                },
                ensure_ascii=False,
            ),
        },
    ]

    raw_text = ""
    try:
        async for chunk_str in provider.chat_completion(
            messages=critic_messages,
            tools=None,
            stream=False,
        ):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw_text += str(chunk.get("content") or "")
    except Exception as exc:
        return _apply_critic_rule_guardrail(
            _build_retrieval_critic_payload(
                stop=False,
                decision="revise",
                reason=f"critic llm unavailable: {_clip_text(str(exc), 90)}",
                mode="llm_error",
                user_query=user_query,
                candidate_answer=candidate_answer,
                evidence_pool=evidence_pool,
            )
        )

    parsed = _extract_json_object(raw_text)

    raw_decision = _normalize_whitespace(str(parsed.get("decision", ""))).lower()
    if raw_decision not in {"accept", "revise", "refuse"}:
        raw_decision = ""

    raw_stop = parsed.get("stop")
    if not isinstance(raw_stop, bool):
        raw_stop = None

    parse_ok = bool(raw_decision) or isinstance(raw_stop, bool)
    reason = _normalize_whitespace(str(parsed.get("reason", "")))
    if not reason:
        reason = _clip_text(raw_text, 120) or "critic output invalid"

    if not parse_ok:
        decision = "revise"
        stop = False
        mode = "llm_parse_error"
        reason = "critic schema invalid, fallback to rule critic"
    else:
        decision = raw_decision or ("accept" if raw_stop else "revise")
        stop = decision == "accept"
        mode = "llm"

    return _apply_critic_rule_guardrail(
        _build_retrieval_critic_payload(
            stop=stop,
            decision=decision,
            reason=reason,
            mode=mode,
            user_query=user_query,
            candidate_answer=candidate_answer,
            evidence_pool=evidence_pool,
        )
    )


async def _llm_route_question(
    provider: LLMProvider,
    *,
    latest_user_query: str,
    message_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    fallback = _guess_question_route(latest_user_query)
    query = _normalize_whitespace(latest_user_query)
    if not query:
        return _normalize_question_route_payload({}, fallback)

    history_preview: List[Dict[str, str]] = []
    for item in (message_history or [])[-6:]:
        role = _normalize_whitespace(str(item.get("role", ""))).lower()
        if role not in {"user", "assistant"}:
            continue
        content = _clip_text(str(item.get("content", "")), 120)
        if not content:
            continue
        history_preview.append({"role": role, "content": content})

    router_messages = [
        {
            "role": "system",
            "content": (
                "You are a question router for a multi-turn RAG assistant. "
                "Return strict JSON only: "
                "{\"route\":\"simple_fact|contextual_followup|deep_analysis\","
                "\"history_mode\":\"latest_only|full\","
                "\"max_iterations\":int,"
                "\"reason\":string}. "
                "Rules: "
                "1) simple_fact = standalone, direct lookup/list question, prefer latest_only with lower iterations. "
                "2) contextual_followup = references prior turns (such as 上面/刚才/previous/above), must use full history. "
                "3) deep_analysis = multi-step comparison/synthesis/planning question, use full history and higher iterations. "
                "4) If uncertain, choose contextual_followup."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "latest_user_query": query,
                    "recent_dialogue": history_preview,
                },
                ensure_ascii=False,
            ),
        },
    ]

    raw_text = ""
    try:
        async for chunk_str in provider.chat_completion(
            messages=router_messages,
            tools=None,
            stream=False,
        ):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw_text += str(chunk.get("content") or "")
    except Exception:
        return _normalize_question_route_payload({}, fallback)

    parsed = _extract_json_object(raw_text)
    return _normalize_question_route_payload(parsed, fallback)


async def _llm_expand_retrieval_query(provider: LLMProvider, user_query: str) -> Dict[str, Any]:
    fallback = {
        "expanded_query": (user_query or "").strip(),
        "image_intent": False,
        "hints": [],
    }
    query = (user_query or "").strip()
    if not query:
        return fallback

    messages = [
        {
            "role": "system",
            "content": (
                "You optimize retrieval queries for bilingual (Chinese/English) lexical search. "
                "Return strict JSON only: "
                "{\"expanded_query\": string, \"image_intent\": boolean, \"hints\": string[]}. "
                "Rules: "
                "1) Preserve original intent. "
                "2) Add concise Chinese+English synonyms and key entities for retrieval. "
                "3) If query is about image/photo/portrait/OCR/scene, set image_intent=true. "
                "4) expanded_query should be one line."
            ),
        },
        {"role": "user", "content": query},
    ]

    raw_text = ""
    try:
        async for chunk_str in provider.chat_completion(messages=messages, tools=None, stream=False):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw_text += str(chunk.get("content") or "")
    except Exception:
        return fallback

    parsed = _extract_json_object(raw_text)
    expanded_query = _normalize_whitespace(str(parsed.get("expanded_query") or query))
    image_intent = bool(parsed.get("image_intent"))
    hints: List[str] = []
    raw_hints = parsed.get("hints")
    if isinstance(raw_hints, list):
        for item in raw_hints:
            hint = _normalize_whitespace(str(item))
            if hint and hint not in hints:
                hints.append(hint)

    return {
        "expanded_query": expanded_query or query,
        "image_intent": image_intent,
        "hints": hints[:16],
    }


async def _llm_refine_answer_with_evidence(
    provider: LLMProvider,
    *,
    user_query: str,
    candidate_answer: str,
    evidence_pool: List[Dict[str, str]],
) -> str:
    evidence_preview: List[Dict[str, str]] = []
    for item in evidence_pool[:6]:
        evidence_preview.append(
            {
                "file_path": str(item.get("file_path", "")),
                "snippet": str(item.get("snippet", ""))[:220],
            }
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are refining an answer in a RAG pipeline. "
                "Keep answer concise, factual, and grounded in provided evidence only. "
                "Do not invent details. Return plain answer text (no markdown headings)."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": user_query,
                    "candidate_answer": candidate_answer,
                    "evidence": evidence_preview,
                },
                ensure_ascii=False,
            ),
        },
    ]

    refined_text = ""
    try:
        async for chunk_str in provider.chat_completion(messages=messages, tools=None, stream=False):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                refined_text += str(chunk.get("content") or "")
    except Exception:
        return _normalize_whitespace(candidate_answer)

    normalized = _normalize_whitespace(refined_text)
    if normalized:
        return normalized
    return _normalize_whitespace(candidate_answer)


def _slugify_filename(value: str, fallback: str = "speaker") -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (value or "").strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _to_local_iso(raw_value: Optional[str]) -> str:
    if not raw_value:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    raw = raw_value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except Exception:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().isoformat(timespec="seconds")


def _guess_image_suffix(filename: str, content_type: str, image_format: str) -> str:
    for candidate in [
        Path(filename or "").suffix.lower(),
        mimetypes.guess_extension(content_type or "") or "",
        f".{(image_format or '').lower()}",
    ]:
        if candidate in ALLOWED_IMAGE_SUFFIXES:
            return candidate
    return ".png"


async def _read_upload_image_bytes(file: UploadFile) -> bytes:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty.")
    if len(image_bytes) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds size limit ({MAX_IMAGE_UPLOAD_BYTES} bytes).",
        )
    return image_bytes


def _fallback_voice_summary(transcript: str) -> str:
    normalized = _normalize_whitespace(transcript)
    if not normalized:
        return ""
    if len(normalized) <= 60:
        return normalized
    return normalized[:60].rstrip() + "..."


def _parse_voice_draft_response(raw_text: str) -> tuple[str, str, str]:
    """Parse draft response from multiple possible formats.

    Returns: (polished_text, summary, parse_note)
    """
    raw = (raw_text or "").strip()
    if not raw:
        return "", "", "empty response"

    # 1) Direct JSON object.
    parsed = _extract_json_object(raw)
    polished = _normalize_whitespace(str(parsed.get("polished_text") or ""))
    summary = _normalize_whitespace(str(parsed.get("summary") or ""))
    if polished or summary:
        return polished, summary, "json"

    # 2) JSON inside fenced code block.
    fence = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fence:
        parsed = _extract_json_object(fence.group(1))
        polished = _normalize_whitespace(str(parsed.get("polished_text") or ""))
        summary = _normalize_whitespace(str(parsed.get("summary") or ""))
        if polished or summary:
            return polished, summary, "json-fence"

    # 3) Tagged format.
    tag_pol = re.search(r"<polished_text>\s*([\s\S]*?)\s*</polished_text>", raw, flags=re.IGNORECASE)
    tag_sum = re.search(r"<summary>\s*([\s\S]*?)\s*</summary>", raw, flags=re.IGNORECASE)
    polished = _normalize_whitespace(tag_pol.group(1) if tag_pol else "")
    summary = _normalize_whitespace(tag_sum.group(1) if tag_sum else "")
    if polished or summary:
        return polished, summary, "xml-tags"

    # 4) Label format: English.
    m_en = re.search(
        r"POLISHED_TEXT\s*[:：]\s*([\s\S]*?)(?:\n+\s*SUMMARY\s*[:：]\s*([\s\S]*))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_en:
        polished = _normalize_whitespace(m_en.group(1) or "")
        summary = _normalize_whitespace(m_en.group(2) or "")
        if polished or summary:
            return polished, summary, "labels-en"

    # 5) Label format: Chinese.
    m_zh = re.search(
        r"(?:润色文本|纠正文本|草稿)\s*[:：]\s*([\s\S]*?)(?:\n+\s*(?:摘要|总结)\s*[:：]\s*([\s\S]*))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_zh:
        polished = _normalize_whitespace(m_zh.group(1) or "")
        summary = _normalize_whitespace(m_zh.group(2) or "")
        if polished or summary:
            return polished, summary, "labels-zh"

    # 6) Free-form fallback: treat entire output as polished text.
    stripped = re.sub(r"```[\s\S]*?```", "", raw).strip()
    polished = _normalize_whitespace(stripped)
    if polished:
        return polished, "", "free-form"

    return "", "", "unparsed"


async def _llm_voice_summary(
    provider: LLMProvider,
    transcript: str,
    polished_text: str,
    author: str,
) -> str:
    summary_messages = [
        {
            "role": "system",
            "content": (
                "Summarize the core meaning in ONE concise sentence. "
                "Output plain text only. No markdown, no bullet points, no labels."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "author": author,
                    "transcript": transcript,
                    "polished_text": polished_text,
                },
                ensure_ascii=False,
            ),
        },
    ]

    raw = ""
    try:
        async for chunk_str in provider.chat_completion(
            messages=summary_messages,
            tools=None,
            stream=False,
        ):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw += str(chunk.get("content") or "")
    except Exception:
        return ""

    plain = _normalize_whitespace(raw)
    plain = re.sub(r"^(?:summary|摘要|总结)\s*[:：]\s*", "", plain, flags=re.IGNORECASE)
    return plain


async def _llm_voice_draft(
    provider: LLMProvider,
    transcript: str,
    author: str,
) -> Dict[str, str]:
    fallback_polished = _normalize_whitespace(transcript)
    fallback_summary = _fallback_voice_summary(fallback_polished)

    draft_messages = [
        {
            "role": "system",
            "content": (
                "You are a voice note editor. Rewrite colloquial transcript into clean readable text "
                "without changing facts, then provide one-sentence summary. "
                "Preferred output format (plain text, no markdown):\n"
                "POLISHED_TEXT: <rewritten text>\n"
                "SUMMARY: <one sentence>\n"
                "If you return JSON, schema must be {\"polished_text\": string, \"summary\": string}. "
                "Keep original facts, remove obvious fillers and disfluencies, and keep the same language "
                "as transcript. Summary should express the core meaning in one concise sentence."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "author": author,
                    "transcript": transcript,
                },
                ensure_ascii=False,
            ),
        },
    ]

    raw_text = ""
    try:
        async for chunk_str in provider.chat_completion(
            messages=draft_messages,
            tools=None,
            stream=False,
        ):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw_text += str(chunk.get("content") or "")
    except Exception:
        return {
            "polished_text": fallback_polished,
            "summary": fallback_summary,
            "warning": "LLM unavailable, returned fallback draft.",
        }

    polished_text, summary, parse_note = _parse_voice_draft_response(raw_text)

    warning = ""
    if not polished_text:
        polished_text = fallback_polished
        warning = "LLM draft parse failed, used transcript directly."
    if not summary:
        summary = await _llm_voice_summary(
            provider=provider,
            transcript=transcript,
            polished_text=polished_text,
            author=author,
        )
    if not summary:
        summary = _fallback_voice_summary(polished_text)
        warning = warning or "LLM summary parse failed, used heuristic summary."

    result = {
        "polished_text": polished_text,
        "summary": summary,
    }
    if warning:
        result["warning"] = warning
    return result


async def _plan_retrieval_route(provider: LLMProvider, user_query: str) -> Dict[str, int]:
    plan = dict(DEFAULT_ROUTE_PLAN)
    if not user_query.strip():
        return plan

    router_messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval router for an agentic RAG pipeline. "
                "Return strict JSON only, no markdown, no prose. "
                "Schema: {\"top_k\": int, \"max_sections_per_file\": int, \"min_file_paths\": int}. "
                "Guidelines: list/comparison/coverage-heavy questions should use larger top_k and min_file_paths; "
                "factoid questions use smaller values."
            ),
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    raw_text = ""
    try:
        async for chunk_str in provider.chat_completion(
            messages=router_messages,
            tools=None,
            stream=False,
        ):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "content":
                raw_text += chunk.get("content", "")
    except Exception:
        return plan

    parsed = _extract_json_object(raw_text)
    plan["top_k"] = _coerce_int(parsed.get("top_k"), plan["top_k"], 3, 10)
    plan["max_sections_per_file"] = _coerce_int(
        parsed.get("max_sections_per_file"),
        plan["max_sections_per_file"],
        1,
        4,
    )
    plan["min_file_paths"] = _coerce_int(parsed.get("min_file_paths"), plan["min_file_paths"], 2, 8)
    return plan

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check"""
    providers = settings.list_available_providers()
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "providers": providers
    }

@app.get("/config")
async def get_config():
    """Get current configuration - dynamically return the current provider's model"""
    config = settings.get_provider_config(settings.api_provider)
    
    return {
        "default_provider": settings.api_provider,
        "default_model": config.get("model", "")
    }

@app.get("/api/config")
async def get_env_config():
    """Read the original content of .env file"""
    env_path = find_dotenv()
    if not env_path:
        raise HTTPException(status_code=404, detail=".env file not found")
    
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {"content": content}

@app.post("/api/config")
async def update_env_config(request: dict):
    """Directly save .env file content"""
    env_path = find_dotenv()
    if not env_path:
        raise HTTPException(status_code=404, detail=".env file not found")
    
    content = request.get("content", "")
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Config content cannot be empty")
    
    try:
        # Write content directly
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Reload environment variables
        load_dotenv(override=True)
        
        # Reinitialize settings object
        global settings
        from backend.config import Settings
        settings = Settings()
        
        return {"status": "success", "message": "Configuration updated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base/info", response_model=KnowledgeBaseInfo)
async def get_knowledge_base_info():
    try:
        summary = await knowledge_base.get_file_summary()
        file_tree = knowledge_base.list_files()
        return {
            "summary": summary,
            "file_tree": file_tree
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-prompt")
async def get_system_prompt():
    """Return the system prompt currently in use"""
    try:
        file_summary = await knowledge_base.get_file_summary()
        
        # Return corresponding system prompt based on configuration
        if settings.tool_calling_mode == "react":
            system_prompt = create_react_system_prompt(file_summary)
        else:
            system_prompt = create_system_prompt(file_summary)
        
        return {
            "system_prompt": system_prompt,
            "mode": settings.tool_calling_mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base/retrieve", response_model=FileRetrievalResponse)
async def retrieve_files(request: FileRetrievalRequest):
    try:
        content = await knowledge_base.retrieve_files(request.file_paths)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-base/file/{file_path:path}")
async def get_knowledge_base_file(file_path: str):
    kb_root = Path(settings.knowledge_base_chunks).resolve()
    normalized = (file_path or "").strip().replace("\\", "/").lstrip("/")
    if not normalized:
        raise HTTPException(status_code=400, detail="file_path cannot be empty")

    try:
        target = (kb_root / normalized).resolve()
        target.relative_to(kb_root)
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid file path")

    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(target))


@app.post("/voice/draft", response_model=VoiceDraftResponse)
async def draft_voice_note(request: VoiceDraftRequest):
    transcript = _normalize_whitespace(request.transcript)
    if not transcript:
        raise HTTPException(status_code=400, detail="transcript cannot be empty")

    author = _normalize_whitespace(request.author or "Unknown")
    provider = LLMProvider(provider=request.provider or settings.api_provider)
    draft = await _llm_voice_draft(provider=provider, transcript=transcript, author=author)

    return {
        "polished_text": draft.get("polished_text", transcript),
        "summary": draft.get("summary", _fallback_voice_summary(transcript)),
        "warning": draft.get("warning"),
    }


@app.post("/voice/ingest", response_model=VoiceIngestResponse)
async def ingest_voice_note(request: VoiceIngestRequest):
    transcript = (request.transcript or "").strip()
    summary = (request.summary or "").strip()
    author = (request.author or "Unknown").strip()
    source = (request.source or "Realtime voice input").strip()

    if not transcript:
        raise HTTPException(status_code=400, detail="transcript cannot be empty")
    if not summary:
        raise HTTPException(status_code=400, detail="summary cannot be empty")

    occurred_at = _to_local_iso(request.occurred_at)
    dt = datetime.fromisoformat(occurred_at)

    voice_dir = Path(settings.knowledge_base_chunks) / VOICE_NOTES_DIR / dt.strftime("%Y-%m-%d")
    voice_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify_filename(author, fallback="speaker")
    base_name = f"{dt.strftime('%H%M%S')}_{slug}"
    target_file = voice_dir / f"{base_name}.md"
    suffix = 2
    while target_file.exists():
        target_file = voice_dir / f"{base_name}_{suffix}.md"
        suffix += 1

    raw_transcript = (request.raw_transcript or "").strip()
    lines = [
        "# Voice Note",
        "",
        f"- Timestamp: `{occurred_at}`",
        f"- Author: `{author}`",
        f"- Source: `{source}`",
        f"- Summary: {summary}",
        "",
        "## Corrected Transcript",
        transcript,
    ]
    if raw_transcript:
        lines.extend(["", "## Raw Transcript", raw_transcript])

    content = "\n".join(lines).strip() + "\n"
    target_file.write_text(content, encoding="utf-8")

    rel_path = str(target_file.relative_to(Path(settings.knowledge_base_chunks))).replace("\\", "/")
    try:
        transcript_excerpt = _normalize_whitespace(transcript)[:240]
        summary_index_text = (
            f"{summary} | author:{author} | source:{source} | date:{dt.strftime('%Y-%m-%d')} "
            f"| content:{transcript_excerpt}"
        )
        knowledge_base.update_summary_entry(rel_path, summary_index_text)
    except Exception as e:
        print(f"[WARN] Failed to update summary index for {rel_path}: {e}")

    return {
        "status": "ok",
        "file_path": rel_path,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }


@app.post("/image/draft", response_model=ImageDraftResponse)
async def draft_image_note(
    file: UploadFile = File(...),
    author: str = Form(default="Unknown"),
    source: str = Form(default="Image upload"),
    provider: Optional[str] = Form(default=None),
):
    author_text = _normalize_whitespace(author or "Unknown")
    source_text = _normalize_whitespace(source or "Image upload")

    image_bytes = await _read_upload_image_bytes(file)
    try:
        width, height, image_format = inspect_image_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image_content_type = (file.content_type or "").strip() or f"image/{image_format.lower()}"
    image_suffix = _guess_image_suffix(file.filename or "", image_content_type, image_format)

    try:
        ocr_result = await run_paddle_ocr(image_bytes=image_bytes, suffix=image_suffix)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    provider_client = LLMProvider(provider=provider or settings.api_provider)
    try:
        vision_result = await run_vision_analysis(
            provider=provider_client,
            image_bytes=image_bytes,
            content_type=image_content_type,
            ocr_text=ocr_result.get("ocr_text", ""),
            author=author_text,
            source=source_text,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "ocr_text": ocr_result.get("ocr_text", ""),
        "visual_summary": vision_result.get("visual_summary", ""),
        "visual_description": vision_result.get("visual_description", ""),
        "tags": vision_result.get("tags", []),
        "retrieval_keywords": vision_result.get("retrieval_keywords", []),
        "ocr_line_count": int(ocr_result.get("ocr_line_count", 0)),
        "image_width": width,
        "image_height": height,
    }


@app.post("/image/ingest", response_model=ImageIngestResponse)
async def ingest_image_note(
    file: UploadFile = File(...),
    visual_summary: str = Form(...),
    visual_description: str = Form(...),
    ocr_text: str = Form(default=""),
    tags: str = Form(default=""),
    retrieval_keywords: str = Form(default=""),
    author: str = Form(default="Unknown"),
    source: str = Form(default="Image upload"),
    occurred_at: Optional[str] = Form(default=None),
):
    summary_text = (visual_summary or "").strip()
    description_text = (visual_description or "").strip()
    ocr_text_value = (ocr_text or "").strip()
    author_text = (author or "Unknown").strip()
    source_text = (source or "Image upload").strip()
    tags_list = parse_tags_text(tags)
    retrieval_keywords_list = parse_tags_text(retrieval_keywords)
    if not retrieval_keywords_list:
        retrieval_keywords_list = tags_list[:]

    if not summary_text:
        raise HTTPException(status_code=400, detail="visual_summary cannot be empty")
    if not description_text:
        raise HTTPException(status_code=400, detail="visual_description cannot be empty")

    image_bytes = await _read_upload_image_bytes(file)
    try:
        width, height, image_format = inspect_image_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image_content_type = (file.content_type or "").strip() or f"image/{image_format.lower()}"
    image_suffix = _guess_image_suffix(file.filename or "", image_content_type, image_format)

    occurred_local = _to_local_iso(occurred_at)
    dt = datetime.fromisoformat(occurred_local)

    kb_root = Path(settings.knowledge_base_chunks)
    day_dir = kb_root / IMAGE_NOTES_DIR / dt.strftime("%Y-%m-%d")
    assets_dir = day_dir / IMAGE_ASSETS_DIR
    assets_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify_filename(author_text, fallback="image")
    base_name = f"{dt.strftime('%H%M%S')}_{slug}"
    note_path = day_dir / f"{base_name}.md"
    image_path = assets_dir / f"{base_name}{image_suffix}"
    suffix_counter = 2
    while note_path.exists() or image_path.exists():
        note_path = day_dir / f"{base_name}_{suffix_counter}.md"
        image_path = assets_dir / f"{base_name}_{suffix_counter}{image_suffix}"
        suffix_counter += 1

    image_path.write_bytes(image_bytes)
    rel_image_path = str(image_path.relative_to(kb_root)).replace("\\", "/")

    tags_text = ", ".join(tags_list) if tags_list else "(none)"
    retrieval_keywords_text = (
        ", ".join(retrieval_keywords_list) if retrieval_keywords_list else "(none)"
    )
    lines = [
        "# Image Note",
        "",
        f"- Timestamp: `{occurred_local}`",
        f"- Author: `{author_text}`",
        f"- Source: `{source_text}`",
        f"- Image: `{rel_image_path}`",
        f"- Image Size: `{width}x{height}`",
        f"- Summary: {summary_text}",
        f"- Tags: {tags_text}",
        f"- Retrieval Keywords: {retrieval_keywords_text}",
        "",
        "## Visual Description",
        description_text,
        "",
        "## OCR Text",
        ocr_text_value or "(No text detected by OCR.)",
    ]
    note_content = "\n".join(lines).strip() + "\n"
    note_path.write_text(note_content, encoding="utf-8")

    rel_note_path = str(note_path.relative_to(kb_root)).replace("\\", "/")
    try:
        ocr_excerpt = _normalize_whitespace(ocr_text_value)[:260]
        desc_excerpt = _normalize_whitespace(description_text)[:260]
        summary_index_text = (
            f"{summary_text} | tags:{tags_text} | author:{author_text} | source:{source_text} "
            f"| retrieval_keywords:{retrieval_keywords_text} | date:{dt.strftime('%Y-%m-%d')} "
            f"| image:{rel_image_path} | desc:{desc_excerpt} "
            f"| ocr:{ocr_excerpt}"
        )
        knowledge_base.update_summary_entry(rel_note_path, summary_index_text)
    except Exception as e:
        print(f"[WARN] Failed to update summary index for {rel_note_path}: {e}")

    return {
        "status": "ok",
        "file_path": rel_note_path,
        "image_path": rel_image_path,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        provider = LLMProvider(provider=request.provider or settings.api_provider)
        budget_config = request.budget.model_dump() if request.budget else None
        budget_state = create_budget_state(budget_config)

        file_summary = await knowledge_base.get_file_summary()
        
        # Check whether to use function calling or ReAct mode
        use_react = settings.tool_calling_mode == "react"
        
        if use_react:
            system_prompt = create_react_system_prompt(file_summary)
        else:
            system_prompt = create_system_prompt(file_summary)
        
        request_history_messages = [msg.dict() for msg in request.messages]
        latest_user_query = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"),
            "",
        )
        question_route = await _llm_route_question(
            provider,
            latest_user_query=latest_user_query,
            message_history=request_history_messages,
        )
        route_name = str(question_route.get("route", "contextual_followup"))
        history_mode = str(question_route.get("history_mode", "full"))
        route_reason = _normalize_whitespace(str(question_route.get("reason", "")))
        route_max_iterations = _coerce_int(question_route.get("max_iterations"), 10, 4, 16)

        routed_history_messages = _select_messages_by_history_mode(
            request_history_messages,
            history_mode,
        )
        if not routed_history_messages and request_history_messages:
            routed_history_messages = [request_history_messages[-1]]

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(routed_history_messages)
        query_plan = await _llm_expand_retrieval_query(provider, latest_user_query)
        retrieval_query = query_plan.get("expanded_query") or latest_user_query
        hints = query_plan.get("hints") or []
        if isinstance(hints, list) and hints:
            retrieval_query = _normalize_whitespace(
                f"{retrieval_query} {' '.join(str(h) for h in hints[:10])}"
            )
        route_plan = await _plan_retrieval_route(provider, latest_user_query)
        if route_name == "simple_fact":
            route_plan["top_k"] = _coerce_int(route_plan.get("top_k"), 5, 4, 6)
            route_plan["min_file_paths"] = _coerce_int(route_plan.get("min_file_paths"), 3, 2, 4)
            route_plan["max_sections_per_file"] = _coerce_int(
                route_plan.get("max_sections_per_file"),
                2,
                1,
                2,
            )
        elif route_name == "deep_analysis":
            route_plan["top_k"] = _coerce_int(route_plan.get("top_k"), 8, 6, 10)
            route_plan["min_file_paths"] = _coerce_int(route_plan.get("min_file_paths"), 5, 4, 8)
            route_plan["max_sections_per_file"] = _coerce_int(
                route_plan.get("max_sections_per_file"),
                3,
                2,
                4,
            )
        if query_plan.get("image_intent"):
            route_plan["top_k"] = max(route_plan["top_k"], 8)
            route_plan["min_file_paths"] = max(route_plan["min_file_paths"], 5)
            route_plan["max_sections_per_file"] = max(route_plan["max_sections_per_file"], 2)
        if use_react:
            async def generate_response() -> AsyncIterator[str]:
                if ENABLE_PLAN_EXECUTE_STAGES:
                    plan_payload = _build_reasoning_stage_payload(
                        stage_key="plan",
                        status="completed",
                        title="检索计划已生成",
                        summary=_build_plan_stage_summary(latest_user_query),
                        mode="llm",
                        badge="LLM ROUTER",
                        metrics=[
                            {"label": "TopK", "value": str(route_plan["top_k"])},
                            {"label": "Files", "value": str(route_plan["min_file_paths"])},
                            {"label": "Sections", "value": str(route_plan["max_sections_per_file"])},
                            {"label": "Hints", "value": str(len(hints) if isinstance(hints, list) else 0)},
                            {"label": "Route", "value": route_name},
                            {"label": "History", "value": history_mode},
                            {"label": "MaxIter", "value": str(route_max_iterations)},
                            {"label": "CtxMsgs", "value": str(len(routed_history_messages))},
                            {"label": "RouteWhy", "value": route_reason or "n/a"},
                            {"label": "ImageIntent", "value": "yes" if query_plan.get("image_intent") else "no"},
                        ],
                    )
                    yield f"data: {json.dumps(plan_payload, ensure_ascii=False)}\n\n"
                async for chunk in handle_react_mode(
                    provider,
                    messages,
                    user_query=retrieval_query,
                    route_plan=route_plan,
                    budget_state=budget_state,
                    max_iterations=route_max_iterations,
                ):
                    yield chunk
                budget_summary_payload = build_budget_event(
                    budget_state,
                    event_type="budget_summary",
                )
                yield f"data: {json.dumps(budget_summary_payload, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
            return StreamingResponse(
                generate_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        tools = [
            create_search_paths_tool(),
            create_retrieve_sections_tool(),
        ]
        
        async def generate_response() -> AsyncIterator[str]:
            conversation_messages = messages.copy()
            max_iterations = _coerce_int(
                route_max_iterations + (2 if history_mode == "full" else 1),
                10,
                6,
                18,
            )
            iteration = 0
            allowed_tool_names = {"search_paths", "retrieve_sections"}
            has_search_step = False
            candidate_paths: List[str] = []
            evidence_pool: List[Dict[str, str]] = []
            forced_retrieval_attempted = False
            final_answer_sent = False
            budget_exceeded = False
            budget_guard_reason = ""
            latest_candidate_answer = ""
            control_stop_reason = ""
            retrieval_rounds = 0
            tool_call_rounds = 0
            retry_count_total = 0
            retry_exhausted_count = 0
            cache_hit_count = 0
            auto_retrieval_rounds = 0

            if ENABLE_PLAN_EXECUTE_STAGES:
                plan_payload = _build_reasoning_stage_payload(
                    stage_key="plan",
                    status="completed",
                    title="检索计划已生成",
                    summary=_build_plan_stage_summary(latest_user_query),
                    mode="llm",
                    badge="LLM ROUTER",
                    metrics=[
                        {"label": "TopK", "value": str(route_plan["top_k"])},
                        {"label": "Files", "value": str(route_plan["min_file_paths"])},
                        {"label": "Sections", "value": str(route_plan["max_sections_per_file"])},
                        {"label": "Hints", "value": str(len(hints) if isinstance(hints, list) else 0)},
                        {"label": "Route", "value": route_name},
                        {"label": "History", "value": history_mode},
                        {"label": "MaxIter", "value": str(route_max_iterations)},
                        {"label": "CtxMsgs", "value": str(len(routed_history_messages))},
                        {"label": "RouteWhy", "value": route_reason or "n/a"},
                        {"label": "ImageIntent", "value": "yes" if query_plan.get("image_intent") else "no"},
                    ],
                )
                yield f"data: {json.dumps(plan_payload, ensure_ascii=False)}\n\n"

            # Fast lane for simple standalone questions:
            # prefetch one deterministic retrieval round so evidence_pool is not
            # fully dependent on model tool-call quality.
            if route_name == "simple_fact":
                if tool_call_rounds < MAX_TOOL_CALL_ROUNDS and retrieval_rounds < MAX_RETRIEVAL_ROUNDS:
                    prefetch_top_k = _coerce_int(route_plan.get("top_k"), 5, 3, 10)
                    prefetch_search_call = _build_tool_call(
                        call_id="prefetch_search_0",
                        name="search_paths",
                        arguments={
                            "query": retrieval_query,
                            "top_k": prefetch_top_k,
                        },
                    )
                    if ENABLE_PLAN_EXECUTE_STAGES:
                        prefetch_stage = _build_reasoning_stage_payload(
                            stage_key="execute",
                            status="running",
                            title="简单问题预检索：路径",
                            summary="进入主循环前先做一轮确定性检索，减少多轮漂移。",
                            mode="router",
                            badge="PREFETCH",
                            metrics=[
                                {"label": "TopK", "value": str(prefetch_top_k)},
                                {"label": "Route", "value": route_name},
                            ],
                        )
                        yield f"data: {json.dumps(prefetch_stage, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [prefetch_search_call]})}\n\n"
                    prefetch_search_results = await process_tool_calls([prefetch_search_call])
                    tool_call_rounds += 1
                    retrieval_rounds += 1
                    prefetch_search_stats = _extract_tool_runtime_stats(prefetch_search_results)
                    cache_hit_count += prefetch_search_stats["cache_hits"]
                    retry_count_total += prefetch_search_stats["retry_total"]
                    retry_exhausted_count += prefetch_search_stats["retry_exhausted"]
                    yield f"data: {json.dumps({'type': 'tool_results', 'results': prefetch_search_results})}\n\n"
                    conversation_messages.append(
                        {"role": "assistant", "content": None, "tool_calls": [prefetch_search_call]}
                    )
                    conversation_messages.extend(prefetch_search_results)

                    candidate_paths = _extract_candidate_paths(
                        prefetch_search_results[0].get("content", "")
                    )
                    has_search_step = bool(candidate_paths)

                    if candidate_paths and tool_call_rounds < MAX_TOOL_CALL_ROUNDS:
                        prefetch_targets = candidate_paths[: max(route_plan["min_file_paths"], 2)]
                        prefetch_retrieve_call = _build_tool_call(
                            call_id="prefetch_retrieve_0",
                            name="retrieve_sections",
                            arguments={
                                "file_paths": prefetch_targets,
                                "query": retrieval_query,
                                "max_sections_per_file": route_plan["max_sections_per_file"],
                            },
                        )
                        if ENABLE_PLAN_EXECUTE_STAGES:
                            prefetch_stage = _build_reasoning_stage_payload(
                                stage_key="execute",
                                status="running",
                                title="简单问题预检索：片段",
                                summary="已拿到候选文件，预先回收证据片段。",
                                mode="router",
                                badge="PREFETCH",
                                metrics=[
                                    {"label": "Targets", "value": str(len(prefetch_targets))},
                                    {"label": "Sections", "value": str(route_plan["max_sections_per_file"])},
                                ],
                            )
                            yield f"data: {json.dumps(prefetch_stage, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [prefetch_retrieve_call]})}\n\n"
                        prefetch_retrieve_results = await process_tool_calls([prefetch_retrieve_call])
                        tool_call_rounds += 1
                        prefetch_retrieve_stats = _extract_tool_runtime_stats(prefetch_retrieve_results)
                        cache_hit_count += prefetch_retrieve_stats["cache_hits"]
                        retry_count_total += prefetch_retrieve_stats["retry_total"]
                        retry_exhausted_count += prefetch_retrieve_stats["retry_exhausted"]
                        yield f"data: {json.dumps({'type': 'tool_results', 'results': prefetch_retrieve_results})}\n\n"
                        conversation_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [prefetch_retrieve_call],
                            }
                        )
                        conversation_messages.extend(prefetch_retrieve_results)
                        evidence_pool = _merge_evidence_pool(
                            evidence_pool,
                            _extract_evidence_entries(prefetch_retrieve_results[0].get("content", "")),
                        )
            
            while iteration < max_iterations:
                if tool_call_rounds >= MAX_TOOL_CALL_ROUNDS:
                    control_stop_reason = (
                        f"工具调用轮次达到上限（{tool_call_rounds}/{MAX_TOOL_CALL_ROUNDS}）"
                    )
                    break
                if retrieval_rounds >= MAX_RETRIEVAL_ROUNDS and has_search_step:
                    control_stop_reason = (
                        f"检索轮次达到上限（{retrieval_rounds}/{MAX_RETRIEVAL_ROUNDS}）"
                    )
                    break

                guard_reason = check_budget_guard(budget_state)
                if guard_reason:
                    budget_guard_reason = guard_reason
                    budget_exceeded = True
                    guard_payload = build_budget_event(
                        budget_state,
                        event_type="budget_guard",
                        reason=guard_reason,
                    )
                    yield f"data: {json.dumps(guard_payload, ensure_ascii=False)}\n\n"
                    break

                iteration += 1
                accumulated_tool_call = None
                iteration_content_buffer = ""
                call_usage_total = 0
                break_by_budget_guard = False
                
                async for chunk_str in provider.chat_completion(
                    messages=conversation_messages,
                    tools=tools,
                    stream=True
                ):
                    try:
                        chunk = json.loads(chunk_str)
                        
                        if chunk["type"] == "content":
                            iteration_content_buffer += chunk.get("content", "")
                        
                        elif chunk["type"] == "usage":
                            usage_payload = chunk.get("usage", {}) or {}
                            call_usage_total = accumulate_usage_tokens(
                                budget_state,
                                usage_payload,
                                call_usage_total,
                            )
                            yield f"data: {json.dumps({'type': 'usage', 'usage': usage_payload})}\n\n"
                            budget_update_payload = build_budget_event(
                                budget_state,
                                event_type="budget_update",
                            )
                            yield f"data: {json.dumps(budget_update_payload, ensure_ascii=False)}\n\n"
                            guard_reason = check_budget_guard(budget_state)
                            if guard_reason:
                                budget_guard_reason = guard_reason
                                budget_exceeded = True
                                guard_payload = build_budget_event(
                                    budget_state,
                                    event_type="budget_guard",
                                    reason=guard_reason,
                                )
                                yield f"data: {json.dumps(guard_payload, ensure_ascii=False)}\n\n"
                                break_by_budget_guard = True
                                break
                        
                        elif chunk["type"] == "tool_calls":
                            tool_calls = chunk["tool_calls"]
                            
                            for tool_call in tool_calls:
                                if accumulated_tool_call is None:
                                    if tool_call.get("id") and tool_call.get("type"):
                                        accumulated_tool_call = {
                                            "index": tool_call.get("index", 0),
                                            "id": tool_call["id"],
                                            "type": tool_call["type"],
                                            "function": {
                                                "name": tool_call.get("function", {}).get("name", ""),
                                                "arguments": tool_call.get("function", {}).get("arguments", "")
                                            }
                                        }
                                else:
                                    if "function" in tool_call and "arguments" in tool_call["function"]:
                                        accumulated_tool_call["function"]["arguments"] += tool_call["function"]["arguments"]
                    
                    except json.JSONDecodeError:
                        continue

                if break_by_budget_guard:
                    break
                
                if accumulated_tool_call:
                    if tool_call_rounds >= MAX_TOOL_CALL_ROUNDS:
                        control_stop_reason = (
                            f"工具调用轮次达到上限（{tool_call_rounds}/{MAX_TOOL_CALL_ROUNDS}）"
                        )
                        break

                    tool_name = accumulated_tool_call.get("function", {}).get("name", "")
                    if tool_name not in allowed_tool_names:
                        accumulated_tool_call["function"]["name"] = "search_paths"
                        accumulated_tool_call["function"]["arguments"] = json.dumps(
                            {
                                "query": retrieval_query,
                                "top_k": route_plan["top_k"],
                            },
                            ensure_ascii=False,
                        )
                        tool_name = "search_paths"

                    # Force retrieval to start from search_paths so retrieve_sections
                    # always receives routed candidate paths first.
                    if tool_name == "retrieve_sections" and not has_search_step:
                        accumulated_tool_call["function"]["name"] = "search_paths"
                        accumulated_tool_call["function"]["arguments"] = json.dumps(
                            {
                                "query": retrieval_query,
                                "top_k": route_plan["top_k"],
                            },
                            ensure_ascii=False,
                        )
                        tool_name = "search_paths"

                    if tool_name == "search_paths":
                        args = _safe_parse_tool_args(accumulated_tool_call["function"].get("arguments", ""))
                        query = args.get("query") or retrieval_query
                        accumulated_tool_call["function"]["arguments"] = json.dumps(
                            {
                                "query": query,
                                "top_k": route_plan["top_k"],
                            },
                            ensure_ascii=False,
                        )

                    if tool_name == "retrieve_sections":
                        args = _safe_parse_tool_args(accumulated_tool_call["function"].get("arguments", ""))
                        file_paths = args.get("file_paths") or []
                        if not isinstance(file_paths, list):
                            file_paths = []

                        # Expand with routed candidates to improve coverage without full-file reads.
                        if len(file_paths) < route_plan["min_file_paths"] and candidate_paths:
                            existing = {str(p) for p in file_paths}
                            for p in candidate_paths:
                                if p in existing:
                                    continue
                                file_paths.append(p)
                                existing.add(p)
                                if len(file_paths) >= route_plan["min_file_paths"]:
                                    break

                        max_sections = _coerce_int(
                            args.get("max_sections_per_file"),
                            route_plan["max_sections_per_file"],
                            1,
                            4,
                        )
                        max_sections = max(max_sections, route_plan["max_sections_per_file"])
                        query = args.get("query") or retrieval_query
                        accumulated_tool_call["function"]["arguments"] = json.dumps(
                            {
                                "file_paths": file_paths,
                                "query": query or retrieval_query,
                                "max_sections_per_file": max_sections,
                            },
                            ensure_ascii=False,
                        )

                    if tool_name == "search_paths" and retrieval_rounds >= MAX_RETRIEVAL_ROUNDS:
                        control_stop_reason = (
                            f"检索轮次达到上限（{retrieval_rounds}/{MAX_RETRIEVAL_ROUNDS}）"
                        )
                        break

                    tool_label = _tool_display_name(tool_name)
                    if ENABLE_PLAN_EXECUTE_STAGES:
                        execute_running_payload = _build_reasoning_stage_payload(
                            stage_key="execute",
                            status="running",
                            title=f"{tool_label}进行中",
                            summary=f"第 {iteration} 轮执行 {tool_label}，正在补充候选路径与证据片段。",
                            mode="llm",
                            badge="RUNTIME",
                            metrics=[
                                {"label": "Iteration", "value": str(iteration)},
                                {"label": "Tool", "value": tool_label},
                                {"label": "Candidates", "value": str(len(candidate_paths))},
                                {"label": "Evidence", "value": str(len(evidence_pool))},
                            ],
                        )
                        yield f"data: {json.dumps(execute_running_payload, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [accumulated_tool_call]})}\n\n"
                    
                    tool_results = await process_tool_calls([accumulated_tool_call])
                    tool_call_rounds += 1
                    if tool_name == "search_paths":
                        retrieval_rounds += 1
                    runtime_stats = _extract_tool_runtime_stats(tool_results)
                    cache_hit_count += runtime_stats["cache_hits"]
                    retry_count_total += runtime_stats["retry_total"]
                    retry_exhausted_count += runtime_stats["retry_exhausted"]

                    if tool_name == "search_paths":
                        has_search_step = True
                        candidate_paths = _extract_candidate_paths(tool_results[0].get("content", ""))
                    elif tool_name == "retrieve_sections":
                        evidence_pool = _merge_evidence_pool(
                            evidence_pool,
                            _extract_evidence_entries(tool_results[0].get("content", "")),
                        )
                    
                    yield f"data: {json.dumps({'type': 'tool_results', 'results': tool_results})}\n\n"
                    if ENABLE_PLAN_EXECUTE_STAGES:
                        execute_done_payload = _build_reasoning_stage_payload(
                            stage_key="execute",
                            status="completed",
                            title=f"{tool_label}已完成",
                            summary="执行结果已回填到上下文，继续综合生成候选答案。",
                            mode="llm",
                            badge="UPDATED",
                            metrics=[
                                {"label": "Iteration", "value": str(iteration)},
                                {"label": "Tool", "value": tool_label},
                                {"label": "Candidates", "value": str(len(candidate_paths))},
                                {"label": "Evidence", "value": str(len(evidence_pool))},
                            ],
                        )
                        yield f"data: {json.dumps(execute_done_payload, ensure_ascii=False)}\n\n"
                    
                    conversation_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [accumulated_tool_call]
                    })
                    
                    conversation_messages.extend(tool_results)
                    continue

                candidate_answer = iteration_content_buffer.strip()
                if not candidate_answer:
                    break
                latest_candidate_answer = candidate_answer
                if ENABLE_PLAN_EXECUTE_STAGES:
                    execute_answer_payload = _build_reasoning_stage_payload(
                        stage_key="execute",
                        status="completed",
                        title="候选答案已生成",
                        summary="执行阶段已完成检索与证据汇总，进入 Critic 校验。",
                        mode="llm",
                        badge="READY",
                        metrics=[
                            {"label": "Iteration", "value": str(iteration)},
                            {"label": "Candidates", "value": str(len(candidate_paths))},
                            {"label": "Evidence", "value": str(len(evidence_pool))},
                        ],
                    )
                    yield f"data: {json.dumps(execute_answer_payload, ensure_ascii=False)}\n\n"

                # Evidence hard constraint:
                # if no evidence has been retrieved yet, force one more retrieval round.
                if ENFORCE_CRITIC_RETRIEVAL and not evidence_pool and not forced_retrieval_attempted:
                    forced_retrieval_attempted = True
                    critic_payload = await _llm_run_retrieval_critic(
                        provider,
                        user_query=latest_user_query,
                        candidate_answer=candidate_answer,
                        evidence_pool=evidence_pool,
                    )
                    critic_payload = _append_runtime_metrics(
                        critic_payload,
                        retrieval_rounds=retrieval_rounds,
                        tool_call_rounds=tool_call_rounds,
                        auto_retrieval_rounds=0,
                        retry_total=retry_count_total,
                        retry_exhausted=retry_exhausted_count,
                        cache_hits=cache_hit_count,
                        stop_reason=control_stop_reason,
                    )
                    yield (
                        "data: "
                        + json.dumps(critic_payload, ensure_ascii=False)
                        + "\n\n"
                    )
                    yield (
                        "data: "
                        + json.dumps(_build_critic_stage_payload(critic_payload), ensure_ascii=False)
                        + "\n\n"
                    )

                    if auto_retrieval_rounds >= MAX_AUTO_RETRIEVAL_ROUNDS:
                        control_stop_reason = (
                            f"自动补检索达到上限（{auto_retrieval_rounds}/{MAX_AUTO_RETRIEVAL_ROUNDS}）"
                        )
                        break
                    if tool_call_rounds >= MAX_TOOL_CALL_ROUNDS:
                        control_stop_reason = (
                            f"工具调用轮次达到上限（{tool_call_rounds}/{MAX_TOOL_CALL_ROUNDS}）"
                        )
                        break
                    if retrieval_rounds >= MAX_RETRIEVAL_ROUNDS:
                        control_stop_reason = (
                            f"检索轮次达到上限（{retrieval_rounds}/{MAX_RETRIEVAL_ROUNDS}）"
                        )
                        break

                    auto_retrieval_rounds += 1
                    auto_search_call = _build_tool_call(
                        call_id=f"auto_search_{iteration}",
                        name="search_paths",
                        arguments={
                            "query": retrieval_query,
                            "top_k": min(route_plan["top_k"] + AUTO_RETRIEVAL_TOP_K_BOOST, 10),
                        },
                    )
                    if ENABLE_PLAN_EXECUTE_STAGES:
                        auto_search_stage_payload = _build_reasoning_stage_payload(
                            stage_key="execute",
                            status="running",
                            title="自动补检索：路径检索",
                            summary="Critic 判定证据不足，系统触发自动扩搜。",
                            mode="llm",
                            badge="AUTO",
                            metrics=[
                                {"label": "Iteration", "value": str(iteration)},
                                {"label": "TopK", "value": str(min(route_plan["top_k"] + AUTO_RETRIEVAL_TOP_K_BOOST, 10))},
                            ],
                        )
                        yield f"data: {json.dumps(auto_search_stage_payload, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [auto_search_call]})}\n\n"
                    auto_search_results = await process_tool_calls([auto_search_call])
                    tool_call_rounds += 1
                    retrieval_rounds += 1
                    auto_search_stats = _extract_tool_runtime_stats(auto_search_results)
                    cache_hit_count += auto_search_stats["cache_hits"]
                    retry_count_total += auto_search_stats["retry_total"]
                    retry_exhausted_count += auto_search_stats["retry_exhausted"]
                    yield f"data: {json.dumps({'type': 'tool_results', 'results': auto_search_results})}\n\n"
                    conversation_messages.append(
                        {"role": "assistant", "content": None, "tool_calls": [auto_search_call]}
                    )
                    conversation_messages.extend(auto_search_results)

                    auto_paths = _extract_candidate_paths(auto_search_results[0].get("content", ""))
                    if auto_paths:
                        candidate_paths = auto_paths
                    has_search_step = True

                    retrieve_targets = candidate_paths[: max(route_plan["min_file_paths"], 2)]
                    if retrieve_targets:
                        if tool_call_rounds >= MAX_TOOL_CALL_ROUNDS:
                            control_stop_reason = (
                                f"工具调用轮次达到上限（{tool_call_rounds}/{MAX_TOOL_CALL_ROUNDS}）"
                            )
                            break
                        auto_retrieve_call = _build_tool_call(
                            call_id=f"auto_retrieve_{iteration}",
                            name="retrieve_sections",
                            arguments={
                                "file_paths": retrieve_targets,
                                "query": retrieval_query,
                                "max_sections_per_file": min(
                                    route_plan["max_sections_per_file"] + 1,
                                    4,
                                ),
                            },
                        )
                        if ENABLE_PLAN_EXECUTE_STAGES:
                            auto_retrieve_stage_payload = _build_reasoning_stage_payload(
                                stage_key="execute",
                                status="running",
                                title="自动补检索：片段检索",
                                summary="自动扩搜完成，正在回收更多证据片段。",
                                mode="llm",
                                badge="AUTO",
                                metrics=[
                                    {"label": "Iteration", "value": str(iteration)},
                                    {"label": "Targets", "value": str(len(retrieve_targets))},
                                ],
                            )
                            yield f"data: {json.dumps(auto_retrieve_stage_payload, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [auto_retrieve_call]})}\n\n"
                        auto_retrieve_results = await process_tool_calls([auto_retrieve_call])
                        tool_call_rounds += 1
                        auto_retrieve_stats = _extract_tool_runtime_stats(auto_retrieve_results)
                        cache_hit_count += auto_retrieve_stats["cache_hits"]
                        retry_count_total += auto_retrieve_stats["retry_total"]
                        retry_exhausted_count += auto_retrieve_stats["retry_exhausted"]
                        yield f"data: {json.dumps({'type': 'tool_results', 'results': auto_retrieve_results})}\n\n"
                        conversation_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [auto_retrieve_call],
                            }
                        )
                        conversation_messages.extend(auto_retrieve_results)
                        evidence_pool = _merge_evidence_pool(
                            evidence_pool,
                            _extract_evidence_entries(auto_retrieve_results[0].get("content", "")),
                        )

                    conversation_messages.append({"role": "assistant", "content": candidate_answer})
                    conversation_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous answer did not include evidence citations. "
                                "Re-answer strictly with a final section titled '### 证据' "
                                "and include source file path plus evidence snippet."
                            ),
                        }
                    )
                    continue

                critic_payload = await _llm_run_retrieval_critic(
                    provider,
                    user_query=latest_user_query,
                    candidate_answer=candidate_answer,
                    evidence_pool=evidence_pool,
                )
                critic_payload = _append_runtime_metrics(
                    critic_payload,
                    retrieval_rounds=retrieval_rounds,
                    tool_call_rounds=tool_call_rounds,
                    auto_retrieval_rounds=auto_retrieval_rounds,
                    retry_total=retry_count_total,
                    retry_exhausted=retry_exhausted_count,
                    cache_hits=cache_hit_count,
                    stop_reason=control_stop_reason,
                )
                yield (
                    "data: "
                    + json.dumps(critic_payload, ensure_ascii=False)
                    + "\n\n"
                )
                yield (
                    "data: "
                    + json.dumps(_build_critic_stage_payload(critic_payload), ensure_ascii=False)
                    + "\n\n"
                )
                if critic_payload.get("stop", False) and evidence_pool:
                    final_answer = _format_answer_with_evidence(candidate_answer, evidence_pool)
                    yield f"data: {json.dumps({'type': 'content', 'content': final_answer})}\n\n"
                    final_answer_sent = True
                    break

                if not ENFORCE_CRITIC_RETRIEVAL:
                    final_answer = _compose_answer_for_delivery(candidate_answer, evidence_pool)
                    yield f"data: {json.dumps({'type': 'content', 'content': final_answer})}\n\n"
                    final_answer_sent = True
                    break

                conversation_messages.append({"role": "assistant", "content": candidate_answer})
                conversation_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Critic requires additional retrieval. "
                            f"Reason: {critic_payload.get('reason', '')}. "
                            "Continue with search_paths / retrieve_sections and provide a stronger answer "
                            "with a final section titled '### 证据'."
                        ),
                    }
                )
                continue
            
            if not final_answer_sent:
                if ENFORCE_CRITIC_RETRIEVAL and not evidence_pool and not budget_exceeded:
                    if ENABLE_PLAN_EXECUTE_STAGES:
                        emergency_stage_payload = _build_reasoning_stage_payload(
                            stage_key="execute",
                            status="running",
                            title="结束前兜底检索",
                            summary="主循环结束但证据为空，触发确定性补检索。",
                            mode="router",
                            badge="GUARDRAIL",
                            metrics=[
                                {"label": "Candidates", "value": str(len(candidate_paths))},
                                {"label": "ToolRounds", "value": f"{tool_call_rounds}/{MAX_TOOL_CALL_ROUNDS}"},
                                {"label": "SearchRounds", "value": f"{retrieval_rounds}/{MAX_RETRIEVAL_ROUNDS}"},
                            ],
                        )
                        yield f"data: {json.dumps(emergency_stage_payload, ensure_ascii=False)}\n\n"

                    emergency_paths = list(candidate_paths)
                    if (not emergency_paths) and tool_call_rounds < MAX_TOOL_CALL_ROUNDS and retrieval_rounds < MAX_RETRIEVAL_ROUNDS:
                        emergency_search_call = _build_tool_call(
                            call_id="emergency_search_final",
                            name="search_paths",
                            arguments={
                                "query": retrieval_query or latest_user_query,
                                "top_k": route_plan["top_k"],
                            },
                        )
                        yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [emergency_search_call]})}\n\n"
                        emergency_search_results = await process_tool_calls([emergency_search_call])
                        tool_call_rounds += 1
                        retrieval_rounds += 1
                        emergency_search_stats = _extract_tool_runtime_stats(emergency_search_results)
                        cache_hit_count += emergency_search_stats["cache_hits"]
                        retry_count_total += emergency_search_stats["retry_total"]
                        retry_exhausted_count += emergency_search_stats["retry_exhausted"]
                        yield f"data: {json.dumps({'type': 'tool_results', 'results': emergency_search_results})}\n\n"
                        emergency_paths = _extract_candidate_paths(
                            emergency_search_results[0].get("content", "")
                        )
                        if emergency_paths:
                            candidate_paths = emergency_paths
                            has_search_step = True

                    if emergency_paths and tool_call_rounds < MAX_TOOL_CALL_ROUNDS:
                        emergency_targets = emergency_paths[: max(route_plan["min_file_paths"], 2)]
                        emergency_retrieve_call = _build_tool_call(
                            call_id="emergency_retrieve_final",
                            name="retrieve_sections",
                            arguments={
                                "file_paths": emergency_targets,
                                "query": retrieval_query or latest_user_query,
                                "max_sections_per_file": route_plan["max_sections_per_file"],
                            },
                        )
                        yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [emergency_retrieve_call]})}\n\n"
                        emergency_retrieve_results = await process_tool_calls([emergency_retrieve_call])
                        tool_call_rounds += 1
                        emergency_retrieve_stats = _extract_tool_runtime_stats(emergency_retrieve_results)
                        cache_hit_count += emergency_retrieve_stats["cache_hits"]
                        retry_count_total += emergency_retrieve_stats["retry_total"]
                        retry_exhausted_count += emergency_retrieve_stats["retry_exhausted"]
                        yield f"data: {json.dumps({'type': 'tool_results', 'results': emergency_retrieve_results})}\n\n"
                        evidence_pool = _merge_evidence_pool(
                            evidence_pool,
                            _extract_evidence_entries(emergency_retrieve_results[0].get("content", "")),
                        )

                critic_payload = await _llm_run_retrieval_critic(
                    provider,
                    user_query=latest_user_query,
                    candidate_answer=latest_candidate_answer,
                    evidence_pool=evidence_pool,
                )
                critic_payload = _append_runtime_metrics(
                    critic_payload,
                    retrieval_rounds=retrieval_rounds,
                    tool_call_rounds=tool_call_rounds,
                    auto_retrieval_rounds=auto_retrieval_rounds,
                    retry_total=retry_count_total,
                    retry_exhausted=retry_exhausted_count,
                    cache_hits=cache_hit_count,
                    stop_reason=control_stop_reason,
                )
                yield ("data: " + json.dumps(critic_payload, ensure_ascii=False) + "\n\n")
                yield (
                    "data: "
                    + json.dumps(_build_critic_stage_payload(critic_payload), ensure_ascii=False)
                    + "\n\n"
                )
                if budget_exceeded:
                    fallback_answer = _format_budget_guard_answer(
                        budget_guard_reason,
                        evidence_pool,
                    )
                elif bool(critic_payload.get("stop", False)) and bool(evidence_pool):
                    fallback_answer = _format_answer_with_evidence(
                        latest_candidate_answer,
                        evidence_pool,
                    )
                else:
                    fallback_answer = _format_insufficient_evidence_answer(
                        control_stop_reason or str(critic_payload.get("reason", "")),
                    )
                yield f"data: {json.dumps({'type': 'content', 'content': fallback_answer})}\n\n"
            budget_summary_payload = build_budget_event(
                budget_state,
                event_type="budget_summary",
            )
            yield f"data: {json.dumps(budget_summary_payload, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/providers")
async def list_providers():
    """List all configured providers - use dynamic scanning"""
    provider_ids = settings.list_available_providers()
    
    providers = []
    for provider_id in provider_ids:
        config = settings.get_provider_config(provider_id)
        if config.get("model"):
            providers.append({
                "id": provider_id,
                "name": provider_id.replace('_', ' ').title(),
                "models": [config["model"]]
            })
    
    return {"providers": providers}


@app.get("/evaluation/datasets")
async def list_evaluation_datasets():
    """List available evaluation datasets."""
    try:
        return {"datasets": evaluation_manager.list_datasets()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation/summaries")
async def list_evaluation_summaries():
    """List historical evaluation summaries."""
    try:
        return {"summaries": evaluation_manager.list_summaries()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation/jobs")
async def list_evaluation_jobs():
    """List in-memory evaluation jobs for this backend process."""
    try:
        return {"jobs": evaluation_manager.list_jobs()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation/jobs/{job_id}")
async def get_evaluation_job(job_id: str):
    """Get an evaluation job status."""
    try:
        job = evaluation_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Evaluation job not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluation/jobs/start")
async def start_evaluation_job(request: EvalStartRequest):
    """Start an asynchronous evaluation job."""
    try:
        return evaluation_manager.start_job(request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

