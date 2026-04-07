from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List
import json
import os
import re

from backend.budget_control import (
    accumulate_usage_tokens,
    build_budget_event,
    check_budget_guard,
)
from backend.prompts import process_react_response, process_tool_calls

DEFAULT_ROUTE_PLAN = {
    "top_k": 6,
    "max_sections_per_file": 2,
    "min_file_paths": 4,
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
MAX_RETRIEVAL_ROUNDS = max(1, min(int(os.getenv("MAX_RETRIEVAL_ROUNDS", "4")), 20))
MAX_TOOL_CALL_ROUNDS = max(1, min(int(os.getenv("MAX_TOOL_CALL_ROUNDS", "10")), 40))
MAX_AUTO_RETRIEVAL_ROUNDS = max(0, min(int(os.getenv("MAX_AUTO_RETRIEVAL_ROUNDS", "3")), 10))
ENABLE_PLAN_EXECUTE_STAGES = os.getenv("ENABLE_PLAN_EXECUTE_STAGES", "false").strip().lower() in {"1", "true", "yes", "on"}
ENFORCE_CRITIC_RETRIEVAL = os.getenv("ENFORCE_CRITIC_RETRIEVAL", "true").strip().lower() in {"1", "true", "yes", "on"}
MAX_FINAL_EVIDENCE_ITEMS = 3
MAX_EVIDENCE_SNIPPET_LENGTH = 280
FALLBACK_NO_EVIDENCE_ANSWER = (
    "我无法给出有证据支撑的回答。已自动再检索多轮，但仍未找到可引用片段。"
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
FINAL_ANSWER_PATTERN = re.compile(r"<\|Final Answer\|>\s*([\s\S]*)", re.IGNORECASE)
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


def _evaluate_evidence_strength(
    user_query: str,
    candidate_answer: str,
    evidence_pool: List[Dict[str, str]],
) -> Dict[str, Any]:
    evidence_items = [
        item
        for item in evidence_pool
        if (item.get("file_path", "").strip() and item.get("snippet", "").strip())
    ]
    distinct_files = {item["file_path"].strip() for item in evidence_items}
    total_snippet_chars = sum(len(item["snippet"].strip()) for item in evidence_items)

    query_signals = _extract_query_signals(user_query)
    evidence_text = " ".join(item["snippet"] for item in evidence_items).lower()
    query_term_hits = sum(
        1 for variants in query_signals if any(term in evidence_text for term in variants)
    )
    matched_evidence_items = (
        sum(
            1
            for item in evidence_items
            if any(
                any(term in item.get("snippet", "").lower() for term in variants)
                for variants in query_signals
            )
        )
        if query_signals
        else len(evidence_items)
    )
    uncertain_answer = _is_uncertain_answer(candidate_answer)
    return {
        "evidence_items": len(evidence_items),
        "distinct_files": len(distinct_files),
        "total_snippet_chars": total_snippet_chars,
        "query_term_hits": query_term_hits,
        "query_terms": len(query_signals),
        "matched_evidence_items": matched_evidence_items,
        "uncertain_answer": uncertain_answer,
    }


def _estimate_critic_confidence(strength: Dict[str, Any]) -> float:
    evidence_items = float(strength.get("evidence_items", 0) or 0)
    distinct_files = float(strength.get("distinct_files", 0) or 0)
    matched_evidence_items = float(strength.get("matched_evidence_items", 0) or 0)
    query_term_hits = float(strength.get("query_term_hits", 0) or 0)
    query_terms = float(strength.get("query_terms", 0) or 0)

    hit_ratio = 1.0 if query_terms <= 0 else query_term_hits / max(query_terms, 1.0)
    snippet_match_ratio = (
        matched_evidence_items / max(evidence_items, 1.0)
        if evidence_items > 0
        else 0.0
    )
    file_diversity_ratio = (
        min(distinct_files / (1.0 if evidence_items <= 2 else 2.0), 1.0)
        if evidence_items > 0
        else 0.0
    )

    score = hit_ratio * 0.50 + snippet_match_ratio * 0.30 + file_diversity_ratio * 0.20
    if bool(strength.get("uncertain_answer", False)):
        score *= 0.85
    return round(max(0.0, min(score, 1.0)), 2)


def _build_rule_based_critic_decision(strength: Dict[str, Any]) -> Dict[str, Any]:
    evidence_items = int(strength.get("evidence_items", 0) or 0)
    distinct_files = int(strength.get("distinct_files", 0) or 0)
    query_terms = int(strength.get("query_terms", 0) or 0)
    query_term_hits = int(strength.get("query_term_hits", 0) or 0)
    confidence = _estimate_critic_confidence(strength)
    uncertain_answer = bool(strength.get("uncertain_answer", False))
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


def _build_critic_decision(
    *,
    stop: bool,
    decision: str,
    reason: str,
    mode: str,
    strength: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_decision = str(decision or "").strip().lower()
    if normalized_decision not in {"accept", "revise", "refuse"}:
        normalized_decision = "accept" if stop else "revise"
    final_stop = normalized_decision == "accept"
    return {
        "stop": final_stop,
        "decision": normalized_decision,
        "mode": mode,
        "reason": reason,
        "evidence_items": int(strength.get("evidence_items", 0) or 0),
        "distinct_files": int(strength.get("distinct_files", 0) or 0),
        "matched_evidence_items": int(strength.get("matched_evidence_items", 0) or 0),
        "total_snippet_chars": int(strength.get("total_snippet_chars", 0) or 0),
        "uncertain_answer": bool(strength.get("uncertain_answer", False)),
        "query_term_hits": int(strength.get("query_term_hits", 0) or 0),
        "query_terms": int(strength.get("query_terms", 0) or 0),
        "confidence": _estimate_critic_confidence(strength),
    }


async def _llm_run_retrieval_critic(
    provider,
    user_query: str,
    candidate_answer: str,
    evidence_pool: List[Dict[str, str]],
) -> Dict[str, Any]:
    evidence_preview = []
    for item in evidence_pool[:4]:
        evidence_preview.append(
            {
                "file_path": item.get("file_path", ""),
                "snippet": item.get("snippet", "")[:180],
            }
        )

    critic_messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval critic for RAG. "
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
                raw_text += chunk.get("content", "")
    except Exception as exc:
        strength = _evaluate_evidence_strength(
            user_query=user_query,
            candidate_answer=candidate_answer,
            evidence_pool=evidence_pool,
        )
        return _apply_critic_rule_guardrail(
            _build_critic_decision(
                stop=False,
                decision="revise",
                reason=f"critic llm unavailable: {_shorten_text(str(exc), 90)}",
                mode="llm_error",
                strength=strength,
            )
        )

    parsed = _extract_json_object(raw_text)
    strength = _evaluate_evidence_strength(
        user_query=user_query,
        candidate_answer=candidate_answer,
        evidence_pool=evidence_pool,
    )

    raw_decision = _normalize_whitespace(str(parsed.get("decision", ""))).lower()
    if raw_decision not in {"accept", "revise", "refuse"}:
        raw_decision = ""
    raw_stop = parsed.get("stop")
    if not isinstance(raw_stop, bool):
        raw_stop = None

    parse_ok = bool(raw_decision) or isinstance(raw_stop, bool)
    reason = _normalize_whitespace(str(parsed.get("reason", "")))
    if not reason:
        reason = _shorten_text(raw_text, 120) or "critic output invalid"

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
        _build_critic_decision(
            stop=stop,
            decision=decision,
            reason=reason,
            mode=mode,
            strength=strength,
        )
    )


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _shorten_text(text: str, max_len: int = 72) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3].rstrip() + "..."


def _normalize_stage_metrics(metrics: List[Dict[str, Any]] | None) -> List[Dict[str, str]]:
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
    metrics: List[Dict[str, Any]] | None = None,
    order: int | None = None,
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
        "title": _shorten_text(title, 100) or f"{stage_label} update",
        "summary": _shorten_text(summary, 128),
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
            title = "证据存在争议，建议人工复核"
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

    summary = str(critic_payload.get("reason", ""))
    summary_prefix = "注：证据分用于衡量检索充分性，不等于答案正确率。"
    summary = f"{summary_prefix} {summary}".strip()
    if decision == "refuse" and has_some_evidence:
        summary = (
            "该结论属于保守判定（证据匹配度偏弱，不代表答案必错）。"
            f" {summary}".strip()
        )

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


def _build_critic_stream_payload(critic_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "retrieval_critic",
        "decision": str(critic_result.get("decision", "revise")),
        "stop": bool(critic_result.get("stop", False)),
        "mode": str(critic_result.get("mode", "rule")),
        "reason": _shorten_text(str(critic_result.get("reason", ""))),
        "confidence": float(critic_result.get("confidence", 0.0) or 0.0),
        "evidence_items": int(critic_result.get("evidence_items", 0) or 0),
        "distinct_files": int(critic_result.get("distinct_files", 0) or 0),
        "matched_evidence_items": int(critic_result.get("matched_evidence_items", 0) or 0),
        "total_snippet_chars": int(critic_result.get("total_snippet_chars", 0) or 0),
        "uncertain_answer": bool(critic_result.get("uncertain_answer", False)),
        "query_term_hits": int(critic_result.get("query_term_hits", 0) or 0),
        "query_terms": int(critic_result.get("query_terms", 0) or 0),
        "evidence_sufficient": bool(critic_result.get("evidence_sufficient", False)),
        "rule_reason": _shorten_text(str(critic_result.get("rule_reason", "")), 120),
        "retrieval_rounds": int(critic_result.get("retrieval_rounds", 0) or 0),
        "max_retrieval_rounds": int(critic_result.get("max_retrieval_rounds", MAX_RETRIEVAL_ROUNDS) or MAX_RETRIEVAL_ROUNDS),
        "tool_call_rounds": int(critic_result.get("tool_call_rounds", 0) or 0),
        "max_tool_call_rounds": int(critic_result.get("max_tool_call_rounds", MAX_TOOL_CALL_ROUNDS) or MAX_TOOL_CALL_ROUNDS),
        "retry_count_total": int(critic_result.get("retry_count_total", 0) or 0),
        "retry_exhausted_count": int(critic_result.get("retry_exhausted_count", 0) or 0),
        "cache_hit_count": int(critic_result.get("cache_hit_count", 0) or 0),
        "auto_retrieval_rounds": int(critic_result.get("auto_retrieval_rounds", 0) or 0),
        "max_auto_retrieval_rounds": int(critic_result.get("max_auto_retrieval_rounds", MAX_AUTO_RETRIEVAL_ROUNDS) or MAX_AUTO_RETRIEVAL_ROUNDS),
    }


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
        entries.append({"file_path": file_path, "snippet": snippet})
    return entries


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


def _extract_final_answer_text(accumulated_response: str) -> str:
    if not accumulated_response:
        return ""

    match = FINAL_ANSWER_PATTERN.search(accumulated_response)
    if match:
        return match.group(1).strip()

    return accumulated_response.strip()


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
        lines.append(f"{idx}. 来源文件: `{file_path}`")
        lines.append(f"   证据片段: \"{snippet}\"")

    return "\n".join(lines).strip()


def _compose_answer_for_delivery(candidate_answer: str, evidence_pool: List[Dict[str, str]]) -> str:
    if evidence_pool:
        return _format_answer_with_evidence(candidate_answer, evidence_pool)
    clean_answer = _strip_existing_evidence_section(candidate_answer or "").strip()
    if clean_answer:
        return clean_answer
    return FALLBACK_NO_EVIDENCE_ANSWER


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


async def handle_react_mode(
    provider,
    messages,
    user_query: str = "",
    route_plan: Dict[str, int] | None = None,
    budget_state: Dict[str, Any] | None = None,
    max_iterations: int = 10,
) -> AsyncIterator[str]:
    """Handle ReAct-style tool calling with hard evidence validation and auto-retrieval."""

    conversation_messages = messages.copy()
    iteration = 0

    safe_plan = dict(DEFAULT_ROUTE_PLAN)
    if route_plan:
        safe_plan.update(route_plan)

    evidence_pool: List[Dict[str, str]] = []
    candidate_paths: List[str] = []
    auto_retrieval_rounds = 0
    retrieval_rounds = 0
    tool_call_rounds = 0
    retry_count_total = 0
    retry_exhausted_count = 0
    cache_hit_count = 0
    final_answer_sent = False
    budget_guard_reason = ""
    latest_candidate_answer = ""
    control_stop_reason = ""

    while iteration < max_iterations:
        if tool_call_rounds >= MAX_TOOL_CALL_ROUNDS:
            control_stop_reason = f"工具调用轮次达到上限（{tool_call_rounds}/{MAX_TOOL_CALL_ROUNDS}）"
            break
        if retrieval_rounds >= MAX_RETRIEVAL_ROUNDS and candidate_paths:
            control_stop_reason = f"检索轮次达到上限（{retrieval_rounds}/{MAX_RETRIEVAL_ROUNDS}）"
            break

        if budget_state:
            guard_reason = check_budget_guard(budget_state)
            if guard_reason:
                budget_guard_reason = guard_reason
                guard_payload = build_budget_event(
                    budget_state,
                    event_type="budget_guard",
                    reason=guard_reason,
                )
                yield f"data: {json.dumps(guard_payload, ensure_ascii=False)}\\n\\n"
                break

        iteration += 1
        accumulated_response = ""
        action_detected = False
        action_result = None
        has_action = False
        call_usage_total = 0
        break_by_budget_guard = False

        async for chunk_str in provider.chat_completion(
            messages=conversation_messages,
            tools=None,
            stream=True,
        ):
            try:
                chunk = json.loads(chunk_str)

                if chunk["type"] == "content":
                    content = chunk.get("content") or ""
                    accumulated_response += content

                    if "<|Action Input|>" in accumulated_response and "{" in accumulated_response:
                        parsed_result, parsed_has_action = await process_react_response(accumulated_response)
                        if parsed_has_action:
                            action_detected = True
                            action_result = parsed_result
                            has_action = True
                            break

                elif chunk["type"] == "usage":
                    usage_payload = chunk.get("usage", {}) or {}
                    call_usage_total = accumulate_usage_tokens(
                        budget_state,
                        usage_payload,
                        call_usage_total,
                    )
                    yield f"data: {json.dumps({'type': 'usage', 'usage': usage_payload})}\\n\\n"
                    if budget_state:
                        budget_update_payload = build_budget_event(
                            budget_state,
                            event_type="budget_update",
                        )
                        yield f"data: {json.dumps(budget_update_payload, ensure_ascii=False)}\\n\\n"
                        guard_reason = check_budget_guard(budget_state)
                        if guard_reason:
                            budget_guard_reason = guard_reason
                            guard_payload = build_budget_event(
                                budget_state,
                                event_type="budget_guard",
                                reason=guard_reason,
                            )
                            yield f"data: {json.dumps(guard_payload, ensure_ascii=False)}\\n\\n"
                            break_by_budget_guard = True
                            break

            except json.JSONDecodeError:
                continue

        if break_by_budget_guard:
            break

        if not action_detected:
            action_result, has_action = await process_react_response(accumulated_response)

        if has_action and action_result:
            action_name = str(action_result.get("action", "")).strip() or "tool"
            if tool_call_rounds >= MAX_TOOL_CALL_ROUNDS:
                control_stop_reason = f"工具调用轮次达到上限（{tool_call_rounds}/{MAX_TOOL_CALL_ROUNDS}）"
                break
            if action_name == "search_paths" and retrieval_rounds >= MAX_RETRIEVAL_ROUNDS:
                control_stop_reason = f"检索轮次达到上限（{retrieval_rounds}/{MAX_RETRIEVAL_ROUNDS}）"
                break

            action_label = _tool_display_name(action_name)
            if ENABLE_PLAN_EXECUTE_STAGES:
                execute_running_payload = _build_reasoning_stage_payload(
                    stage_key="execute",
                    status="running",
                    title=f"{action_label}进行中",
                    summary=f"第 {iteration} 轮执行 {action_label}，正在补充候选路径与证据片段。",
                    mode="llm",
                    badge="RUNTIME",
                    metrics=[
                        {"label": "Iteration", "value": str(iteration)},
                        {"label": "Tool", "value": action_label},
                        {"label": "Candidates", "value": str(len(candidate_paths))},
                        {"label": "Evidence", "value": str(len(evidence_pool))},
                    ],
                )
                yield f"data: {json.dumps(execute_running_payload, ensure_ascii=False)}\\n\\n"
            tool_call_info = {
                "id": f"call_{iteration}",
                "type": "function",
                "function": {
                    "name": action_result["action"],
                    "arguments": json.dumps(action_result["input"], ensure_ascii=False),
                },
            }
            yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [tool_call_info]})}\\n\\n"

            observation = action_result.get("observation", "")
            content_length = len(observation)
            if content_length > 50000:
                frontend_content = (
                    f"[Retrieved {content_length:,} characters from knowledge base. "
                    "Content sent to LLM for analysis.]"
                )
            else:
                frontend_content = observation

            tool_result_info = {
                "role": "tool",
                "tool_call_id": f"call_{iteration}",
                "content": frontend_content,
                "meta": action_result.get("meta", {}),
            }
            yield f"data: {json.dumps({'type': 'tool_results', 'results': [tool_result_info]})}\\n\\n"

            tool_call_rounds += 1
            if action_result["action"] == "search_paths":
                retrieval_rounds += 1
            action_meta = action_result.get("meta", {})
            if isinstance(action_meta, dict):
                retry_count_total += int(action_meta.get("retry_count", 0) or 0)
                retry_exhausted_count += 1 if bool(action_meta.get("retry_exhausted", False)) else 0
                cache_hit_count += 1 if bool(action_meta.get("cache_hit", False)) else 0

            if action_result["action"] == "search_paths":
                candidate_paths = _extract_candidate_paths(observation)
            elif action_result["action"] == "retrieve_sections":
                evidence_pool = _merge_evidence_pool(
                    evidence_pool,
                    _extract_evidence_entries(observation),
                )

            if ENABLE_PLAN_EXECUTE_STAGES:
                execute_done_payload = _build_reasoning_stage_payload(
                    stage_key="execute",
                    status="completed",
                    title=f"{action_label}已完成",
                    summary="执行结果已回填到上下文，继续综合生成候选答案。",
                    mode="llm",
                    badge="UPDATED",
                    metrics=[
                        {"label": "Iteration", "value": str(iteration)},
                        {"label": "Tool", "value": action_label},
                        {"label": "Candidates", "value": str(len(candidate_paths))},
                        {"label": "Evidence", "value": str(len(evidence_pool))},
                    ],
                )
                yield f"data: {json.dumps(execute_done_payload, ensure_ascii=False)}\\n\\n"

            conversation_messages.append(
                {
                    "role": "assistant",
                    "content": accumulated_response,
                }
            )
            conversation_messages.append(
                {
                    "role": "user",
                    "content": f"<|Observation|> {observation}\\n\\nContinue with your reasoning.",
                }
            )
            continue

        candidate_answer = _extract_final_answer_text(accumulated_response)
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
            yield f"data: {json.dumps(execute_answer_payload, ensure_ascii=False)}\\n\\n"
        critic_result = await _llm_run_retrieval_critic(
            provider=provider,
            user_query=user_query,
            candidate_answer=candidate_answer,
            evidence_pool=evidence_pool,
        )
        critic_result = _append_runtime_metrics(
            critic_result,
            retrieval_rounds=retrieval_rounds,
            tool_call_rounds=tool_call_rounds,
            auto_retrieval_rounds=auto_retrieval_rounds,
            retry_total=retry_count_total,
            retry_exhausted=retry_exhausted_count,
            cache_hits=cache_hit_count,
            stop_reason=control_stop_reason,
        )
        critic_payload = _build_critic_stream_payload(critic_result)
        yield f"data: {json.dumps(critic_payload, ensure_ascii=False)}\\n\\n"
        yield f"data: {json.dumps(_build_critic_stage_payload(critic_payload), ensure_ascii=False)}\\n\\n"
        if critic_result.get("stop", False) and evidence_pool:
            final_answer = _format_answer_with_evidence(candidate_answer, evidence_pool)
            yield f"data: {json.dumps({'type': 'content', 'content': final_answer})}\\n\\n"
            final_answer_sent = True
            break

        if not ENFORCE_CRITIC_RETRIEVAL:
            final_answer = _compose_answer_for_delivery(candidate_answer, evidence_pool)
            yield f"data: {json.dumps({'type': 'content', 'content': final_answer})}\\n\\n"
            final_answer_sent = True
            break

        if auto_retrieval_rounds >= MAX_AUTO_RETRIEVAL_ROUNDS:
            control_stop_reason = (
                f"自动补检索达到上限（{auto_retrieval_rounds}/{MAX_AUTO_RETRIEVAL_ROUNDS}）"
            )
            critic_result = await _llm_run_retrieval_critic(
                provider=provider,
                user_query=user_query,
                candidate_answer=candidate_answer,
                evidence_pool=evidence_pool,
            )
            critic_result = _append_runtime_metrics(
                critic_result,
                retrieval_rounds=retrieval_rounds,
                tool_call_rounds=tool_call_rounds,
                auto_retrieval_rounds=auto_retrieval_rounds,
                retry_total=retry_count_total,
                retry_exhausted=retry_exhausted_count,
                cache_hits=cache_hit_count,
                stop_reason=control_stop_reason,
            )
            critic_payload = _build_critic_stream_payload(critic_result)
            yield f"data: {json.dumps(critic_payload, ensure_ascii=False)}\\n\\n"
            yield f"data: {json.dumps(_build_critic_stage_payload(critic_payload), ensure_ascii=False)}\\n\\n"
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "content",
                        "content": _format_insufficient_evidence_answer(control_stop_reason),
                    },
                    ensure_ascii=False,
                )
                + "\\n\\n"
            )
            final_answer_sent = True
            break

        if tool_call_rounds >= MAX_TOOL_CALL_ROUNDS:
            control_stop_reason = f"工具调用轮次达到上限（{tool_call_rounds}/{MAX_TOOL_CALL_ROUNDS}）"
            break
        if retrieval_rounds >= MAX_RETRIEVAL_ROUNDS:
            control_stop_reason = f"检索轮次达到上限（{retrieval_rounds}/{MAX_RETRIEVAL_ROUNDS}）"
            break

        auto_retrieval_rounds += 1
        retrieval_query = user_query or candidate_answer or "knowledge base question"

        auto_search_call = _build_tool_call(
            call_id=f"auto_search_{iteration}_{auto_retrieval_rounds}",
            name="search_paths",
            arguments={
                "query": retrieval_query,
                "top_k": min(safe_plan["top_k"] + AUTO_RETRIEVAL_TOP_K_BOOST, 10),
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
                    {"label": "Round", "value": str(auto_retrieval_rounds)},
                    {"label": "TopK", "value": str(min(safe_plan["top_k"] + AUTO_RETRIEVAL_TOP_K_BOOST, 10))},
                ],
            )
            yield f"data: {json.dumps(auto_search_stage_payload, ensure_ascii=False)}\\n\\n"
        yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [auto_search_call]})}\\n\\n"
        auto_search_results = await process_tool_calls([auto_search_call])
        tool_call_rounds += 1
        retrieval_rounds += 1
        auto_search_stats = _extract_tool_runtime_stats(auto_search_results)
        retry_count_total += auto_search_stats["retry_total"]
        retry_exhausted_count += auto_search_stats["retry_exhausted"]
        cache_hit_count += auto_search_stats["cache_hits"]
        yield f"data: {json.dumps({'type': 'tool_results', 'results': auto_search_results})}\\n\\n"
        auto_search_output = auto_search_results[0].get("content", "")
        auto_paths = _extract_candidate_paths(auto_search_output)
        if auto_paths:
            candidate_paths = auto_paths

        retrieve_targets = candidate_paths[: max(int(safe_plan.get("min_file_paths", 4)), 2)]
        auto_retrieve_output = ""
        if retrieve_targets:
            if tool_call_rounds >= MAX_TOOL_CALL_ROUNDS:
                control_stop_reason = f"工具调用轮次达到上限（{tool_call_rounds}/{MAX_TOOL_CALL_ROUNDS}）"
                break
            auto_retrieve_call = _build_tool_call(
                call_id=f"auto_retrieve_{iteration}_{auto_retrieval_rounds}",
                name="retrieve_sections",
                arguments={
                    "file_paths": retrieve_targets,
                    "query": retrieval_query,
                    "max_sections_per_file": min(
                        int(safe_plan.get("max_sections_per_file", 2)) + 1,
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
                        {"label": "Round", "value": str(auto_retrieval_rounds)},
                        {"label": "Targets", "value": str(len(retrieve_targets))},
                    ],
                )
                yield f"data: {json.dumps(auto_retrieve_stage_payload, ensure_ascii=False)}\\n\\n"
            yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [auto_retrieve_call]})}\\n\\n"
            auto_retrieve_results = await process_tool_calls([auto_retrieve_call])
            tool_call_rounds += 1
            auto_retrieve_stats = _extract_tool_runtime_stats(auto_retrieve_results)
            retry_count_total += auto_retrieve_stats["retry_total"]
            retry_exhausted_count += auto_retrieve_stats["retry_exhausted"]
            cache_hit_count += auto_retrieve_stats["cache_hits"]
            yield f"data: {json.dumps({'type': 'tool_results', 'results': auto_retrieve_results})}\\n\\n"
            auto_retrieve_output = auto_retrieve_results[0].get("content", "")
            evidence_pool = _merge_evidence_pool(
                evidence_pool,
                _extract_evidence_entries(auto_retrieve_output),
            )

        conversation_messages.append({"role": "assistant", "content": accumulated_response})

        observation_parts = [
            f"<|Observation|> Auto search_paths result:\n{auto_search_output}",
        ]
        if auto_retrieve_output:
            observation_parts.append(
                f"<|Observation|> Auto retrieve_sections result:\n{auto_retrieve_output}"
            )

        if evidence_pool:
            guard_prompt = (
                "Evidence snippets are now available. Decide whether they are sufficient. "
                "If sufficient, provide final answer with `### 证据`; "
                "if not, continue retrieval."
            )
            if isinstance(critic_result, dict):
                critic_reason = str(critic_result.get("reason", "")).strip()
                if critic_reason:
                    guard_prompt += f" Stop critic reason: {critic_reason}"
        else:
            guard_prompt = (
                "No evidence snippet found yet. Please continue retrieval with broader/narrower query. "
                "If still insufficient after reasonable attempts, answer I don't know."
            )

        conversation_messages.append(
            {
                "role": "user",
                "content": "\\n\\n".join(observation_parts + [guard_prompt]),
            }
        )

    if not final_answer_sent:
        if ENFORCE_CRITIC_RETRIEVAL and not evidence_pool:
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
                yield f"data: {json.dumps(emergency_stage_payload, ensure_ascii=False)}\\n\\n"

            emergency_paths = list(candidate_paths)
            if (not emergency_paths) and tool_call_rounds < MAX_TOOL_CALL_ROUNDS and retrieval_rounds < MAX_RETRIEVAL_ROUNDS:
                emergency_search_call = _build_tool_call(
                    call_id="emergency_search_final",
                    name="search_paths",
                    arguments={
                        "query": user_query or latest_candidate_answer or "knowledge base question",
                        "top_k": int(safe_plan.get("top_k", 6)),
                    },
                )
                yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [emergency_search_call]})}\\n\\n"
                emergency_search_results = await process_tool_calls([emergency_search_call])
                tool_call_rounds += 1
                retrieval_rounds += 1
                emergency_search_stats = _extract_tool_runtime_stats(emergency_search_results)
                retry_count_total += emergency_search_stats["retry_total"]
                retry_exhausted_count += emergency_search_stats["retry_exhausted"]
                cache_hit_count += emergency_search_stats["cache_hits"]
                yield f"data: {json.dumps({'type': 'tool_results', 'results': emergency_search_results})}\\n\\n"
                emergency_paths = _extract_candidate_paths(emergency_search_results[0].get("content", ""))
                if emergency_paths:
                    candidate_paths = emergency_paths

            if emergency_paths and tool_call_rounds < MAX_TOOL_CALL_ROUNDS:
                emergency_targets = emergency_paths[: max(int(safe_plan.get("min_file_paths", 4)), 2)]
                emergency_retrieve_call = _build_tool_call(
                    call_id="emergency_retrieve_final",
                    name="retrieve_sections",
                    arguments={
                        "file_paths": emergency_targets,
                        "query": user_query or latest_candidate_answer or "knowledge base question",
                        "max_sections_per_file": int(safe_plan.get("max_sections_per_file", 2)),
                    },
                )
                yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [emergency_retrieve_call]})}\\n\\n"
                emergency_retrieve_results = await process_tool_calls([emergency_retrieve_call])
                tool_call_rounds += 1
                emergency_retrieve_stats = _extract_tool_runtime_stats(emergency_retrieve_results)
                retry_count_total += emergency_retrieve_stats["retry_total"]
                retry_exhausted_count += emergency_retrieve_stats["retry_exhausted"]
                cache_hit_count += emergency_retrieve_stats["cache_hits"]
                yield f"data: {json.dumps({'type': 'tool_results', 'results': emergency_retrieve_results})}\\n\\n"
                evidence_pool = _merge_evidence_pool(
                    evidence_pool,
                    _extract_evidence_entries(emergency_retrieve_results[0].get("content", "")),
                )

        critic_result = await _llm_run_retrieval_critic(
            provider=provider,
            user_query=user_query,
            candidate_answer=latest_candidate_answer,
            evidence_pool=evidence_pool,
        )
        critic_result = _append_runtime_metrics(
            critic_result,
            retrieval_rounds=retrieval_rounds,
            tool_call_rounds=tool_call_rounds,
            auto_retrieval_rounds=auto_retrieval_rounds,
            retry_total=retry_count_total,
            retry_exhausted=retry_exhausted_count,
            cache_hits=cache_hit_count,
            stop_reason=control_stop_reason,
        )
        critic_payload = _build_critic_stream_payload(critic_result)
        yield f"data: {json.dumps(critic_payload, ensure_ascii=False)}\\n\\n"
        yield f"data: {json.dumps(_build_critic_stage_payload(critic_payload), ensure_ascii=False)}\\n\\n"

        if budget_state and budget_state.get("guard_triggered"):
            reason = str(budget_state.get("guard_reason") or budget_guard_reason or "").strip()
            budget_guard_answer = _format_budget_guard_answer(reason, evidence_pool)
            yield (
                f"data: {json.dumps({'type': 'content', 'content': budget_guard_answer}, ensure_ascii=False)}\\n\\n"
            )
        elif bool(critic_result.get("stop", False)) and bool(evidence_pool):
            final_answer = _format_answer_with_evidence(
                latest_candidate_answer,
                evidence_pool,
            )
            yield f"data: {json.dumps({'type': 'content', 'content': final_answer})}\\n\\n"
        else:
            fallback = _format_insufficient_evidence_answer(
                control_stop_reason or str(critic_result.get("reason", "")),
            )
            yield f"data: {json.dumps({'type': 'content', 'content': fallback})}\\n\\n"


