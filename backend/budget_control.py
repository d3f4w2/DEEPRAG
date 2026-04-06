from __future__ import annotations

from time import perf_counter
from typing import Any, Dict


def _to_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(low, min(parsed, high))


def _to_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if parsed != parsed:  # NaN
        parsed = default
    return max(low, min(parsed, high))


def normalize_budget_config(raw_budget: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = raw_budget or {}
    return {
        "max_total_tokens": _to_int(payload.get("max_total_tokens"), 0, 0, 20_000_000),
        "max_latency_ms": _to_int(payload.get("max_latency_ms"), 0, 0, 3_600_000),
        "price_per_1m_tokens": _to_float(payload.get("price_per_1m_tokens"), 0.0, 0.0, 1_000_000.0),
        "cost_multiplier": _to_float(payload.get("cost_multiplier"), 1.0, 0.0, 100.0),
    }


def create_budget_state(config: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = normalize_budget_config(config)
    return {
        **cfg,
        "started_at": perf_counter(),
        "elapsed_ms": 0,
        "total_tokens": 0,
        "guard_triggered": False,
        "guard_reason": "",
    }


def extract_total_tokens(usage: Dict[str, Any] | None) -> int:
    payload = usage or {}
    total = payload.get("total_tokens")
    if isinstance(total, (int, float)) and total >= 0:
        return int(total)

    prompt = payload.get("prompt_tokens")
    completion = payload.get("completion_tokens")
    prompt_value = int(prompt) if isinstance(prompt, (int, float)) and prompt >= 0 else 0
    completion_value = int(completion) if isinstance(completion, (int, float)) and completion >= 0 else 0
    fallback_total = prompt_value + completion_value
    return fallback_total if fallback_total > 0 else 0


def accumulate_usage_tokens(
    budget_state: Dict[str, Any] | None,
    usage: Dict[str, Any] | None,
    call_usage_total: int,
) -> int:
    if not budget_state:
        return call_usage_total
    current_total = extract_total_tokens(usage)
    delta = max(0, current_total - max(0, int(call_usage_total or 0)))
    if delta > 0:
        budget_state["total_tokens"] = int(budget_state.get("total_tokens", 0)) + delta
    return max(current_total, call_usage_total)


def update_elapsed_ms(budget_state: Dict[str, Any] | None) -> int:
    if not budget_state:
        return 0
    started_at = budget_state.get("started_at")
    if not isinstance(started_at, (int, float)):
        started_at = perf_counter()
        budget_state["started_at"] = started_at
    elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
    budget_state["elapsed_ms"] = elapsed_ms
    return elapsed_ms


def estimate_cost_usd(budget_state: Dict[str, Any] | None) -> float:
    if not budget_state:
        return 0.0
    total_tokens = max(0, int(budget_state.get("total_tokens", 0)))
    price_per_1m = float(budget_state.get("price_per_1m_tokens", 0.0) or 0.0)
    multiplier = float(budget_state.get("cost_multiplier", 1.0) or 1.0)
    return (total_tokens / 1_000_000.0) * price_per_1m * multiplier


def check_budget_guard(budget_state: Dict[str, Any] | None) -> str:
    if not budget_state:
        return ""

    if budget_state.get("guard_triggered"):
        return str(budget_state.get("guard_reason") or "budget guard triggered")

    elapsed_ms = update_elapsed_ms(budget_state)
    total_tokens = max(0, int(budget_state.get("total_tokens", 0)))
    max_total_tokens = max(0, int(budget_state.get("max_total_tokens", 0)))
    max_latency_ms = max(0, int(budget_state.get("max_latency_ms", 0)))

    reason = ""
    if max_total_tokens > 0 and total_tokens >= max_total_tokens:
        reason = f"token budget reached ({total_tokens}/{max_total_tokens})"
    elif max_latency_ms > 0 and elapsed_ms >= max_latency_ms:
        reason = f"latency budget reached ({elapsed_ms}ms/{max_latency_ms}ms)"

    if reason:
        budget_state["guard_triggered"] = True
        budget_state["guard_reason"] = reason
    return reason


def build_budget_event(
    budget_state: Dict[str, Any] | None,
    event_type: str,
    reason: str | None = None,
) -> Dict[str, Any]:
    if not budget_state:
        return {
            "type": event_type,
            "triggered": False,
            "reason": reason or "",
            "usage": {"total_tokens": 0, "elapsed_ms": 0},
            "pricing": {"price_per_1m_tokens": 0.0, "cost_multiplier": 1.0},
            "limits": {"max_total_tokens": 0, "max_latency_ms": 0},
            "cost_estimate_usd": 0.0,
        }

    elapsed_ms = update_elapsed_ms(budget_state)
    total_tokens = max(0, int(budget_state.get("total_tokens", 0)))
    default_reason = str(budget_state.get("guard_reason") or "")
    return {
        "type": event_type,
        "triggered": bool(budget_state.get("guard_triggered", False)),
        "reason": (reason if reason is not None else default_reason),
        "usage": {
            "total_tokens": total_tokens,
            "elapsed_ms": elapsed_ms,
        },
        "pricing": {
            "price_per_1m_tokens": float(budget_state.get("price_per_1m_tokens", 0.0) or 0.0),
            "cost_multiplier": float(budget_state.get("cost_multiplier", 1.0) or 1.0),
        },
        "limits": {
            "max_total_tokens": max(0, int(budget_state.get("max_total_tokens", 0))),
            "max_latency_ms": max(0, int(budget_state.get("max_latency_ms", 0))),
        },
        "cost_estimate_usd": round(estimate_cost_usd(budget_state), 10),
    }
