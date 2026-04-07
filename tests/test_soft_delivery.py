from backend.main import (
    _format_bounded_answer_for_delivery,
    _get_soft_delivery_reason,
)


def test_soft_delivery_reason_triggers_for_bounded_answer_with_evidence():
    reason = _get_soft_delivery_reason(
        critic_payload={
            "evidence_items": 1,
            "query_term_hits": 1,
            "confidence": 0.52,
            "uncertain_answer": False,
            "bounded_answer": True,
        },
        candidate_answer=(
            "Based on the current evidence, we can confirm OLED and AMOLED, "
            "but this does not cover the full LED taxonomy."
        ),
        evidence_pool=[
            {
                "file_path": "display.md",
                "snippet": "OLED and AMOLED are listed in the display roadmap.",
            }
        ],
        retrieval_rounds=2,
        tool_call_rounds=5,
        critic_revise_followup_rounds=0,
        budget_state={"total_tokens": 12000},
    )

    assert reason


def test_soft_delivery_reason_stays_empty_without_evidence():
    reason = _get_soft_delivery_reason(
        critic_payload={
            "evidence_items": 0,
            "query_term_hits": 0,
            "confidence": 0.91,
            "uncertain_answer": False,
            "bounded_answer": True,
        },
        candidate_answer="Based on the current evidence, we can confirm several items.",
        evidence_pool=[],
        retrieval_rounds=3,
        tool_call_rounds=6,
        critic_revise_followup_rounds=1,
        budget_state={"total_tokens": 24000},
    )

    assert reason == ""


def test_format_bounded_answer_for_delivery_adds_cost_control_note():
    text = _format_bounded_answer_for_delivery(
        "OLED and AMOLED are supported by the currently retrieved evidence.",
        [{"file_path": "display.md", "snippet": "OLED and AMOLED are listed."}],
        soft_reason="检索轮次已偏高",
    )

    assert "保守版结论" in text
    assert "停止继续检索" in text
    assert "### 证据" in text
