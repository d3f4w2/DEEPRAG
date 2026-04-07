from backend.main import _build_rule_based_critic_decision, _evaluate_retrieval_critic_metrics


def test_rule_based_critic_accepts_bounded_answer_with_single_strong_evidence():
    metrics = _evaluate_retrieval_critic_metrics(
        user_query="What display types can be confirmed from the current evidence?",
        candidate_answer=(
            "Based on the current evidence, we can confirm micro-LED and AMOLED as related "
            "display types, but not a full LED taxonomy."
        ),
        evidence_pool=[
            {
                "file_path": "R&D-Center-Teams/Display-Tech-Team.md",
                "snippet": "The display team works on micro-LED, AMOLED, PMOLED, OLED and flexible panels.",
            }
        ],
    )

    decision = _build_rule_based_critic_decision(metrics)

    assert metrics["bounded_answer"] is True
    assert decision["decision"] == "accept"
    assert decision["stop"] is True


def test_rule_based_critic_still_rejects_when_no_evidence_exists():
    metrics = _evaluate_retrieval_critic_metrics(
        user_query="What display types are supported?",
        candidate_answer="I think there may be several display types.",
        evidence_pool=[],
    )

    decision = _build_rule_based_critic_decision(metrics)

    assert decision["decision"] in {"revise", "refuse"}
    assert decision["stop"] is False
