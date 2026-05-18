import pytest
from unittest.mock import patch
from agents.state import (
    AgentState, EnrichedProspectModel, SignalModel, ScoreResultModel, ProspectModel
)


def _make_state(message: str = "", attempt: int = 1) -> AgentState:
    prospect = ProspectModel(
        apollo_id="test-001", name="Jane Smith", first_name="Jane", last_name="Smith",
        title="Practice Manager", company="Sunrise Medical", company_domain="sunrise.com",
    )
    enriched = EnrichedProspectModel(
        **prospect.model_dump(), email="jane@sunrise.com",
    )
    signal = SignalModel(type="open_rcm_jobs", content="hiring for Prior Auth Specialist", priority=1)
    return AgentState(
        prospect=prospect, run_id="test-run", enriched=enriched,
        signals=[signal], current_signal_index=0, current_signal=signal,
        current_message=message, attempt_number=attempt, previous_feedback=None,
        score_result=None, final_status=None, route_decision=None,
        step_latencies={}, errors=[],
    )


def test_empty_message_auto_fails():
    from agents.critic import critic_agent
    state = _make_state(message="")
    result = critic_agent(state)
    assert result["score_result"].status == "review_needed"
    assert result["score_result"].total == 0.0


def test_approved_when_score_meets_threshold():
    from agents.critic import critic_agent
    llm_response = '{"specificity": 0.9, "relevance": 0.9, "naturalness": 0.9, "feedback": "Excellent."}'
    with patch("agents.critic.call_llm", return_value=llm_response):
        state = _make_state(message="Hi Jane, saw you're hiring for prior auth. That's usually a sign of heavy call volume — how are you managing it now?")
        result = critic_agent(state)
    assert result["score_result"].status == "approved"
    assert result["score_result"].total == pytest.approx(2.7)


def test_review_needed_when_score_below_threshold():
    from agents.critic import critic_agent
    llm_response = '{"specificity": 0.6, "relevance": 0.7, "naturalness": 0.5, "feedback": "Too generic."}'
    with patch("agents.critic.call_llm", return_value=llm_response):
        state = _make_state(message="Hi Jane, hope you're well. Let's connect!")
        result = critic_agent(state)
    assert result["score_result"].status == "review_needed"


def test_feedback_stored_in_previous_feedback():
    from agents.critic import critic_agent
    llm_response = '{"specificity": 0.5, "relevance": 0.5, "naturalness": 0.5, "feedback": "Be more specific."}'
    with patch("agents.critic.call_llm", return_value=llm_response):
        state = _make_state(message="Short message.")
        result = critic_agent(state)
    assert result["previous_feedback"] == "Be more specific."


def test_route_approved_via_node():
    from agents.critic import critic_agent, route_after_critique
    llm_resp = '{"specificity": 0.9, "relevance": 0.9, "naturalness": 0.9, "feedback": "Great."}'
    with patch("agents.critic.call_llm", return_value=llm_resp):
        state = _make_state(message="Hi Jane, noticed you're hiring for Prior Auth — heavy call volume signal. How are you handling it currently?")
        result = critic_agent(state)
    assert result["route_decision"] == "approved"
    assert route_after_critique(result) == "approved"


def test_route_retry_on_first_failure():
    from agents.critic import critic_agent, route_after_critique
    llm_resp = '{"specificity": 0.5, "relevance": 0.5, "naturalness": 0.5, "feedback": "Too generic."}'
    with patch("agents.critic.call_llm", return_value=llm_resp):
        state = _make_state(message="Hi Jane, hope you're well. Let's connect!")
        state["attempt_number"] = 1
        result = critic_agent(state)
    assert result["route_decision"] == "retry"
    assert result["attempt_number"] == 2
    assert route_after_critique(result) == "retry"


def test_route_next_signal_after_3_attempts():
    from agents.critic import critic_agent, route_after_critique
    s1 = SignalModel(type="open_rcm_jobs", content="hiring", priority=1)
    s2 = SignalModel(type="work_history", content="background", priority=8)
    llm_resp = '{"specificity": 0.4, "relevance": 0.4, "naturalness": 0.4, "feedback": "Still generic."}'
    with patch("agents.critic.call_llm", return_value=llm_resp):
        state = _make_state(message="Generic message.")
        state["signals"] = [s1, s2]
        state["current_signal_index"] = 0
        state["attempt_number"] = 3
        result = critic_agent(state)
    assert result["route_decision"] == "next_signal"
    assert result["current_signal_index"] == 1
    assert result["attempt_number"] == 1
    assert result["previous_feedback"] is None
    assert route_after_critique(result) == "next_signal"


def test_route_exhausted_when_no_more_signals():
    from agents.critic import critic_agent, route_after_critique
    signal = SignalModel(type="work_history", content="background", priority=8)
    llm_resp = '{"specificity": 0.3, "relevance": 0.3, "naturalness": 0.3, "feedback": "Weak."}'
    with patch("agents.critic.call_llm", return_value=llm_resp):
        state = _make_state(message="Weak message.")
        state["signals"] = [signal]
        state["current_signal_index"] = 0
        state["attempt_number"] = 3
        result = critic_agent(state)
    assert result["route_decision"] == "exhausted"
    assert route_after_critique(result) == "exhausted"
