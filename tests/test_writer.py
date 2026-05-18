import pytest
from unittest.mock import patch
from agents.state import (
    AgentState, EnrichedProspectModel, SignalModel, ProspectModel
)


def _make_state(attempt: int = 1, previous_feedback: str = None) -> AgentState:
    prospect = ProspectModel(
        apollo_id="test-001", name="Jane Smith", first_name="Jane", last_name="Smith",
        title="Practice Manager", company="Sunrise Medical", company_domain="sunrise.com",
    )
    enriched = EnrichedProspectModel(**prospect.model_dump(), email="jane@sunrise.com")
    signal = SignalModel(
        type="open_rcm_jobs",
        content="hiring for Prior Authorization Specialist",
        priority=1,
    )
    return AgentState(
        prospect=prospect, run_id="test-run", enriched=enriched,
        signals=[signal], current_signal_index=0, current_signal=signal,
        current_message=None, attempt_number=attempt, previous_feedback=previous_feedback,
        score_result=None, final_status=None, route_decision=None,
        step_latencies={}, errors=[],
    )


def test_message_written_to_state():
    from agents.writer import writer_agent
    mock_msg = "Jane, noticed you're hiring for Prior Auth — that's usually a major call volume signal. How are you handling it?"
    with patch("agents.writer.call_llm", return_value=mock_msg):
        result = writer_agent(_make_state())
    assert result["current_message"] == mock_msg


def test_feedback_injected_on_retry():
    from agents.writer import writer_agent
    captured_prompts = {}

    def capture_llm(system_prompt, user_prompt, **kwargs):
        captured_prompts["user"] = user_prompt
        return "Retry message text."

    state = _make_state(attempt=2, previous_feedback="Be more specific about the job title.")
    with patch("agents.writer.call_llm", side_effect=capture_llm):
        writer_agent(state)

    assert "Be more specific" in captured_prompts["user"]
    assert "REJECTED" in captured_prompts["user"]


def test_llm_error_produces_empty_message():
    from agents.writer import writer_agent
    with patch("agents.writer.call_llm", side_effect=Exception("API timeout")):
        result = writer_agent(_make_state())
    assert result["current_message"] == ""
    assert any("writer_error" in e for e in result["errors"])


def test_signal_content_in_prompt():
    from agents.writer import writer_agent
    captured = {}

    def capture(system_prompt, user_prompt, **kwargs):
        captured["user"] = user_prompt
        return "Message"

    with patch("agents.writer.call_llm", side_effect=capture):
        writer_agent(_make_state())

    assert "Prior Authorization Specialist" in captured["user"]
