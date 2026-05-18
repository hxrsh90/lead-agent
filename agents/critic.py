import json
import re
from loguru import logger
from agents.state import AgentState, ScoreResultModel
from llm import call_llm
from config import settings

CRITIC_SYSTEM_PROMPT = """\
You are a strict quality evaluator for LinkedIn outreach messages targeting healthcare \
Revenue Cycle and practice management leaders.

Score the given message on three dimensions from 0.0 to 1.0:

SPECIFICITY (0.0–1.0): Does the message reference a real, specific, named detail from the \
provided signal? Generic references score 0.2. Naming a specific job title, company action, \
or exact post topic scores 0.9–1.0.

RELEVANCE (0.0–1.0): Does the message connect the signal to a real RCM or admin phone call \
pain point a Practice Manager or Revenue Cycle Director would recognize? \
(prior auth calls, benefits verification, claim status calls, staff burnout from insurance \
calls, denials workload). No connection = 0.0, strong and clear connection = 1.0.

NATURALNESS (0.0–1.0): Does the message sound like a real human peer sending a genuine note, \
or like a sales bot template? Generic phrases, hollow compliments, or feature-dumping score \
near 0.0. Conversational, peer-to-peer tone with a soft question scores near 1.0. \
If the message exceeds 300 characters, deduct 0.3 from naturalness.

Return ONLY a valid JSON object with these exact keys:
{
  "specificity": <float 0.0–1.0>,
  "relevance": <float 0.0–1.0>,
  "naturalness": <float 0.0–1.0>,
  "feedback": "<one concise sentence: the single most important weakness to fix>"
}

No markdown fences. No explanation outside the JSON.\
"""


def critic_agent(state: AgentState) -> AgentState:
    """
    CriticAgent node.
    Scores message on specificity, relevance, naturalness.
    Stores feedback for WriterAgent injection on retry.
    """
    enriched = state["enriched"]
    signal = state["current_signal"]
    message = state.get("current_message") or ""
    attempt = state["attempt_number"]
    errors = list(state.get("errors") or [])

    logger.info(f"[CriticAgent] {enriched.name} | attempt={attempt} | chars={len(message)}")

    if not message:
        score_result = ScoreResultModel(
            specificity=0.0, relevance=0.0, naturalness=0.0,
            total=0.0, status="review_needed",
            feedback="Writer produced an empty message.",
        )
        return {**state, "score_result": score_result,
                "previous_feedback": score_result.feedback, "errors": errors}

    user_prompt = (
        f"Score this LinkedIn message:\n\n"
        f"Message ({len(message)} characters):\n{message}\n\n"
        f"Signal used:\n{signal.content}"
    )

    try:
        raw = call_llm(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            use_mcp=False,
            max_tokens=300,
            temperature=0.1,
            step_name=f"critic_attempt_{attempt}",
        )
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean).rstrip("`").strip()
        data = json.loads(clean)

        specificity = max(0.0, min(1.0, float(data.get("specificity", 0.0))))
        relevance = max(0.0, min(1.0, float(data.get("relevance", 0.0))))
        naturalness = max(0.0, min(1.0, float(data.get("naturalness", 0.0))))
        total = round(specificity + relevance + naturalness, 4)

        score_result = ScoreResultModel(
            specificity=specificity,
            relevance=relevance,
            naturalness=naturalness,
            total=total,
            status="approved" if total >= settings.quality_threshold else "review_needed",
            feedback=data.get("feedback", "No specific feedback provided."),
        )
        logger.info(
            f"[CriticAgent] {enriched.name}: "
            f"spec={specificity:.2f} rel={relevance:.2f} nat={naturalness:.2f} "
            f"total={total:.2f} → {score_result.status}"
        )

    except Exception as exc:
        logger.error(f"[CriticAgent] Scoring failed for {enriched.name}: {exc}")
        errors.append(f"critic_error_attempt_{attempt}: {exc}")
        score_result = ScoreResultModel(
            specificity=0.0, relevance=0.0, naturalness=0.0,
            total=0.0, status="review_needed",
            feedback=f"Scoring error on attempt {attempt}: {exc}",
        )

    # ── Compute routing decision and update state transitions inside the node ──
    # LangGraph routing functions cannot persist state mutations — only node
    # return values are checkpointed. We encode the decision in route_decision.
    signals = state.get("signals") or []
    attempt = state.get("attempt_number", 1)
    signal_index = state.get("current_signal_index", 0)

    next_attempt = attempt
    next_signal_index = signal_index
    next_signal = state.get("current_signal")
    next_feedback = score_result.feedback

    if score_result.status == "approved":
        route = "approved"
    elif attempt < 3:
        route = "retry"
        next_attempt = attempt + 1
    else:
        next_index = signal_index + 1
        if next_index < len(signals):
            route = "next_signal"
            next_signal_index = next_index
            next_signal = signals[next_index]
            next_attempt = 1
            next_feedback = None
        else:
            route = "exhausted"

    return {
        **state,
        "score_result": score_result,
        "previous_feedback": next_feedback,
        "attempt_number": next_attempt,
        "current_signal_index": next_signal_index,
        "current_signal": next_signal,
        "route_decision": route,
        "errors": errors,
    }


def route_after_critique(state: AgentState) -> str:
    """
    Stateless conditional edge — reads route_decision set by critic_agent node.
    approved | retry | next_signal | exhausted
    """
    return state.get("route_decision") or "exhausted"
