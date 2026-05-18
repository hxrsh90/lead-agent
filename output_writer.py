from loguru import logger
from agents.state import AgentState
from database import save_approved, save_review, update_signal_history


def save_approved_node(state: AgentState) -> AgentState:
    """LangGraph node: persist approved message to SQLite and update signal history."""
    enriched = state["enriched"]
    signal = state["current_signal"]
    message = state["current_message"]
    score = state["score_result"]
    run_id = state["run_id"]

    record = {
        "apollo_id": enriched.apollo_id,
        "name": enriched.name,
        "company": enriched.company,
        "email": enriched.email,
        "signal_type": signal.type,
        "signal_content": signal.content,
        "message": message,
        "specificity": score.specificity,
        "relevance": score.relevance,
        "naturalness": score.naturalness,
        "total_score": score.total,
        "run_id": run_id,
    }
    try:
        save_approved(record)
        update_signal_history(signal.type, approved=True)
        logger.info(
            f"[OutputWriter] ✓ Approved — {enriched.name} "
            f"(score={score.total:.2f}, signal={signal.type})"
        )
    except Exception as exc:
        logger.error(f"[OutputWriter] Failed to save approved message for {enriched.name}: {exc}")

    return {**state, "final_status": "approved"}


def save_review_node(state: AgentState) -> AgentState:
    """LangGraph node: persist review-needed message to SQLite and update signal history."""
    enriched = state["enriched"]
    signal = state.get("current_signal")
    message = state.get("current_message") or ""
    score = state.get("score_result")
    run_id = state["run_id"]

    record = {
        "apollo_id": enriched.apollo_id,
        "name": enriched.name,
        "company": enriched.company,
        "signal_type": signal.type if signal else "unknown",
        "message": message,
        "total_score": score.total if score else 0.0,
        "feedback": score.feedback if score else "All signals and retries exhausted.",
        "run_id": run_id,
    }
    try:
        save_review(record)
        if signal:
            update_signal_history(signal.type, approved=False)
        logger.info(
            f"[OutputWriter] ⚠ Review needed — {enriched.name} "
            f"(score={record['total_score']:.2f})"
        )
    except Exception as exc:
        logger.error(f"[OutputWriter] Failed to save review record for {enriched.name}: {exc}")

    return {**state, "final_status": "review_needed"}
