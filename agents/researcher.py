from loguru import logger
from agents.state import AgentState
from tools.enricher import enrich_prospect
from tools.signal_extractor import get_ranked_signals
from database import mark_contacted, get_signal_weights, save_review


def researcher_agent(state: AgentState) -> AgentState:
    """
    ResearcherAgent node.
    Enriches the prospect via Clay+Vibe (or Claude MCP), ranks signals,
    and marks the prospect as contacted immediately (crash-safe).
    """
    prospect = state["prospect"]
    run_id = state["run_id"]
    errors = list(state.get("errors") or [])

    logger.info(f"[ResearcherAgent] {prospect.name} @ {prospect.company}")

    try:
        enriched = enrich_prospect(prospect)
    except Exception as exc:
        logger.error(f"[ResearcherAgent] Enrichment crash for {prospect.name}: {exc}")
        errors.append(f"enrichment_crash: {exc}")
        return {**state, "enriched": None, "signals": [], "errors": errors, "final_status": "skipped"}

    if not enriched.email:
        logger.info(f"[ResearcherAgent] No email for {prospect.name} — will need manual review")

    history_weights = get_signal_weights()
    signals = get_ranked_signals(enriched, history_weights)

    if not signals:
        logger.info(f"[ResearcherAgent] No signals for {prospect.name} — sending to review queue")
        mark_contacted(prospect.apollo_id, prospect.name, prospect.company, run_id)
        email_note = f" | email: {enriched.email}" if enriched.email else " | email: not found"
        save_review({
            "apollo_id": prospect.apollo_id,
            "name": prospect.name,
            "company": prospect.company,
            "signal_type": "no_signal",
            "message": f"[Needs manual outreach] {prospect.title} at {prospect.company}{email_note}",
            "total_score": 0.0,
            "feedback": "No signals found — enrich manually or skip.",
            "run_id": run_id,
        })
        return {**state, "enriched": enriched, "signals": [], "errors": errors, "final_status": "review_needed"}

    mark_contacted(prospect.apollo_id, prospect.name, prospect.company, run_id)

    return {
        **state,
        "enriched": enriched,
        "signals": signals,
        "current_signal_index": 0,
        "current_signal": signals[0],
        "attempt_number": 1,
        "previous_feedback": None,
        "errors": errors,
    }


def route_after_research(state: AgentState) -> str:
    if state.get("final_status") == "skipped" or not state.get("signals"):
        return "skipped"
    return "has_signals"
