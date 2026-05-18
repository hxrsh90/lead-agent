from loguru import logger
from agents.state import AgentState
from tools.enricher import enrich_prospect
from tools.signal_extractor import get_ranked_signals
from database import mark_contacted, get_signal_weights


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
        logger.info(f"[ResearcherAgent] No email found for {prospect.name} — skipping")
        mark_contacted(prospect.apollo_id, prospect.name, prospect.company, run_id)
        return {**state, "enriched": enriched, "signals": [], "errors": errors, "final_status": "skipped"}

    history_weights = get_signal_weights()
    signals = get_ranked_signals(enriched, history_weights)

    if not signals:
        logger.info(f"[ResearcherAgent] No signals for {prospect.name} — skipping")
        mark_contacted(prospect.apollo_id, prospect.name, prospect.company, run_id)
        return {**state, "enriched": enriched, "signals": [], "errors": errors, "final_status": "skipped"}

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
