from typing import List, Optional, Dict
from datetime import date, datetime
from loguru import logger
from agents.state import EnrichedProspectModel, SignalModel
from config import icp

BASE_PRIORITIES: Dict[str, int] = {
    "open_rcm_jobs": 1,
    "prospect_changed_company": 2,
    "prospect_changed_role": 3,
    "linkedin_pain_post": 4,
    "company_news": 5,
    "thought_leadership": 6,
    "prospect_job_start_anniversary": 7,
    "work_history": 8,
}

LOOKBACK_DAYS: Dict[str, int] = {
    "open_rcm_jobs": 7,
    "prospect_changed_company": icp.job_change_days,
    "prospect_changed_role": icp.job_change_days,
    "linkedin_pain_post": icp.linkedin_post_days,
    "company_news": icp.company_news_days,
    "thought_leadership": 180,
    "prospect_job_start_anniversary": 30,
    "work_history": 3650,
}


def _date_freshness(date_str: Optional[str], max_days: int) -> float:
    """1.0 within window, decays to 0.1 beyond 2× window, 0.5 if date unknown."""
    if not date_str:
        return 0.5
    try:
        signal_date = datetime.fromisoformat(date_str).date()
        age_days = (date.today() - signal_date).days
        if age_days <= max_days:
            return 1.0
        elif age_days <= max_days * 2:
            return 0.6
        return 0.1
    except (ValueError, TypeError):
        return 0.5


def _get_all_signals(enriched: EnrichedProspectModel) -> List[SignalModel]:
    signals: List[SignalModel] = []

    if enriched.open_rcm_jobs:
        signals.append(SignalModel(
            type="open_rcm_jobs",
            content=f"hiring for {enriched.open_rcm_jobs[0]} — signals high prior auth / admin call volume right now",
            priority=BASE_PRIORITIES["open_rcm_jobs"],
        ))

    for event in enriched.events:
        if event.type == "prospect_changed_company":
            signals.append(SignalModel(
                type="prospect_changed_company",
                content=f"recently joined {event.company or enriched.company} as {event.title or enriched.title}",
                date=event.date,
                priority=BASE_PRIORITIES["prospect_changed_company"],
            ))
            break

    for event in enriched.events:
        if event.type == "prospect_changed_role":
            signals.append(SignalModel(
                type="prospect_changed_role",
                content=f"recently moved into {event.new_title or event.title or enriched.title} at {event.company or enriched.company}",
                date=event.date,
                priority=BASE_PRIORITIES["prospect_changed_role"],
            ))
            break

    if enriched.linkedin_pain_post:
        signals.append(SignalModel(
            type="linkedin_pain_post",
            content=enriched.linkedin_pain_post,
            priority=BASE_PRIORITIES["linkedin_pain_post"],
        ))

    if enriched.company_news:
        signals.append(SignalModel(
            type="company_news",
            content=enriched.company_news,
            priority=BASE_PRIORITIES["company_news"],
        ))

    if enriched.thought_leadership:
        signals.append(SignalModel(
            type="thought_leadership",
            content=enriched.thought_leadership,
            priority=BASE_PRIORITIES["thought_leadership"],
        ))

    for event in enriched.events:
        if event.type == "prospect_job_start_anniversary":
            years = event.years or 1
            signals.append(SignalModel(
                type="prospect_job_start_anniversary",
                content=f"celebrating {years} year{'s' if years != 1 else ''} at {event.company or enriched.company}",
                date=event.date,
                priority=BASE_PRIORITIES["prospect_job_start_anniversary"],
            ))
            break

    if enriched.work_history:
        signals.append(SignalModel(
            type="work_history",
            content=enriched.work_history,
            priority=BASE_PRIORITIES["work_history"],
        ))

    return signals


def _rank(signals: List[SignalModel], history_weights: Dict[str, float]) -> List[SignalModel]:
    max_p = max(BASE_PRIORITIES.values())
    for s in signals:
        history_w = history_weights.get(s.type, 0.5)
        freshness = _date_freshness(s.date, LOOKBACK_DAYS.get(s.type, 90))
        priority_norm = (max_p - s.priority + 1) / max_p
        s.score = round(priority_norm * history_w * freshness, 4)
    return sorted(signals, key=lambda x: x.score, reverse=True)


def get_ranked_signals(
    enriched: EnrichedProspectModel,
    history_weights: Dict[str, float],
) -> List[SignalModel]:
    signals = _get_all_signals(enriched)
    ranked = _rank(signals, history_weights)
    top = ranked[0].type if ranked else "none"
    logger.info(f"[signal] {enriched.name}: {len(ranked)} signals, top={top}")
    return ranked
