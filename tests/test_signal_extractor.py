import pytest
from agents.state import EnrichedProspectModel, EventModel
from tools.signal_extractor import get_ranked_signals, _date_freshness
from datetime import date, timedelta


def _make_enriched(**kwargs) -> EnrichedProspectModel:
    defaults = dict(
        apollo_id="test-123",
        name="Jane Smith",
        first_name="Jane",
        last_name="Smith",
        title="Practice Manager",
        company="Sunrise Medical Group",
        company_domain="sunrisemedical.com",
    )
    defaults.update(kwargs)
    return EnrichedProspectModel(**defaults)


def test_open_rcm_jobs_is_top_signal():
    enriched = _make_enriched(
        open_rcm_jobs=["Prior Authorization Specialist"],
        work_history="10 years in healthcare admin",
    )
    signals = get_ranked_signals(enriched, {})
    assert signals[0].type == "open_rcm_jobs"


def test_signals_are_ranked_by_score_descending():
    enriched = _make_enriched(
        open_rcm_jobs=["RCM Coordinator"],
        work_history="Background in RCM",
    )
    signals = get_ranked_signals(enriched, {})
    scores = [s.score for s in signals]
    assert scores == sorted(scores, reverse=True)


def test_no_signals_returns_empty():
    enriched = _make_enriched()
    signals = get_ranked_signals(enriched, {})
    assert signals == []


def test_date_freshness_within_window():
    recent = (date.today() - timedelta(days=5)).isoformat()
    assert _date_freshness(recent, 14) == 1.0


def test_date_freshness_beyond_window():
    old = (date.today() - timedelta(days=200)).isoformat()
    assert _date_freshness(old, 14) == 0.1


def test_date_freshness_unknown():
    assert _date_freshness(None, 30) == 0.5


def test_history_weights_boost_signal():
    enriched = _make_enriched(
        open_rcm_jobs=["Prior Auth Specialist"],
        thought_leadership="Posted about RCM automation",
    )
    weights = {"thought_leadership": 1.0, "open_rcm_jobs": 0.1}
    signals = get_ranked_signals(enriched, weights)
    types = [s.type for s in signals]
    assert types.index("thought_leadership") < types.index("open_rcm_jobs")


def test_company_change_event_captured():
    enriched = _make_enriched(events=[
        EventModel(type="prospect_changed_company", company="NewCorp", title="RCM Director")
    ])
    signals = get_ranked_signals(enriched, {})
    assert any(s.type == "prospect_changed_company" for s in signals)
