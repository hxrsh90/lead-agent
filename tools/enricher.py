import asyncio
import json
import re
import time
from typing import Optional, Dict, Any, List
import requests
from loguru import logger
from config import settings, icp
from database import get_live_setting
from llm import call_llm
from agents.state import ProspectModel, EnrichedProspectModel, EventModel

CLAY_BASE_URL = "https://api.clay.com/v3"
APOLLO_BASE_URL = "https://api.apollo.io/api/v1"


# ── Auth headers ───────────────────────────────────────────────────────────────

def _clay_headers() -> Dict[str, str]:
    key = get_live_setting("CLAY_API_KEY", settings.clay_api_key)
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _apollo_headers() -> Dict[str, str]:
    return {"x-api-key": get_live_setting("APOLLO_API_KEY", settings.apollo_api_key),
            "Content-Type": "application/json", "Cache-Control": "no-cache"}


# ── Clay retry wrapper (shared with prospect_finder) ──────────────────────────

def _clay_request(method: str, url: str, **kwargs) -> requests.Response:
    """429 → wait 5s, retry ×3. 5xx → wait 2s, retry ×2."""
    for attempt in range(3):
        try:
            resp = requests.request(method, url, headers=_clay_headers(), **kwargs)
            if resp.status_code == 429:
                logger.warning(f"[Clay] 429 rate limit {url} — waiting 5s (attempt {attempt + 1})")
                time.sleep(5)
                continue
            if resp.status_code >= 500:
                logger.warning(f"[Clay] {resp.status_code} on {url} — waiting 2s (attempt {attempt + 1})")
                if attempt < 2:
                    time.sleep(2)
                    continue
            if 400 <= resp.status_code < 500:
                logger.warning(f"[Clay] {resp.status_code} on {url}: {resp.text[:200]}")
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except requests.HTTPError:
            raise
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            logger.warning(f"[Clay] Request error {url}: {exc}, retrying")
    raise RuntimeError(f"[Clay] All retries exhausted for {url}")


def _parse_events(events_raw: Any) -> List[EventModel]:
    if not events_raw:
        return []
    if not isinstance(events_raw, list):
        events_raw = [events_raw]
    result = []
    for e in events_raw:
        if isinstance(e, dict) and e.get("type"):
            result.append(EventModel(
                type=e.get("type", ""),
                company=e.get("company"),
                title=e.get("title"),
                new_title=e.get("new_title"),
                years=e.get("years"),
                date=e.get("date"),
            ))
    return result


# ── Claude mode ────────────────────────────────────────────────────────────────

def _enrich_claude(prospect: ProspectModel) -> EnrichedProspectModel:
    """Single LLM call with MCP tools; Claude picks which tools to call."""
    system = (
        "You are a sales intelligence assistant. Use the Clay and Vibe Prospecting MCP tools "
        "to find enrichment data about the given healthcare contact. "
        "Return ONLY a valid JSON object — no markdown, no explanation."
    )
    user = (
        f"Enrich this healthcare prospect:\n"
        f"Name: {prospect.name}\n"
        f"Title: {prospect.title}\n"
        f"Company: {prospect.company}\n"
        f"Domain: {prospect.company_domain}\n\n"
        f"Use Vibe Prospecting to:\n"
        f"1. match-prospects (full_name, company_name)\n"
        f"2. enrich-prospects for email\n"
        f"3. fetch-prospects-events for: prospect_changed_company, prospect_changed_role, "
        f"prospect_job_start_anniversary\n\n"
        f"Use Clay to:\n"
        f"1. find-and-enrich-contacts-at-company for Email, Summarize Work History, "
        f"Find Thought Leadership\n"
        f"2. Find active job openings matching: {', '.join(icp.rcm_job_signals)}\n"
        f"3. Get company tech stack (EHR/PMS software)\n"
        f"4. Claygent: did {prospect.name} post on LinkedIn in the last 14 days about prior "
        f"auth, admin burden, or RCM challenges? Return post summary or null.\n"
        f"5. Claygent: any company news for {prospect.company} in the last 30 days "
        f"(new locations, acquisitions, service lines)? Return headline or null.\n\n"
        f"Return JSON with keys: email, work_history, thought_leadership, "
        f"tech_stack (list of strings), open_rcm_jobs (list of job title strings), "
        f"linkedin_pain_post (string or null), company_news (string or null), "
        f"events (list of {{type, company, title, new_title, years, date}})"
    )

    raw = call_llm(system, user, use_mcp=True, max_tokens=1500, step_name="enricher_claude")

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean).rstrip("`").strip()
        data = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"[enricher_claude] Non-JSON response for {prospect.name}: {raw[:200]}")
        data = {}

    return EnrichedProspectModel(
        **prospect.model_dump(),
        email=data.get("email"),
        work_history=data.get("work_history"),
        thought_leadership=data.get("thought_leadership"),
        tech_stack=data.get("tech_stack") or [],
        open_rcm_jobs=data.get("open_rcm_jobs") or [],
        linkedin_pain_post=data.get("linkedin_pain_post"),
        company_news=data.get("company_news"),
        events=_parse_events(data.get("events")),
        enrichment_source="claude",
    )


# ── Custom mode — Clay taskId-based enrich + poll ─────────────────────────────

def _clay_poll_task(task_id: str) -> Dict[str, Any]:
    """
    GET /tasks/{taskId} every 3 seconds, up to 30 seconds.
    Returns completed task data dict, or {} on timeout/failure.
    """
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            resp = _clay_request("GET", f"{CLAY_BASE_URL}/tasks/{task_id}", timeout=10)
            data = resp.json()
            status = data.get("status", "")
            logger.info(f"[Clay] task {task_id} status={status}")
            if status == "completed":
                return data
            if status == "failed":
                logger.error(f"[Clay] task {task_id} failed: {data.get('error')}")
                return {}
        except Exception as exc:
            logger.warning(f"[Clay] poll error for task {task_id}: {exc}")
        time.sleep(3)
    logger.warning(f"[Clay] task {task_id} timed out after 30s")
    return {}


def _clay_enrich_and_poll(task_id: str) -> Dict[str, Any]:
    """
    Trigger Clay enrichment for all datapoints in one call, then poll until done.
    Returns the completed task response dict.
    """
    try:
        resp = _clay_request(
            "POST",
            f"{CLAY_BASE_URL}/sources/person/enrich",
            json={
                "taskId": task_id,
                "dataPoints": [
                    {"type": "Email"},
                    {"type": "Find Thought Leadership"},
                    {"type": "Summarize Work History"},
                ],
            },
            timeout=30,
        )
        data = resp.json()
        enrich_task_id = data.get("taskId") or data.get("task_id") or task_id
        return _clay_poll_task(enrich_task_id)
    except Exception as exc:
        logger.warning(f"[Clay] enrich+poll failed for taskId={task_id}: {exc}")
        return {}


def _extract_contact_data(task_data: Dict[str, Any], prospect_name: str) -> Dict[str, Any]:
    """
    Find the specific prospect's enriched data inside a completed Clay task response.
    Falls back to top-level fields if contacts list isn't present.
    """
    if not task_data:
        return {"email": None, "work_history": None, "thought_leadership": None}

    contacts = task_data.get("contacts") or task_data.get("results") or task_data.get("data") or []

    # Try to match by name
    match: Dict[str, Any] = {}
    if contacts:
        name_lower = prospect_name.lower()
        for c in contacts:
            c_name = (c.get("name") or c.get("full_name") or "").lower()
            if c_name and c_name in name_lower or name_lower in c_name:
                match = c
                break
        if not match:
            match = contacts[0]  # fall back to first contact in the task
    else:
        match = task_data  # flat response

    def _first(*keys):
        for k in keys:
            v = match.get(k)
            if v and str(v).lower() not in ("null", "none", ""):
                return v
        return None

    return {
        "email": _first("email", "work_email", "personal_email"),
        "work_history": _first("summarize_work_history", "work_history", "experience_summary"),
        "thought_leadership": _first("find_thought_leadership", "thought_leadership", "recent_content"),
    }


def _pick_events_from_work_history(work_history: Optional[str], name: str, company: str) -> List[EventModel]:
    """
    Derive EventModel signals from work history text using keyword heuristics.
    Priority: recent job change > role change > general background.
    """
    if not work_history:
        return []
    text = work_history.lower()
    events: List[EventModel] = []

    recent_keywords = ["started", "joined", "promoted", "recently", "new role", "just became"]
    if any(kw in text for kw in recent_keywords):
        events.append(EventModel(
            type="job_change",
            company=company,
            title=name,
            raw={"source": "clay_work_history", "summary": work_history[:300]},
        ))
    elif "promoted" in text or "new role" in text:
        events.append(EventModel(
            type="role_change",
            company=company,
            title=name,
            raw={"source": "clay_work_history", "summary": work_history[:300]},
        ))
    else:
        events.append(EventModel(
            type="work_background",
            company=company,
            title=name,
            raw={"source": "clay_work_history", "summary": work_history[:300]},
        ))
    return events


def _apollo_enrich(prospect: ProspectModel) -> Dict[str, Any]:
    """
    Apollo /people/match — get full contact record by name + company.
    Returns email + employment_history for signal extraction.
    """
    if not settings.apollo_api_key:
        return {}
    try:
        resp = requests.post(
            f"{APOLLO_BASE_URL}/people/match",
            json={
                "name": prospect.name,
                "organization_name": prospect.company,
                "domain": prospect.company_domain or None,
                "reveal_personal_emails": False,
            },
            headers=_apollo_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        person = resp.json().get("person") or {}
        logger.info(f"[Apollo] Enriched {prospect.name}: email={'✓' if person.get('email') else '✗'}")
        return person
    except Exception as exc:
        logger.warning(f"[Apollo] match failed for {prospect.name}: {exc}")
        return {}


def _signals_from_apollo(person: Dict[str, Any], prospect: ProspectModel) -> List[EventModel]:
    """Extract job change / work background events from Apollo employment_history."""
    history = person.get("employment_history") or []
    if not history:
        return []
    # Most recent role is first
    current = history[0] if history else {}
    summary = (
        f"{current.get('title', '')} at {current.get('organization_name', '')} "
        f"(started {current.get('start_date', 'unknown')})"
    ).strip()

    # Detect recent start (within ~90 days) by checking start_date year/month
    start_date = current.get("start_date") or ""
    event_type = "job_change" if start_date and "2024" in start_date or "2025" in start_date else "work_background"

    return [EventModel(
        type=event_type,
        company=prospect.company,
        title=prospect.title,
        raw={"source": "apollo_employment_history", "summary": summary},
    )]


def _enrich_custom(prospect: ProspectModel) -> EnrichedProspectModel:
    """
    Custom-mode enrichment — branches on prospect source:

    Apollo-sourced (apollo_id starts with 'apollo_'):
      → POST /people/match to get email + employment_history
      → Derive events from employment history

    Clay-sourced (has clay_task_id):
      → POST /sources/person/enrich with taskId + all data points
      → Poll /tasks/{taskId} until completed
      → Extract email, thought_leadership, work_history
    """
    # ── Explorium-sourced prospect — Apollo /people/match → Hunter for email ──
    if prospect.apollo_id.startswith("vibe_"):
        email = prospect.prefetched_email  # usually None (Explorium only has hashed emails)
        history_text = None
        events = []
        person: Dict[str, Any] = {}

        if not email and settings.apollo_api_key:
            person = _apollo_enrich(prospect)
            email = person.get("email")

        history = person.get("employment_history") or []
        if history:
            history_text = "; ".join(
                f"{e.get('title','')} @ {e.get('organization_name','')} ({e.get('start_date','')}–{e.get('end_date','present')})"
                for e in history[:3]
            )
            events = _signals_from_apollo(person, prospect)

        if not events:
            events.append(EventModel(
                type="work_background",
                company=prospect.company,
                title=prospect.title,
                raw={"source": "explorium", "summary": f"{prospect.title} at {prospect.company}"},
            ))
        source = "explorium" if email else "explorium_no_email"
        logger.info(f"[enricher_custom] {prospect.name} [explorium]: email={'✓' if email else '✗'}")
        return EnrichedProspectModel(
            **prospect.model_dump(),
            email=email,
            work_history=history_text,
            thought_leadership=None,
            tech_stack=[], open_rcm_jobs=[],
            linkedin_pain_post=None, company_news=None,
            events=events,
            enrichment_source=source,
        )

    # ── Apollo-sourced prospect ───────────────────────────────────────────────
    if prospect.apollo_id.startswith("apollo_"):
        person = _apollo_enrich(prospect)
        email = person.get("email")
        history = person.get("employment_history") or []
        work_history = (
            "; ".join(
                f"{e.get('title','')} @ {e.get('organization_name','')} ({e.get('start_date','')}–{e.get('end_date','present')})"
                for e in history[:3]
            ) or None
        )
        events = _signals_from_apollo(person, prospect)
        source = "apollo" if (email or work_history) else "apollo_no_data"
        logger.info(f"[enricher_custom] {prospect.name} [apollo]: email={'✓' if email else '✗'}")
        return EnrichedProspectModel(
            **prospect.model_dump(),
            email=email,
            work_history=work_history,
            thought_leadership=None,
            tech_stack=[],
            open_rcm_jobs=[],
            linkedin_pain_post=None,
            company_news=None,
            events=events,
            enrichment_source=source,
        )

    # ── Clay-sourced prospect (has taskId) ────────────────────────────────────
    task_id = prospect.clay_task_id
    if not task_id:
        logger.warning(f"[enricher_custom] No clay_task_id for {prospect.name}")
        return EnrichedProspectModel(**prospect.model_dump(), enrichment_source="clay_failed")

    logger.info(f"[enricher_custom] {prospect.name} [clay] taskId={task_id}")
    task_data = _clay_enrich_and_poll(task_id)
    contact = _extract_contact_data(task_data, prospect.name)

    email = contact.get("email")
    work_history = contact.get("work_history")
    thought_leadership = contact.get("thought_leadership")
    events = _pick_events_from_work_history(work_history, prospect.name, prospect.company)
    source = "clay" if (email or work_history or thought_leadership) else "clay_failed"

    logger.info(
        f"[enricher_custom] {prospect.name} [clay]: email={'✓' if email else '✗'} "
        f"tl={'✓' if thought_leadership else '✗'} wh={'✓' if work_history else '✗'}"
    )
    return EnrichedProspectModel(
        **prospect.model_dump(),
        email=email,
        work_history=work_history,
        thought_leadership=thought_leadership,
        tech_stack=[],
        open_rcm_jobs=[],
        linkedin_pain_post=None,
        company_news=None,
        events=events,
        enrichment_source=source,
    )


def enrich_prospect(prospect: ProspectModel) -> EnrichedProspectModel:
    """Route enrichment based on AGENT_MODE."""
    logger.info(f"[enricher] {prospect.name} @ {prospect.company} [mode={settings.agent_mode}]")
    if settings.agent_mode == "claude":
        return _enrich_claude(prospect)
    return _enrich_custom(prospect)
