import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
import requests
from loguru import logger
from config import settings, icp
from database import is_contacted, has_searched_company, mark_company_searched, get_live_setting
from agents.state import ProspectModel

APOLLO_BASE = "https://api.apollo.io/api/v1"
EXPLORIUM_BASE = "https://api.explorium.ai/v1"
CLAY_BASE = "https://api.clay.com/v3"

_EMPLOYEE_BUCKETS = ["1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000"]


# ── Auth headers ───────────────────────────────────────────────────────────────

def _apollo_headers() -> Dict[str, str]:
    return {"x-api-key": get_live_setting("APOLLO_API_KEY", settings.apollo_api_key),
            "Content-Type": "application/json", "Cache-Control": "no-cache"}


def _explorium_headers() -> Dict[str, str]:
    return {"API_KEY": get_live_setting("VIBE_API_KEY", settings.vibe_api_key),
            "Content-Type": "application/json"}


def _clay_headers() -> Dict[str, str]:
    key = get_live_setting("CLAY_API_KEY", settings.clay_api_key)
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _stable_id(name: str, company: str) -> str:
    raw = f"{name.lower().strip()}|{company.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Clay retry wrapper ─────────────────────────────────────────────────────────

def _clay_request(method: str, url: str, **kwargs) -> requests.Response:
    """Clay API call with retry: 429 → 5s×3, 5xx → 2s×2."""
    for attempt in range(3):
        try:
            resp = requests.request(method, url, headers=_clay_headers(), **kwargs)
            if resp.status_code == 429:
                logger.warning(f"[Clay] 429 rate limit on {url}, waiting 5s (attempt {attempt + 1})")
                time.sleep(5)
                continue
            if resp.status_code >= 500:
                logger.warning(f"[Clay] {resp.status_code} server error on {url}, waiting 2s (attempt {attempt + 1})")
                if attempt < 2:
                    time.sleep(2)
                    continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            logger.warning(f"[Clay] Request error on {url}: {exc}, retrying")
    raise RuntimeError(f"[Clay] All retries exhausted for {url}")


# ── Source 1: Apollo — direct people search by ICP criteria ───────────────────

def _apollo_search(limit: int) -> List[Dict[str, Any]]:
    """
    POST /mixed_people/search
    Searches Apollo's database directly for ICP-matching contacts.
    Returns people with name, title, company, email (when available), linkedin_url.
    """
    if not get_live_setting("APOLLO_API_KEY", settings.apollo_api_key):
        logger.debug("[ProspectFinder] No APOLLO_API_KEY — skipping Apollo search")
        return []

    # Apollo uses "organization_num_employees_ranges" as "min,max" string pairs
    emp_ranges = [f"{icp.employee_min},{icp.employee_max}"]

    payload = {
        "person_titles": icp.titles,
        "person_locations": ["United States"],
        "q_keywords": "healthcare revenue cycle",
        "organization_num_employees_ranges": emp_ranges,
        "page": 1,
        "per_page": min(limit, 100),
    }
    try:
        resp = requests.post(
            f"{APOLLO_BASE}/mixed_people/search",
            json=payload,
            headers=_apollo_headers(),
            timeout=30,
        )
        if resp.status_code in (401, 403):
            logger.warning(f"[ProspectFinder] Apollo auth failed ({resp.status_code}): {resp.text[:300]}")
            return []
        resp.raise_for_status()
        people = resp.json().get("people") or []
        logger.info(f"[ProspectFinder] Apollo returned {len(people)} candidates")
        return people
    except Exception as exc:
        logger.warning(f"[ProspectFinder] Apollo search failed: {exc}")
        return []


def _parse_apollo(raw: Dict[str, Any]) -> Optional[ProspectModel]:
    """Normalize an Apollo people record into ProspectModel."""
    name = raw.get("name") or ""
    first = raw.get("first_name") or (name.split()[0] if name else "")
    last = raw.get("last_name") or (" ".join(name.split()[1:]) if len(name.split()) > 1 else "")
    title = raw.get("title") or ""
    org = raw.get("organization") or {}
    company = raw.get("organization_name") or org.get("name") or ""
    domain = org.get("primary_domain") or raw.get("organization_domain") or ""
    linkedin = raw.get("linkedin_url")
    person_id = raw.get("id") or _stable_id(name, company)

    if not name or not title or not company:
        return None
    try:
        return ProspectModel(
            apollo_id=f"apollo_{person_id}",
            name=name,
            first_name=first,
            last_name=last,
            title=title,
            company=company,
            company_domain=domain,
            linkedin_url=linkedin,
        )
    except Exception as exc:
        logger.debug(f"[ProspectFinder] Apollo parse error for {name}: {exc}")
        return None


# ── Source 2: Explorium → Clay — company list then contacts per domain ─────────

def _size_buckets() -> List[str]:
    buckets = []
    for b in _EMPLOYEE_BUCKETS:
        lo, hi = (int(x) for x in b.split("-"))
        if lo <= icp.employee_max and hi >= icp.employee_min:
            buckets.append(b)
    return buckets or ["11-50", "51-200", "201-500"]


def _explorium_companies(n: int) -> List[Dict[str, Any]]:
    """POST /businesses → list of {domain, name} for ICP-matching companies."""
    if not get_live_setting("VIBE_API_KEY", settings.vibe_api_key):
        return []
    try:
        batch = min(n, 25)
        resp = requests.post(
            f"{EXPLORIUM_BASE}/businesses",
            json={
                "mode": "full",
                "size": batch,
                "page_size": batch,
                "page": 1,
                "filters": {
                    "country_code": {"values": ["us"]},
                    "company_size": {"values": _size_buckets()},
                    "google_category": {"values": icp.google_categories},
                },
            },
            headers=_explorium_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        companies = resp.json().get("data") or []
        logger.info(f"[ProspectFinder] Explorium returned {len(companies)} companies")
        return companies
    except requests.HTTPError as exc:
        body = exc.response.text[:300] if exc.response is not None else ""
        logger.warning(f"[ProspectFinder] Explorium {getattr(exc.response, 'status_code', '?')}: {body}")
        return []
    except Exception as exc:
        logger.warning(f"[ProspectFinder] Explorium company fetch failed: {exc}")
        return []


def _explorium_prospects(business_ids: List[str], limit: int) -> List[Dict[str, Any]]:
    """
    POST /prospects — search ICP contacts directly by job title + location.
    business_ids are used as a soft hint but we fall back to title-only if they yield nothing.
    """
    batch = min(limit, 25)

    def _call(filters: dict) -> List[Dict[str, Any]]:
        try:
            resp = requests.post(
                f"{EXPLORIUM_BASE}/prospects",
                json={"mode": "full", "size": batch, "page_size": batch, "page": 1, "filters": filters},
                headers=_explorium_headers(),
                timeout=30,
            )
            resp.raise_for_status()
            people = resp.json().get("data") or []
            logger.info(f"[ProspectFinder] Explorium prospects: {len(people)} contacts")
            return people
        except requests.HTTPError as exc:
            body = exc.response.text[:300] if exc.response is not None else ""
            logger.warning(f"[ProspectFinder] Explorium /prospects {getattr(exc.response, 'status_code', '?')}: {body}")
            return []
        except Exception as exc:
            logger.warning(f"[ProspectFinder] Explorium prospects failed: {exc}")
            return []

    # First try: narrow (business_ids + title)
    if business_ids:
        results = _call({
            "business_id": {"type": "includes", "values": business_ids},
            "job_title": {"type": "any_match_phrase", "values": icp.titles},
        })
        if results:
            return results

    # Fallback: title + country only (broader)
    logger.info("[ProspectFinder] Explorium narrow search empty — trying title-only search")
    return _call({
        "job_title": {"type": "any_match_phrase", "values": icp.titles},
        "country_code": {"values": ["us"]},
    })


def _parse_explorium_prospect(raw: Dict[str, Any]) -> Optional[ProspectModel]:
    """Normalize Explorium prospect record into ProspectModel."""
    name = raw.get("full_name") or ""
    if not name:
        return None
    parts = name.strip().split(" ", 1)
    first = parts[0]
    last = parts[1] if len(parts) > 1 else ""
    title = raw.get("job_title") or ""
    company = raw.get("company_name") or ""
    domain = (raw.get("company_website") or raw.get("domain") or "").replace("https://", "").replace("http://", "").rstrip("/")
    linkedin_arr = raw.get("linkedin_url_array") or []
    linkedin = (linkedin_arr[0] if linkedin_arr else None) or raw.get("linkedin") or raw.get("linkedin_url")
    person_id = raw.get("prospect_id") or _stable_id(name, company)
    emails = raw.get("emails") or []
    email = emails[0] if emails else raw.get("email")

    if not name or not title or not company:
        return None
    try:
        return ProspectModel(
            apollo_id=f"vibe_{person_id}",
            name=name, first_name=first, last_name=last,
            title=title, company=company, company_domain=domain,
            linkedin_url=linkedin,
            prefetched_email=email,
        )
    except Exception as exc:
        logger.debug(f"[ProspectFinder] Explorium parse error for {name}: {exc}")
        return None


def _clay_find_at_company(domain: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    POST /sources/person/find-at-company
    Returns (contacts_list, taskId). taskId stored on ProspectModel for enrichment.
    """
    if not get_live_setting("CLAY_API_KEY", settings.clay_api_key):
        return [], None
    logger.info(f"[ProspectFinder] Clay find-at-company: {domain}")
    try:
        resp = _clay_request(
            "POST",
            f"{CLAY_BASE}/sources/person/find-at-company",
            json={
                "companyIdentifier": domain,
                "contactFilters": {
                    "job_title_keywords": icp.titles,
                    "locations": ["United States"],
                    "job_title_exclude_keywords": ["Intern", "Contractor"],
                },
            },
            timeout=30,
        )
        data = resp.json()
        task_id = data.get("taskId") or data.get("task_id")
        contacts = data.get("contacts") or data.get("results") or data.get("data") or []
        logger.info(f"[ProspectFinder] Clay: {len(contacts)} contacts at {domain} (taskId={task_id})")
        return contacts, task_id
    except Exception as exc:
        logger.warning(f"[ProspectFinder] Clay find-at-company failed for {domain}: {exc}")
        return [], None


def _parse_clay_contact(
    raw: Dict[str, Any],
    company_name: str,
    company_domain: str,
    task_id: Optional[str],
) -> Optional[ProspectModel]:
    name = raw.get("name") or f"{raw.get('first_name', '')} {raw.get('last_name', '')}".strip()
    if not name:
        return None
    parts = name.strip().split(" ", 1)
    first = parts[0]
    last = parts[1] if len(parts) > 1 else ""
    title = raw.get("title") or raw.get("job_title") or ""
    if not title:
        return None
    linkedin = raw.get("linkedin_url") or raw.get("linkedin")
    person_id = raw.get("id") or raw.get("contact_id") or _stable_id(name, company_name)
    try:
        return ProspectModel(
            apollo_id=f"clay_{person_id}",
            name=name, first_name=first, last_name=last,
            title=title, company=company_name, company_domain=company_domain,
            linkedin_url=linkedin, clay_task_id=task_id,
        )
    except Exception as exc:
        logger.debug(f"[ProspectFinder] Clay parse error for {name}: {exc}")
        return None


def _matches_icp(title: str) -> bool:
    t = title.lower()
    return any(kw.lower() in t for kw in icp.titles)


# ── Main entry point ───────────────────────────────────────────────────────────

def find_prospects(limit: int = 20) -> List[ProspectModel]:
    """
    Multi-source prospect finding (priority order):
      1. Apollo people search → direct ICP contacts by title + industry
      2. Explorium companies → domains → Clay find-at-company (fallback / top-up)
    Per-day company dedup (Clay). Name+company dedup in-memory. Contacted filter via SQLite.
    """
    logger.info(f"[ProspectFinder] Searching for up to {limit} ICP prospects")

    seen: set = set()
    prospects: List[ProspectModel] = []

    # ── Source 1: Apollo ──────────────────────────────────────────────────────
    for raw in _apollo_search(limit * 2):
        if len(prospects) >= limit:
            break
        p = _parse_apollo(raw)
        if p is None:
            continue
        key = _stable_id(p.name, p.company)
        if key in seen or is_contacted(p.apollo_id):
            continue
        seen.add(key)
        prospects.append(p)

    logger.info(f"[ProspectFinder] Apollo contributed {len(prospects)} prospects")

    # ── Source 2: Explorium businesses → Explorium prospects ─────────────────
    if len(prospects) < limit:
        needed = limit - len(prospects)
        companies = _explorium_companies(min(needed * 3, 25))

        # Collect business_ids for the prospects call; mark domains as searched
        biz_ids = []
        for company in companies:
            biz_id = company.get("business_id")
            domain = (company.get("domain") or company.get("website") or "").strip()
            if biz_id:
                biz_ids.append(biz_id)
            if domain:
                mark_company_searched(domain)

        for raw in _explorium_prospects(biz_ids, needed):
            if len(prospects) >= limit:
                break
            p = _parse_explorium_prospect(raw)
            if p is None:
                continue
            key = _stable_id(p.name, p.company)
            if key in seen or is_contacted(p.apollo_id):
                continue
            seen.add(key)
            prospects.append(p)

    logger.info(f"[ProspectFinder] {len(prospects)} unique uncontacted prospects ready")
    return prospects
