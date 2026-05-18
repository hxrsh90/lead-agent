from typing import TypedDict, Optional, List, Dict, Literal, Any
from pydantic import BaseModel, Field


class ProspectModel(BaseModel):
    apollo_id: str
    name: str
    first_name: str
    last_name: str
    title: str
    company: str
    company_domain: str
    linkedin_url: Optional[str] = None
    clay_task_id: Optional[str] = None
    prefetched_email: Optional[str] = None


class EventModel(BaseModel):
    type: str
    company: Optional[str] = None
    title: Optional[str] = None
    new_title: Optional[str] = None
    years: Optional[int] = None
    date: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class EnrichedProspectModel(BaseModel):
    apollo_id: str
    name: str
    first_name: str
    last_name: str
    title: str
    company: str
    company_domain: str
    linkedin_url: Optional[str] = None
    email: Optional[str] = None
    work_history: Optional[str] = None
    thought_leadership: Optional[str] = None
    tech_stack: List[str] = Field(default_factory=list)
    open_rcm_jobs: List[str] = Field(default_factory=list)
    linkedin_pain_post: Optional[str] = None
    company_news: Optional[str] = None
    events: List[EventModel] = Field(default_factory=list)
    enrichment_source: Literal[
        "clay", "vibe", "both", "claude", "none",
        "apollo", "apollo_no_data",
        "explorium", "explorium_no_email",
        "clay_failed",
    ] = "none"


class SignalModel(BaseModel):
    type: str
    content: str
    date: Optional[str] = None
    priority: int
    score: float = 0.0


class ScoreResultModel(BaseModel):
    specificity: float
    relevance: float
    naturalness: float
    total: float
    status: Literal["approved", "review_needed"]
    feedback: str


class AgentState(TypedDict):
    prospect: ProspectModel
    run_id: str
    enriched: Optional[EnrichedProspectModel]
    signals: List[SignalModel]
    current_signal_index: int
    current_signal: Optional[SignalModel]
    current_message: Optional[str]
    attempt_number: int
    previous_feedback: Optional[str]
    score_result: Optional[ScoreResultModel]
    final_status: Optional[str]
    route_decision: Optional[str]  # set by nodes; read by conditional edges
    step_latencies: Dict[str, float]
    errors: List[str]
