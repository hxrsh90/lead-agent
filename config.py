from typing import List, Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class ICPConfig:
    titles: List[str] = [
        "Practice Manager", "Revenue Cycle Director", "RCM Manager",
        "Practice Administrator", "VP Revenue Operations", "Director of Revenue Cycle",
        "Director of Revenue Operations", "VP of Revenue Cycle",
        "Chief Revenue Officer", "Patient Access Director",
    ]
    industries: List[str] = [
        "Hospitals and Health Care",
        "Medical Practices",
        "Health, Wellness and Fitness",
        "Mental Health Care",
        "Wellness and Fitness Services",
    ]
    google_categories: List[str] = [
        "hospital",
        "medical clinic",
        "medical practice",
        "urgent care center",
        "physical therapist",
        "medical group",
        "health system",
        "physician",
    ]
    employee_min: int = 10
    employee_max: int = 500
    job_change_days: int = 90
    linkedin_post_days: int = 14
    company_news_days: int = 30
    rcm_job_signals: List[str] = [
        "Prior Authorization", "Patient Access", "Benefits Verification",
        "RCM Coordinator", "Revenue Cycle", "Billing Specialist",
        "Insurance Verification", "Claims Specialist",
    ]


class Settings(BaseSettings):
    agent_mode: Literal["claude", "custom"] = Field(default="custom", alias="AGENT_MODE")

    # Supported: anthropic | openrouter | openai | nim | bedrock
    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="mistralai/mixtral-8x7b-instruct", alias="OPENROUTER_MODEL"
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    nim_api_key: str = Field(default="", alias="NIM_API_KEY")
    nim_model: str = Field(default="nvidia/llama-3.1-nemotron-70b-instruct", alias="NIM_MODEL")
    nim_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NIM_BASE_URL")

    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0", alias="BEDROCK_MODEL_ID"
    )

    apollo_api_key: str = Field(default="", alias="APOLLO_API_KEY")
    clay_api_key: str = Field(default="", alias="CLAY_API_KEY")
    vibe_api_key: str = Field(default="", alias="VIBE_API_KEY")
    slack_webhook_url: str = Field(default="", alias="SLACK_WEBHOOK_URL")

    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_from_name: str = Field(default="VoiceCare.ai", alias="SMTP_FROM_NAME")

    daily_prospect_limit: int = Field(default=20, alias="DAILY_PROSPECT_LIMIT")
    quality_threshold: float = Field(default=2.4, alias="QUALITY_THRESHOLD")

    db_path: str = Field(default="voicecare.db", alias="DB_PATH")

    @model_validator(mode="after")
    def validate_required_keys(self) -> "Settings":
        import warnings
        if self.agent_mode == "claude" and not self.anthropic_api_key:
            warnings.warn("ANTHROPIC_API_KEY not set — claude mode will fail at runtime")
        if self.agent_mode == "custom" and not self.openrouter_api_key:
            warnings.warn("OPENROUTER_API_KEY not set — set it via Settings tab or env var")
        return self

    model_config = {"env_file": ".env", "populate_by_name": True}


settings = Settings()
icp = ICPConfig()
