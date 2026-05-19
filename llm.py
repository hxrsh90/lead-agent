import time
from contextvars import ContextVar
from typing import Any, Optional
from loguru import logger
from config import settings
from database import get_live_setting

# Per-request mode override. Set by run_pipeline(); falls back to settings.agent_mode.
_mode_override: ContextVar[Optional[str]] = ContextVar("mode_override", default=None)


def get_active_mode() -> str:
    return _mode_override.get() or settings.agent_mode

CLAY_MCP_URL = "https://api.clay.com/v3/mcp"
VIBE_MCP_URL = "https://vibeprospecting.explorium.ai/mcp"


def _anthropic_key() -> str:
    return get_live_setting("ANTHROPIC_API_KEY", settings.anthropic_api_key)


def _openrouter_key() -> str:
    return get_live_setting("OPENROUTER_API_KEY", settings.openrouter_api_key)


def _openai_key() -> str:
    return get_live_setting("OPENAI_API_KEY", settings.openai_api_key)


def _nim_key() -> str:
    return get_live_setting("NIM_API_KEY", settings.nim_api_key)


def _get_provider() -> str:
    return get_live_setting("LLM_PROVIDER", settings.llm_provider)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    use_mcp: bool = False,
    max_tokens: int = 500,
    temperature: float = 0.7,
    step_name: str = "unknown",
) -> str:
    """
    Central LLM router. Routes to Anthropic (claude mode) or OpenRouter (custom mode).
    use_mcp=True attaches Clay + Vibe MCP servers in claude mode only.
    """
    start = time.time()
    result = ""
    active_mode = get_active_mode()
    provider = _get_provider() if active_mode != "claude" else "anthropic"
    try:
        if active_mode == "claude":
            result = _call_anthropic(system_prompt, user_prompt, use_mcp, max_tokens, temperature)
        elif provider == "openai":
            result = _call_openai_compatible(
                "https://api.openai.com/v1", _openai_key(),
                get_live_setting("OPENAI_MODEL", settings.openai_model),
                system_prompt, user_prompt, max_tokens, temperature,
            )
        elif provider == "nim":
            result = _call_openai_compatible(
                get_live_setting("NIM_BASE_URL", settings.nim_base_url), _nim_key(),
                get_live_setting("NIM_MODEL", settings.nim_model),
                system_prompt, user_prompt, max_tokens, temperature,
            )
        elif provider == "anthropic":
            result = _call_anthropic(system_prompt, user_prompt, False, max_tokens, temperature)
        elif provider == "bedrock":
            result = _call_bedrock(system_prompt, user_prompt, max_tokens, temperature)
        else:
            result = _call_openrouter(system_prompt, user_prompt, max_tokens, temperature)
    except Exception as exc:
        logger.error(f"[llm] step={step_name} mode={active_mode} error={exc}")
        raise
    finally:
        latency_ms = int((time.time() - start) * 1000)
        logger.info(
            f"[llm] step={step_name} mode={active_mode} "
            f"use_mcp={use_mcp} latency={latency_ms}ms chars={len(result)}"
        )
    return result


def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    use_mcp: bool,
    max_tokens: int,
    temperature: float,
) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_anthropic_key())
    kwargs: dict = {
        "model": "claude-opus-4-5",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if use_mcp:
        mcp_servers = []
        clay_key = get_live_setting("CLAY_API_KEY", settings.clay_api_key)
        vibe_key = get_live_setting("VIBE_API_KEY", settings.vibe_api_key)
        if clay_key:
            mcp_servers.append({
                "type": "url", "url": CLAY_MCP_URL, "name": "clay",
                "authorization_token": clay_key,
            })
        if vibe_key:
            mcp_servers.append({
                "type": "url", "url": VIBE_MCP_URL, "name": "vibe_prospecting",
                "authorization_token": vibe_key,
            })
        if mcp_servers:
            kwargs["mcp_servers"] = mcp_servers
    response = client.messages.create(**kwargs)
    for block in reversed(response.content):
        if hasattr(block, "text"):
            return block.text.strip()
    return ""


def _call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _call_openrouter(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    model = get_live_setting("OPENROUTER_MODEL", settings.openrouter_model)
    return _call_openai_compatible(
        "https://openrouter.ai/api/v1", _openrouter_key(), model,
        system_prompt, user_prompt, max_tokens, temperature,
    )


def _call_bedrock(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    import boto3
    import json as _json
    region = get_live_setting("AWS_REGION", settings.aws_region)
    model_id = get_live_setting("BEDROCK_MODEL_ID", settings.bedrock_model_id)
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=get_live_setting("AWS_ACCESS_KEY_ID", settings.aws_access_key_id),
        aws_secret_access_key=get_live_setting("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key),
    )
    body = _json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    })
    resp = client.invoke_model(body=body, modelId=model_id)
    data = _json.loads(resp["body"].read())
    return data["content"][0]["text"].strip()
