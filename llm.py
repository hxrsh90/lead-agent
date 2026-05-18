import time
from contextvars import ContextVar
from typing import Any, Optional
from loguru import logger
from config import settings

# Per-request mode override. Set by run_pipeline(); falls back to settings.agent_mode.
_mode_override: ContextVar[Optional[str]] = ContextVar("mode_override", default=None)


def get_active_mode() -> str:
    return _mode_override.get() or settings.agent_mode

_anthropic_client: Any = None
_openai_client: Any = None

CLAY_MCP_URL = "https://api.clay.com/v3/mcp"
VIBE_MCP_URL = "https://vibeprospecting.explorium.ai/mcp"


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
    return _openai_client


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
    try:
        if active_mode == "claude":
            result = _call_anthropic(system_prompt, user_prompt, use_mcp, max_tokens, temperature)
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
    client = _get_anthropic()
    kwargs: dict = {
        "model": "claude-opus-4-5",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if use_mcp:
        mcp_servers = []
        if settings.clay_api_key:
            mcp_servers.append({
                "type": "url", "url": CLAY_MCP_URL, "name": "clay",
                "authorization_token": settings.clay_api_key,
            })
        if settings.vibe_api_key:
            mcp_servers.append({
                "type": "url", "url": VIBE_MCP_URL, "name": "vibe_prospecting",
                "authorization_token": settings.vibe_api_key,
            })
        if mcp_servers:
            kwargs["mcp_servers"] = mcp_servers
    response = client.messages.create(**kwargs)
    for block in reversed(response.content):
        if hasattr(block, "text"):
            return block.text.strip()
    return ""


def _call_openrouter(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    client = _get_openai()
    response = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
