import asyncio
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from flask import Flask, jsonify, request, render_template
from loguru import logger

from config import settings
from database import (
    init_db, save_run_stats,
    get_approved_messages, get_review_queue, get_run_history,
    get_dashboard_stats, get_approved_by_id,
    get_all_app_settings, set_app_setting, delete_app_setting,
)
from graph import build_graph
from tools.prospect_finder import find_prospects
from agents.state import AgentState
from notifier import send_slack_summary
from mailer import send_message_email
from llm import _mode_override, get_active_mode

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/{time:YYYY-MM-DD}.json", serialize=True, rotation="1 day", retention="30 days")

app = Flask(__name__)


@app.get("/")
def dashboard():
    init_db()
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "mode": get_active_mode()}), 200


@app.get("/api/stats")
def api_stats():
    try:
        init_db()
        stats = get_dashboard_stats()
        stats["mode"] = get_active_mode()
        return jsonify(stats)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/approved")
def api_approved():
    try:
        init_db()
        return jsonify(get_approved_messages())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/review")
def api_review():
    try:
        init_db()
        return jsonify(get_review_queue())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/runs")
def api_runs():
    try:
        init_db()
        return jsonify(get_run_history())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/send/<int:record_id>")
def api_send(record_id: int):
    try:
        init_db()
        record = get_approved_by_id(record_id)
        if not record:
            return jsonify({"success": False, "error": "Message not found."}), 404
        result = send_message_email(record)
        return jsonify(result), 200 if result["success"] else 502
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


_SENSITIVE = {
    "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
    "NIM_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "APOLLO_API_KEY", "CLAY_API_KEY", "VIBE_API_KEY", "SMTP_PASSWORD",
}


@app.get("/api/config")
def api_config_get():
    try:
        init_db()
        rows = get_all_app_settings()
        out = {}
        for key, value in rows.items():
            masked = ("•" * 8) if key in _SENSITIVE and value else value
            out[key] = {"value": masked, "source": "db"}
        return jsonify(out)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/config")
def api_config_post():
    try:
        init_db()
        data = request.get_json(silent=True) or {}
        saved = []
        for key, value in data.items():
            if value and not str(value).startswith("•"):
                set_app_setting(key, str(value))
                saved.append(key)
        return jsonify({"saved": saved})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.delete("/api/config/<key>")
def api_config_delete(key: str):
    try:
        init_db()
        delete_app_setting(key)
        return jsonify({"deleted": key})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/run")
def trigger_run():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode")  # optional override: "claude" or "custom"
    if mode and mode not in ("claude", "custom"):
        return jsonify({"error": f"Invalid mode '{mode}'. Use 'claude' or 'custom'."}), 400
    stats = asyncio.run(run_pipeline(mode_override=mode))
    return jsonify(stats), 200


async def run_pipeline(mode_override: Optional[str] = None) -> Dict[str, Any]:
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    started_at = datetime.now(timezone.utc).isoformat()
    start_ts = datetime.now(timezone.utc).timestamp()

    token = _mode_override.set(mode_override) if mode_override else None
    active_mode = mode_override or settings.agent_mode
    logger.info(f"=== VoiceCare.ai Lead Agent | run_id={run_id} | mode={active_mode} ===")

    init_db()
    graph = build_graph()

    prospects = find_prospects(limit=settings.daily_prospect_limit)
    logger.info(f"Found {len(prospects)} new prospects")

    stats: Dict[str, Any] = {
        "run_id": run_id,
        "mode": active_mode,
        "prospects_found": len(prospects),
        "enriched": 0,
        "approved": 0,
        "review_needed": 0,
        "skipped": 0,
        "errors": 0,
        "started_at": started_at,
        "completed_at": "",
        "duration_seconds": 0.0,
    }

    if not prospects:
        logger.warning("No new prospects found — run complete")
        stats["completed_at"] = datetime.now(timezone.utc).isoformat()
        send_slack_summary(stats)
        return stats

    async def run_prospect(prospect) -> str:
        thread_id = f"{run_id}-{prospect.apollo_id}"
        config = {"configurable": {"thread_id": thread_id}}
        initial: AgentState = {
            "prospect": prospect,
            "run_id": run_id,
            "enriched": None,
            "signals": [],
            "current_signal_index": 0,
            "current_signal": None,
            "current_message": None,
            "attempt_number": 1,
            "previous_feedback": None,
            "score_result": None,
            "final_status": None,
            "route_decision": None,
            "step_latencies": {},
            "errors": [],
        }
        try:
            result = await graph.ainvoke(initial, config=config)
            return result.get("final_status") or "unknown"
        except Exception as exc:
            logger.error(f"Agent thread crashed for {prospect.name}: {exc}")
            return "error"

    results = await asyncio.gather(*[run_prospect(p) for p in prospects])

    for status in results:
        if status == "approved":
            stats["approved"] += 1
            stats["enriched"] += 1
        elif status == "review_needed":
            stats["review_needed"] += 1
            stats["enriched"] += 1
        elif status == "skipped":
            stats["skipped"] += 1
        else:
            stats["errors"] += 1

    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    stats["duration_seconds"] = round(datetime.now(timezone.utc).timestamp() - start_ts, 1)

    logger.info(
        f"=== Run complete | approved={stats['approved']} review={stats['review_needed']} "
        f"skipped={stats['skipped']} errors={stats['errors']} "
        f"duration={stats['duration_seconds']}s ==="
    )

    if token is not None:
        _mode_override.reset(token)
    save_run_stats(stats)
    send_slack_summary(stats)
    return stats


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
