import requests
from datetime import datetime, timezone
from typing import Dict, Any
from loguru import logger
from config import settings


def send_slack_summary(stats: Dict[str, Any]) -> None:
    """Send a Slack Block Kit run summary. Silently skips if webhook not configured."""
    if not settings.slack_webhook_url:
        logger.warning("[Slack] SLACK_WEBHOOK_URL not set — skipping notification")
        return

    approved = stats.get("approved", 0)
    review = stats.get("review_needed", 0)
    skipped = stats.get("skipped", 0)
    errors = stats.get("errors", 0)
    found = stats.get("prospects_found", 0)
    duration = stats.get("duration_seconds", 0)
    run_id = stats.get("run_id", "unknown")
    mode = stats.get("mode", "unknown")

    status_emoji = "✅" if errors == 0 else "⚠️"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{status_emoji} VoiceCare.ai Lead Agent — {today}"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Mode:*\n`{mode}`"},
                {"type": "mrkdwn", "text": f"*Duration:*\n{duration:.0f}s"},
                {"type": "mrkdwn", "text": f"*Prospects Found:*\n{found}"},
                {"type": "mrkdwn", "text": f"*Run ID:*\n`{run_id}`"},
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*✅ Approved:*\n{approved}"},
                {"type": "mrkdwn", "text": f"*👀 Needs Review:*\n{review}"},
                {"type": "mrkdwn", "text": f"*⏭ Skipped:*\n{skipped}"},
                {"type": "mrkdwn", "text": f"*❌ Errors:*\n{errors}"},
            ],
        },
    ]

    if review > 0:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"⚠️ *{review} message(s) need human review.* Query `review_queue` in `voicecare.db`.",
            },
        })

    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"run_id: `{run_id}` | VoiceCare.ai Lead Agent"}],
    })

    try:
        resp = requests.post(
            settings.slack_webhook_url,
            json={"blocks": blocks},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info(f"[Slack] Summary sent — approved={approved} review={review} errors={errors}")
    except Exception as exc:
        logger.error(f"[Slack] Failed to send run summary: {exc}")


def send_review_alert(name: str, company: str, message: str, score: float, feedback: str) -> None:
    """Send an individual Slack alert for a single review-needed message."""
    if not settings.slack_webhook_url:
        return
    blocks = [
        {"type": "section",
         "text": {"type": "mrkdwn", "text": f"*👀 Review Needed: {name} @ {company}*"}},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": f"*Message:*\n```{message}```"}},
        {"type": "section",
         "fields": [
             {"type": "mrkdwn", "text": f"*Score:* {score:.2f} / 3.0"},
             {"type": "mrkdwn", "text": f"*Why rejected:* {feedback}"},
         ]},
    ]
    try:
        requests.post(settings.slack_webhook_url, json={"blocks": blocks}, timeout=10)
    except Exception as exc:
        logger.error(f"[Slack] Review alert failed for {name}: {exc}")
