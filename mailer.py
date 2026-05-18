import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
from loguru import logger
from config import settings


def send_message_email(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send the approved LinkedIn message as an email to the prospect.
    Returns {"success": bool, "error": str | None}.
    """
    if not settings.smtp_user or not settings.smtp_password:
        return {"success": False, "error": "SMTP_USER or SMTP_PASSWORD not configured."}

    to_email = record.get("email", "")
    if not to_email:
        return {"success": False, "error": "No email address for this prospect."}

    name = record.get("name", "there")
    company = record.get("company", "")
    message_body = record.get("message", "")
    signal_type = record.get("signal_type", "").replace("_", " ").title()

    subject = f"Quick note — {name} at {company}"

    html_body = f"""\
<html><body style="font-family: Arial, sans-serif; font-size: 14px; color: #222; max-width: 600px;">
  <p>{message_body}</p>
  <br>
  <hr style="border: none; border-top: 1px solid #eee;">
  <p style="font-size: 11px; color: #999;">
    Sent via VoiceCare.ai Lead Agent &nbsp;|&nbsp;
    Signal: {signal_type} &nbsp;|&nbsp;
    Score: {record.get('total_score', 0):.2f}/3.0
  </p>
</body></html>"""

    plain_body = message_body

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_user}>"
    msg["To"] = to_email

    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_user, to_email, msg.as_string())
        logger.info(f"[Mailer] Email sent to {name} <{to_email}>")
        return {"success": True, "error": None}
    except Exception as exc:
        logger.error(f"[Mailer] Failed to send email to {to_email}: {exc}")
        return {"success": False, "error": str(exc)}
