from loguru import logger
from agents.state import AgentState
from llm import call_llm

VOICECARE_SYSTEM_PROMPT = """\
You are writing a personalized LinkedIn connection message for someone selling an AI voice \
agent that autonomously handles prior authorization calls, benefits verification calls, and \
claim status calls for healthcare practices — eliminating 60-80% of admin phone time. \
It is HIPAA compliant, never skips a call, and is backed by Mayo Clinic.

Your target readers are Practice Managers, Revenue Cycle Directors, RCM Managers, and similar \
healthcare admin leaders who are drowning in insurance phone calls.

HARD RULES — violate any of these and the message will be rejected:
1. Total message MUST be 300 characters or fewer (including spaces). Count carefully.
2. This is a LinkedIn message, NOT an email. No subject line. No sign-off.
3. First sentence references the specific personalization signal naturally and precisely.
4. Second sentence connects their situation to the admin phone call problem.
5. End with ONE soft question — not a pitch, not a hard CTA.
6. NEVER use generic openers: "I hope this finds you well", "I came across your profile", \
"I wanted to reach out", "I noticed your impressive background".
7. Sound like a peer colleague, not a sales representative.
8. NEVER fabricate details not present in the provided signal.
9. Do NOT name the product or company. Describe what it does instead.
10. Output ONLY the message text. No quotes, no labels, no explanation.\
"""


def writer_agent(state: AgentState) -> AgentState:
    """
    WriterAgent node.
    Generates a LinkedIn message grounded in the current signal.
    Injects CriticAgent feedback on retry attempts (attempt > 1).
    """
    enriched = state["enriched"]
    signal = state["current_signal"]
    attempt = state["attempt_number"]
    previous_feedback = state.get("previous_feedback")
    errors = list(state.get("errors") or [])

    logger.info(f"[WriterAgent] {enriched.name} | signal={signal.type} | attempt={attempt}")

    user_prompt = (
        f"Write a LinkedIn message to {enriched.first_name} {enriched.last_name}, "
        f"who is {enriched.title} at {enriched.company}.\n\n"
        f"Personalization signal to use:\n{signal.content}\n"
    )

    if previous_feedback and attempt > 1:
        user_prompt += (
            f"\nPrevious attempt was REJECTED by the quality critic.\n"
            f"Critic feedback: {previous_feedback}\n"
            f"Fix those specific issues. Do not repeat the same mistake."
        )

    try:
        message = call_llm(
            system_prompt=VOICECARE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            use_mcp=False,
            max_tokens=150,
            temperature=0.75 if attempt == 1 else 0.6,
            step_name=f"writer_attempt_{attempt}",
        )
        message = message.strip().strip('"').strip("'")
    except Exception as exc:
        logger.error(f"[WriterAgent] LLM call failed for {enriched.name}: {exc}")
        errors.append(f"writer_error_attempt_{attempt}: {exc}")
        message = ""

    return {**state, "current_message": message, "errors": errors}
