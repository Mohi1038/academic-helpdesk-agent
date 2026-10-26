"""Safety net: log every refusal for human follow-up, even when the model forgets to call log_unanswered."""

import re

from google.adk.agents.callback_context import CallbackContext

from helpdesk.tools.unanswered import record_unanswered

REFUSAL_PATTERN = re.compile(
    r"\b(I (don't|do not) know|I'm not sure|I am not sure|(cannot|can't|unable to) (answer|help|find))\b",
    re.IGNORECASE,
)


def _text(content) -> str:
    return "".join(p.text or "" for p in (content.parts or [])) if content else ""


def log_refusals(callback_context: CallbackContext) -> None:
    """after_agent_callback: if this turn ended in a refusal and nothing was logged yet, log it."""
    if callback_context.state.get("temp:unanswered_logged"):
        return None
    events = [e for e in callback_context.session.events if e.invocation_id == callback_context.invocation_id]
    if any(call.name == "log_unanswered" for e in events for call in e.get_function_calls()):
        return None
    replies = [_text(e.content) for e in events if e.author != "user" and _text(e.content).strip()]
    if replies and REFUSAL_PATTERN.search(replies[-1]):
        record_unanswered(_text(callback_context.user_content), "refusal not logged by the agent",
                          source=f"callback:{callback_context.agent_name}")
        callback_context.state["temp:unanswered_logged"] = True
    return None
