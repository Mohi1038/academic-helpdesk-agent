import json
from datetime import datetime, timezone

from helpdesk import config


def log_unanswered(question: str, reason: str) -> dict:
    """Records a question the helpdesk could not answer so a human can follow up. Call it whenever you
    tell the student you do not know, or the question is outside academic matters.

    Args:
        question: The student's question, verbatim.
        reason: Why it could not be answered, e.g. "out of scope: dining" or "not in the documents".

    Returns:
        dict: `status` "logged".
    """
    record_unanswered(question, reason, source="agent")
    return {"status": "logged", "message": "Logged for a human to answer."}


def record_unanswered(question: str, reason: str, source: str) -> None:
    """Append one entry to the unanswered-questions log (also used by the agent safety-net callback)."""
    config.UNANSWERED_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "question": question.strip(), "reason": reason.strip(), "logged_by": source}
    with config.UNANSWERED_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
