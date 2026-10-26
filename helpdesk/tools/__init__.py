"""Agent tools. Each is a plain function with type hints and a docstring; Google ADK derives the
tool schema from them. Every tool returns a dict with a `status` key and, when it returns facts,
a `source` the agent must cite."""

from helpdesk.tools.courses import (
    find_course,
    get_instructor_contact,
    get_my_schedule,
    get_slot_timetable,
    list_courses,
)
from helpdesk.tools.documents import search_documents
from helpdesk.tools.email import draft_email
from helpdesk.tools.unanswered import log_unanswered

__all__ = [
    "search_documents", "find_course", "list_courses", "get_slot_timetable", "get_my_schedule",
    "get_instructor_contact", "draft_email", "log_unanswered",
]
