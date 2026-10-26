"""Multi-agent helpdesk: an orchestrator transfers each message to one specialist.

    helpdesk_orchestrator --transfer_to_agent--> policy_agent   (search_documents)
                                             --> schedule_agent (find_course, list_courses, slots, my schedule)
                                             --> email_agent    (get_instructor_contact, draft_email)
    out-of-scope: the orchestrator refuses and logs the question itself.

Specialists may not transfer onwards (to the parent or to peers). ADK then hands control back to the
orchestrator at the next user message, so every message is routed afresh and a small model cannot
bounce a question around the tree.
"""

from google.adk.agents import LlmAgent

from helpdesk import tools
from helpdesk.agents import prompts
from helpdesk.agents.callbacks import log_refusals
from helpdesk.agents.model import make_model


def _specialist(name: str, description: str, instruction: str, agent_tools: list, model_name: str | None):
    return LlmAgent(
        name=name,
        model=make_model(model_name),
        description=description,
        instruction=instruction,
        tools=[*agent_tools, tools.log_unanswered],
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        after_agent_callback=log_refusals,
    )


def build(model_name: str | None = None) -> LlmAgent:
    policy = _specialist(
        "policy_agent",
        "Answers questions on academic rules and regulations, programmes, curricula, credit requirements, "
        "the academic calendar, fees and scholarships from official documents.",
        prompts.POLICY_AGENT, [tools.search_documents], model_name)
    schedule = _specialist(
        "schedule_agent",
        "Answers questions on this semester's course offerings, slots, lecture times, rooms, course counts "
        "and the student's own timetable.",
        prompts.SCHEDULE_AGENT,
        [tools.find_course, tools.list_courses, tools.get_slot_timetable, tools.get_my_schedule], model_name)
    email = _specialist(
        "email_agent",
        "Finds course instructors' contact details and drafts emails to them (leave, make-up exams).",
        prompts.EMAIL_AGENT, [tools.get_instructor_contact, tools.draft_email], model_name)

    return LlmAgent(
        name="helpdesk_orchestrator",
        model=make_model(model_name),
        description="IIT Jodhpur academic helpdesk front desk; routes questions to specialists.",
        instruction=prompts.ORCHESTRATOR,
        tools=[tools.log_unanswered],
        sub_agents=[policy, schedule, email],
        after_agent_callback=log_refusals,
    )
