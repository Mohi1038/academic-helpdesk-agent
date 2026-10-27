"""Baseline: one agent with every tool."""

from google.adk.agents import LlmAgent

from helpdesk import tools
from helpdesk.agents import prompts
from helpdesk.agents.callbacks import log_refusals
from helpdesk.agents.model import make_model


def build(model_name: str | None = None) -> LlmAgent:
    return LlmAgent(
        name="helpdesk_agent",
        model=make_model(model_name),
        description="IIT Jodhpur academic helpdesk (single agent).",
        instruction=prompts.SINGLE_AGENT,
        tools=[getattr(tools, name) for name in tools.__all__],
        after_agent_callback=log_refusals,
    )
