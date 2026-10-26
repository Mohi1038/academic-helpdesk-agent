"""Run a helpdesk agent (single or multi) and collect what happened in each turn.

Shared by the CLI, the Streamlit app and the evaluation harness.
"""

from __future__ import annotations

import asyncio
import importlib
import time
import uuid
from dataclasses import dataclass, field

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from helpdesk import config

APP_NAME = "iitj_helpdesk"
USER_ID = "student"
BUILDERS = {"single": "helpdesk.agents.single_agent", "multi": "helpdesk.agents.multi_agent"}


@dataclass
class ToolCall:
    agent: str
    name: str
    args: dict
    response: dict | None = None


@dataclass
class Turn:
    question: str
    answer: str
    agents: list[str] = field(default_factory=list)  # agents that produced events, in order
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None

    @property
    def drafts(self) -> list[str]:
        """Email drafts exactly as saved by draft_email (shown verbatim; the model never retypes them)."""
        return [c.response["email"] for c in self.tool_calls
                if c.name == "draft_email" and (c.response or {}).get("status") == "draft_saved"]

    @property
    def route(self) -> str | None:
        """The specialist the orchestrator transferred to, if any."""
        return next((c.args.get("agent_name") for c in self.tool_calls if c.name == "transfer_to_agent"), None)


class Helpdesk:
    def __init__(self, mode: str = "single", model: str | None = None):
        self.mode = mode
        self.model = model or config.CHAT_MODEL
        agent = importlib.import_module(BUILDERS[mode]).build(self.model)
        self._sessions = InMemorySessionService()
        self._runner = Runner(app_name=APP_NAME, agent=agent, session_service=self._sessions)

    async def new_session(self) -> str:
        session = await self._sessions.create_session(app_name=APP_NAME, user_id=USER_ID,
                                                      session_id=uuid.uuid4().hex)
        return session.id

    async def ask(self, question: str, session_id: str | None = None) -> Turn:
        session_id = session_id or await self.new_session()
        turn, start = Turn(question=question, answer=""), time.perf_counter()
        pending: dict[str, ToolCall] = {}
        message = types.Content(role="user", parts=[types.Part.from_text(text=question)])
        try:
            async for event in self._runner.run_async(user_id=USER_ID, session_id=session_id,
                                                      new_message=message):
                if event.author not in turn.agents and event.author != "user":
                    turn.agents.append(event.author)
                for call in event.get_function_calls():
                    pending[call.id] = ToolCall(event.author, call.name, dict(call.args or {}))
                    turn.tool_calls.append(pending[call.id])
                for resp in event.get_function_responses():
                    if resp.id in pending:
                        pending[resp.id].response = resp.response
                if event.is_final_response() and event.content and event.content.parts:
                    text = "".join(p.text or "" for p in event.content.parts if not p.thought)
                    if text.strip():
                        turn.answer = text.strip()
        except Exception as exc:  # surface model/server failures as a turn result, not a crash
            turn.error = f"{type(exc).__name__}: {exc}"
        turn.latency_s = round(time.perf_counter() - start, 2)
        return turn

    def ask_sync(self, question: str, session_id: str | None = None) -> Turn:
        return asyncio.run(self.ask(question, session_id))
