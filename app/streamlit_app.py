"""Streamlit UI for the IIT Jodhpur Academic Helpdesk.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpdesk import catalog, config  # noqa: E402
from helpdesk.runner import Helpdesk, Turn  # noqa: E402

MODES = {"Multi-agent": "multi", "Single agent": "single"}
DEFAULT_MODELS = ["qwen3:8b", "llama3.1:8b", "granite3.3:8b"]
EXAMPLES = [
    "What is the attendance requirement to sit in the Major exam?",
    "What is the slot for the course Autonomous Systems?",
    "How many 400 level electives have been floated by the Humanities department in this semester?",
    "How many Open Elective credits do I need to earn in order to get a B.Tech. Degree?",
    "Write an email to the course instructor of CSL7570 that I need leave for 2 days",
]

st.set_page_config(page_title="IITJ Academic Helpdesk", page_icon="🎓")


# --- Agent plumbing ---------------------------------------------------------------------------------

@st.cache_resource
def event_loop() -> asyncio.AbstractEventLoop:
    """One long-lived loop for all agent calls; LiteLLM's async clients must not outlive their loop."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def run(coro):
    return asyncio.run_coroutine_threadsafe(coro, event_loop()).result()


@st.cache_resource
def helpdesk(mode: str, model: str) -> Helpdesk:
    return Helpdesk(mode, model)


@st.cache_data(ttl=60)
def available_models() -> list[str]:
    try:
        tags = requests.get(f"{config.OLLAMA_API_BASE}/api/tags", timeout=3).json()["models"]
        names = sorted(m["name"].removesuffix(":latest") for m in tags if "embed" not in m["name"])
        return names or DEFAULT_MODELS
    except (requests.RequestException, KeyError, ValueError):
        return DEFAULT_MODELS


def session_id(desk: Helpdesk) -> str:
    """One ADK session per browser session and agent configuration."""
    key = f"adk_session:{desk.mode}:{desk.model}"
    if key not in st.session_state:
        st.session_state[key] = run(desk.new_session())
    return st.session_state[key]


def sources(turn: Turn) -> list[dict]:
    """Citations behind an answer, taken from the tool responses (not from the model's text)."""
    seen, out = set(), []
    for call in turn.tool_calls:
        resp = call.response or {}
        if call.name == "search_documents":
            items = [{"label": r["citation"], "url": r["url"], "text": r["text"]} for r in resp.get("results", [])]
        elif resp.get("source"):
            items = [{"label": s.strip(), "url": None, "text": None} for s in resp["source"].split(";")]
        else:
            items = []
        for item in items:
            if item["label"] not in seen:
                seen.add(item["label"])
                out.append(item)
    return out


# --- Rendering --------------------------------------------------------------------------------------

def render_turn(turn: Turn, index: int, show_trace: bool) -> None:
    if turn.error:
        st.error(f"The model server did not respond ({turn.error}). Is Ollama running and the SSH tunnel "
                 f"open? ({config.OLLAMA_API_BASE})")
        return
    st.markdown(turn.answer or "_No answer._")

    for d, draft in enumerate(turn.drafts):
        with st.container(border=True):
            st.markdown("**✉️ Email draft** · not sent — edit, then copy or download it")
            edited = st.text_area("Draft", draft, height=240, key=f"draft-{index}-{d}",
                                  label_visibility="collapsed")
            st.download_button("Download .txt", edited, file_name=f"email-draft-{index + 1}.txt",
                               key=f"dl-{index}-{d}")

    meta = " → ".join(turn.agents) + f" · {turn.latency_s}s"
    st.caption(meta)
    cited = sources(turn)
    if cited:
        with st.expander(f"Sources ({len(cited)})"):
            for s in cited:
                link = f"[{s['label']}]({s['url']})" if s["url"] else s["label"]
                st.markdown(f"- {link}")
                if s["text"]:
                    st.caption(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))
    if show_trace and turn.tool_calls:
        with st.expander(f"Tool calls ({len(turn.tool_calls)})"):
            for call in turn.tool_calls:
                st.markdown(f"`{call.agent}` → **{call.name}**")
                st.json({"args": call.args, "response": call.response}, expanded=False)


def chat_tab(desk: Helpdesk, show_trace: bool) -> None:
    messages = st.session_state.setdefault("messages", [])
    if not messages:
        st.markdown("Ask about academic rules, programmes, the calendar, course slots, your timetable, "
                    "or have an email to an instructor drafted. Try:")
        for q in EXAMPLES:
            if st.button(q, key=f"ex-{q}"):
                st.session_state.pending = q
                st.rerun()

    for i, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_turn(msg["turn"], i, show_trace)

    question = st.chat_input("Ask the helpdesk…") or st.session_state.pop("pending", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                turn = run(desk.ask(question, session_id(desk)))
            render_turn(turn, len(messages), show_trace)
        messages.append({"role": "assistant", "turn": turn})


def unanswered_tab() -> None:
    path = config.UNANSWERED_LOG
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    st.markdown("Questions the helpdesk could not answer, for a staff member to follow up. "
                "Tick **resolved** once handled.")
    if not rows:
        st.info("Nothing logged yet.")
        return
    for r in rows:
        r.setdefault("resolved", False)
    edited = st.data_editor(
        rows, key="unanswered", width="stretch", hide_index=True,
        disabled=["timestamp", "question", "reason", "logged_by"],
        column_config={"resolved": st.column_config.CheckboxColumn("resolved")},
    )
    if edited != rows:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in edited), encoding="utf-8")
        st.toast("Saved")
    st.download_button("Download log (JSONL)", path.read_bytes(), file_name="unanswered.jsonl")


def drafts_tab() -> None:
    files = sorted(config.DRAFTS_DIR.glob("*.txt"), reverse=True) if config.DRAFTS_DIR.exists() else []
    if not files:
        st.info("No drafts yet. Ask the helpdesk to write an email to an instructor.")
    for f in files:
        with st.expander(f.stem):
            st.code(f.read_text(encoding="utf-8"), language=None)


# --- Page -------------------------------------------------------------------------------------------

with st.sidebar:
    st.header("🎓 IITJ Academic Helpdesk")
    mode_label = st.radio("Agent", list(MODES), help="Multi-agent: an orchestrator routes each question to "
                          "a policy, schedule or email specialist. Single: one agent with every tool.")
    models = available_models()
    model = st.selectbox("Model", models, index=models.index(config.CHAT_MODEL) if config.CHAT_MODEL in models else 0)
    show_trace = st.toggle("Show tool calls", value=False)
    if st.button("New conversation", width="stretch"):
        for key in [k for k in st.session_state if k.startswith("adk_session:") or k == "messages"]:
            del st.session_state[key]
        st.rerun()

    st.divider()
    profile = catalog.profile()
    st.markdown(f"**Student:** {profile['name']}  \n{profile['programme']}, semester {profile['semester_number']} "
                f"· {profile['academic_semester']}")
    by_code = {c["code"]: c["title"] for c in catalog.courses()}
    st.caption("Registered: " + ", ".join(by_code[c] for c in profile["registered_courses"]))

desk = helpdesk(MODES[mode_label], model)
chat, unanswered, drafts = st.tabs(["Chat", "Unanswered questions", "Email drafts"])
with chat:
    chat_tab(desk, show_trace)
with unanswered:
    unanswered_tab()
with drafts:
    drafts_tab()
