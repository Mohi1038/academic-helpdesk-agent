# academic-helpdesk-agent

Multi-agent academic helpdesk for IIT Jodhpur, built with Google ADK, Ollama-hosted 8B models and a
ChromaDB RAG pipeline. It answers questions on programmes, curricula, rules and regulations, and the timetable.
It also drafts emails to course instructors, and refuses (and logs) questions it cannot answer.

> Work in progress. Phase 1 (data corpus) is complete; Phase 2 (RAG pipeline) is implemented.

## Setup

```bash
uv sync --python 3.12
cp .env.example .env
```

### Model server

The models run in Ollama. It can be installed locally (16 GB RAM is enough for one 8B model) or run on a remote
GPU server.

On the remote server, run this inside the GPU container:

```bash
bash server/setup_ollama.sh        # installs Ollama, serves on the bridge IP, pulls all models
```

Then, on the laptop, keep this tunnel open:

```bash
scripts/tunnel.sh user@gpu-host 172.17.0.4   # localhost:11434 -> container
```

The tunnel goes to the **host**, and the host forwards to the container's bridge IP, so no container port is
published. Ollama has no authentication, so never expose port 11434 publicly.

## Data corpus

```bash
uv run python scripts/fetch_docs.py   # download 29 public IITJ documents -> data/raw, data/manifest.csv
uv run python scripts/ocr_scans.py    # OCR scanned PDFs (needs tesseract) -> data/raw/*.ocr.md
uv run pytest -q                      # integrity checks on the manifest and the structured CSVs
```

- `data/raw/`: 24 PDFs and 5 web pages saved as Markdown. They cover academic regulations, attendance and examination circulars, the 2026-27 academic calendar, holidays, fees, scholarships, forms, and the B.Tech./M.Tech. curricula, including humanities courses.
- `data/manifest.csv`: provenance for each document (title, category, source URL, fetch date, page count, hash).
- `data/structured/`: course offerings, the official 2026-27 Semester I slot system, and the real faculty directory. Course codes are partly synthetic and course-to-instructor assignments are illustrative. See [data/structured/README.md](data/structured/README.md).

## RAG pipeline

```bash
uv run python -m helpdesk.ingest          # chunk -> embed (Ollama) -> data/chroma/
uv run python scripts/check_retrieval.py  # hit@5 / MRR on eval/retrieval_set.jsonl + refusal-threshold calibration
```

- Chunks are about 350 words, follow section boundaries, overlap by one paragraph, and keep page ranges for citations. Repeated page headers and footers are dropped. Scanned PDFs use their OCR text.
- Retrieval is hybrid: dense search (`nomic-embed-text` in ChromaDB) plus BM25, fused with reciprocal rank fusion. BM25 catches exact tokens such as course codes and clause numbers.
- If the best dense similarity is below `MIN_SIMILARITY`, the query is treated as unanswerable from the corpus.

## Single agent (baseline)

```bash
uv run python -m helpdesk.cli "What is the slot for the course Autonomous Systems?"   # one question
uv run python -m helpdesk.cli --model llama3.1:8b                                    # interactive chat
uv run python scripts/smoke_agent.py                                                 # the assignment's sample questions
```

One Google ADK `LlmAgent` has all eight tools and runs an Ollama model through LiteLLM. It must use the
`ollama_chat/` provider; the plain `ollama/` provider loops on tool calls. qwen3's thinking mode is switched off.

- **Grounding:** the prompt requires a tool call before any academic answer, and citations in square brackets. When sources conflict, the newest document wins: circulars and amendments override older regulations.
- **Refusals:** out-of-scope questions get a fixed refusal message and are logged to `logs/unanswered.jsonl`. An `after_agent_callback` also logs any refusal the model forgot to log itself.
- **Emails:** `draft_email` writes to `drafts/` and never sends. It only accepts addresses from the faculty directory, and appends the student's signature.

## Multi-agent helpdesk

```bash
uv run python -m helpdesk.cli --mode multi                 # interactive
uv run python scripts/smoke_agent.py --mode multi
```

```
helpdesk_orchestrator --transfer_to_agent--> policy_agent    search_documents
                                         --> schedule_agent  find_course, list_courses, get_slot_timetable, get_my_schedule
                                         --> email_agent     get_instructor_contact, draft_email
out of scope: the orchestrator refuses and logs the question (log_unanswered + callback safety net)
```

- The orchestrator has no domain tools. Its prompt describes each specialist and gives routing examples.
- Specialists cannot transfer onwards (`disallow_transfer_to_parent` / `_peers`), so ADK returns control to the orchestrator at the next message. Every message is routed afresh, while the session history keeps follow-ups working ("write to *them* that ...").
- Email drafts are shown exactly as the tool saved them. The model only confirms the draft, because when it retyped the email it once garbled the address.

## Streamlit UI

```bash
uv run streamlit run app/streamlit_app.py
```

- **Chat:** each answer shows the route it took (e.g. `helpdesk_orchestrator → schedule_agent`), its latency, and a **Sources** panel built from the tool responses (document citations with links and snippets, or the structured-data source). There are example questions to click, and an optional tool-call trace.
- **Email drafts:** an editable card with a download button. Nothing is ever sent.
- **Unanswered questions:** the log for staff follow-up, with a *resolved* checkbox and a download button.
- **Sidebar:** switch between the multi-agent and single-agent setups, pick any chat model available in Ollama, and start a new conversation.

## Evaluation

```bash
uv run python eval/run_eval.py     # 30 questions x {single, multi} x 3 models x 3 runs -> eval/results/runs/
uv run python eval/judge.py        # Claude judge (needs ANTHROPIC_API_KEY in .env) -> verdicts + run consistency
uv run python eval/score.py        # metrics -> eval/results/summary.md, metrics.csv, per_answer.csv
```

- **Questions:** `eval/questions.jsonl` has 30 questions (12 policy, 8 schedule, 4 email, 6 out-of-scope), each with a reference answer, gold sources, and the expected specialist and tools.
- **Judge:** `claude-sonnet-5-5` at effort `medium`, with structured output. It grades correctness, groundedness (against the tool outputs the agent actually saw), citation quality and refusal, and checks whether the runs of each question agree. Results are cached in `eval/results/judge_cache.jsonl`.
- **Deterministic metrics:** refusal and logging of out-of-scope questions, routing, tool selection, citations that match a returned source, retrieval of the gold source, and latency.

**Report:** [report/report.md](report/report.md). Headline results:

- qwen3:8b scores about 80% in-scope accuracy in both single and multi-agent modes. 94–97% of its answers are grounded, and it refuses and logs 100% of out-of-scope questions.
- llama3.1:8b scores 72% (single) and 60% (multi).
- granite3.3:8b cannot emit structured tool calls through Ollama, and fabricates tool results.
