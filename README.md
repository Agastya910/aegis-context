# Aegis-Context

Local-first ambient screen memory. A capture daemon records what you're doing, a vision model describes each
distinct screen, and everything is stored in SQLite as frames, searchable chunks, activity episodes and hourly/daily
digests. A loopback FastAPI bridge, a streaming chat HUD and a terminal chat let you ask about it:
"what was I doing 10 minutes ago?", "and before that?", "summarize my afternoon", "what did I ask Claude to do?".

## Quick start (uv)

```bash
uv sync --extra capture --extra hud
uv run pytest                 # unit + integration tests (no network)
uv run aegis demo             # API with in-memory sample data on 127.0.0.1:8765
uv run aegis run              # capture daemon + API + memory jobs (writes ~/.aegis; migrates old dbs, with backup)
uv run aegis hud              # Ctrl+Shift+Space overlay: streaming chat, citations, sessions
uv run aegis chat             # the same agent in the terminal
uv run aegis query "what did I ask claude to do?"     # one-shot (add --raw for retrieval hits only)
uv run aegis timeline --minutes 60
uv run aegis eval             # scripted 9-turn conversation against a synthetic day, real LLM
uv run python scripts/probe_live.py                   # same idea against your live memory
```

## Models

| Role | Default | Config |
|---|---|---|
| Vision (one call per distinct frame: ACTIVITY / TYPED / ENTITIES / TEXT) | `moondream` (local) | `vision_backend`, `vision_model` |
| Embeddings | `nomic-embed-text` via local Ollama (or `hash` / `onnx`) | `embedder`, `embed_model` |
| Chat agent + summaries | local `qwen2.5:7b`; a cloud model if `.env` is present | `chat_model`, `chat_fallback_model` |

`.env` (gitignored; repo dir or `~/.aegis/.env`) switches vision and chat to a hosted Ollama:
`OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL`. **Privacy:** in that mode, screenshots and retrieved screen text
leave the machine, which breaks the PRD's "nothing leaves the device" rule. Delete `.env` (and set `vision_model` back
to `moondream`) to return to fully local operation. Secrets (API keys, tokens, passwords, private keys) are redacted
before anything is stored or sent to an LLM; `aegis redact-existing` scrubs older rows.

## Layout

| Module | Role |
|---|---|
| `capture.py` | Active window (+pid), screenshot + perceptual hash, idle time |
| `filters.py` | Never captures the HUD itself, password managers, private windows; title normalization |
| `daemon.py` | Capture thread (cheap change detection) → queue → process thread (vision, ingest) |
| `ingest.py` | Redaction, near-duplicate merge (keeps the latest typed text), chunking, batch embedding, episodes |
| `storage.py` | SQLite schema + migrations (`PRAGMA user_version`), frames, chunks + FTS5, episodes, digests, chat tables |
| `retrieval.py` | Hybrid search (vectors + BM25, Reciprocal Rank Fusion), time/app filters, timelines, sources, day summaries |
| `tools.py` / `agent.py` | Tool-calling chat agent: current local time, conversation memory, citations `[E12]`/`[F345]`/`[D3]`, offline fallback |
| `llm.py` | Ollama `/api/chat` streaming + tools, retries, cloud → local fallback |
| `jobs.py` | Episode summaries, hourly/daily digests, tiered retention (7 d raw text, 90 d episodes, 365 d digests) |
| `ipc_server.py` | `/chat` (SSE), `/chat/sessions`, `/memory/source/{id}`, `/memory/timeline`, `/context/recent`, `/context/query`, `/health` (loopback only) |
| `hud.py` | PyQt5 overlay: streaming Markdown chat, clickable sources, sessions, excluded from screen capture |

Config lives in `~/.aegis/config.json` (`aegis init` writes defaults). Everything is in one SQLite file
(`aegis_vectors.db`); the PRD's separate `aegis_state.db` is deliberately folded in.

## Not built yet
Android / iOS clients (PRD Part 2 §3.2), user-facts memory, local OCR (Windows.Media.Ocr), pause/forget controls,
running as a login service. See `docs/PLAN-chat-agent.md` Phase 8.
