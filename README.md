# Aegis

**The memory layer for your working life.** Aegis quietly remembers everything you do on your computer, and keeps
it on your computer. Ask it anything, hand any AI a safe brief of where you are, and see where your time
actually goes.

> *"What was I doing before the standup?"* · *"And before that?"* · *"What did I ask Claude to build?"*
> *"How can I optimize my time?"* · *"Get only the safe info to share."* · *"Find an agent to plan my week."*

## Why

Knowledge work is fragmented across editors, terminals, browsers, chat and several AI assistants. Every switch
costs you in one of four ways:
- **re-orienting** ("where was I?");
- **re-finding** ("where did I see that error?");
- **re-explaining** (every new AI chat starts from zero);
- **reporting** (standups, timesheets).

Aegis removes all four, and measures the fragmentation itself so you can fix it.
[Read the product brief →](docs/PRODUCT.md)

## What it does

| | |
|---|---|
| 🧠 **Remembers automatically** | Captures only meaningful screen changes, describes them, and records what you *typed* (the final version, not 30 drafts). No tagging, no notes. |
| ⏱️ **Answers in time** | Knows the current time, so *"10 minutes ago"*, *"this morning"* and *"before that"* just work, with follow-ups and clickable sources. |
| 📋 **Safe context, one click** | **Copy safe context** gives a paste-ready brief for Claude/ChatGPT/Cursor. Secrets and PII are scrubbed, email/chat/meeting/finance content is withheld, and no raw screen text is included. |
| 📈 **Coaches from your data** | Time per category, context switches, refocus cost, deep-work blocks, your best focus hour, then concrete changes. |
| 🤝 **Finds agents for you** | Searches the [Darwin](https://darwin.so) agent marketplace with a *generic* description of the need. Your memory never leaves the device. |
| 🔒 **Local by default** | One SQLite file in `~/.aegis`, an API bound to `127.0.0.1`, and local models (Ollama). Hosted models are an explicit opt-in. |

## Quick start (Windows, ~5 minutes)

```powershell
# 1. Install uv and the app
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/Agastya910/aegis-context; cd aegis-context
uv sync --extra capture --extra hud

# 2. Local models (https://ollama.com)
ollama pull moondream; ollama pull nomic-embed-text; ollama pull qwen2.5:7b
uv run aegis init          # then set in ~/.aegis/config.json:
                           #   "vision_backend": "ollama", "embedder": "ollama"

# 3. Run
uv run aegis run           # window 1: capture + memory + API
uv run aegis hud           # window 2: Ctrl+Shift+Space
```

Try it without capturing anything: `uv run aegis demo`.

**Hosted models (optional, faster on weak hardware):** create `.env` with `OLLAMA_BASE_URL`, `OLLAMA_API_KEY` and
`OLLAMA_MODEL` (e.g. `gemma4:31b-cloud`). In this mode, screenshots and retrieved snippets are sent to that
provider. Delete `.env` to go back to fully local.

## Everyday use

| You want to… | Do this |
|---|---|
| Resume after an interruption | Hotkey → *"what was I doing before …?"* |
| Find something you saw | *"where did I see the database is locked error?"* |
| Brief an AI assistant | **Copy safe context** (or press Enter on an empty box), then paste |
| Write a standup / summary | *"summarize my morning"* · `aegis share --minutes 240` |
| Understand your day | **Insights** · *"how can I optimize my time?"* · `aegis insights` |
| Get outside help | **Find agents** · *"find an agent that could review my webhook code"* |
| Chat from the terminal | `aegis chat` |

## How it works

```
capture ─▶ change detection (perceptual hash) ─▶ one vision call ─▶ redact ─▶ dedupe ─▶ episodes
        ─▶ chunks → SQLite text + FTS5 (BM25) + local vectors ─▶ hourly/daily digests
ask ─▶ time-aware tool-calling agent ─▶ hybrid retrieval (RRF) ─▶ cited, streamed answer
```

The full design, with the math behind each stage, is in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**:
- perceptual-hash change detection;
- Jaccard near-duplicate merging;
- episode segmentation;
- Reciprocal Rank Fusion of BM25 and cosine similarity, with a recency prior;
- hierarchical summaries (frames → episodes → hours → days) and tiered retention;
- bounded tool-loop agent with conversation memory notes and a prompt-injection guard;
- focus and fragmentation metrics;
- the privacy firewall.

## Privacy

- Memory lives in `~/.aegis/aegis_vectors.db`. The server only listens on `127.0.0.1`.
- Never captured: the Aegis HUD itself, password managers, the lock screen, InPrivate/Incognito windows.
- Secrets (API keys, tokens, passwords, private keys) are redacted before storage. `aegis redact-existing` scrubs
  older data.
- **Outbound:**
  - Safe context packs and Darwin queries pass through the privacy firewall.
  - Hosted models only see data if you add a `.env`.
- Retention: screen text 7 days, episode summaries 90 days, daily digests 365 days (configurable).

## Commands

```
aegis run | serve | demo | hud | chat
aegis query "…" [--raw]       aegis timeline --minutes 60
aegis insights [--hours 8]    aegis share --minutes 30 [--focus aegis]
aegis migrate | reembed | redact-existing | init
aegis eval                    # 12-turn conversation on a synthetic day, real model
```

HTTP (loopback):
- `POST /chat` (Server-Sent Events)
- `/chat/sessions`
- `/context/safe`
- `/insights`
- `/memory/timeline`
- `/memory/source/{id}`
- `/context/recent`
- `/context/query`
- `/health`

## Development

```bash
uv run pytest        # 75 tests, no network
uv run aegis eval    # end-to-end against your configured model
```

Source map: `capture` · `filters` · `daemon` · `ingest` · `storage` · `retrieval` · `agent` / `tools` / `llm` ·
`jobs` · `insights` · `safe` · `darwin` · `ipc_server` · `hud`, all under `src/aegis_context/`.

## Roadmap

- **MCP server**, so any agent (Claude, Cursor) can ask Aegis for context through the same privacy firewall.
- **Local OCR**, so screenshots stay on the device even when the chat model is hosted.
- **Pause and forget buttons.**
- **Start at login.**
- **An editable "what Aegis knows about me" fact memory.**
- **Phone access.**
