# Aegis-Context

Local-first ambient screen memory. A capture daemon (Part 1) records what you're doing,
embeds it, and stores it in SQLite; a loopback FastAPI bridge + HUD (Part 2) lets you recall it
by hotkey, voice shortcut, or HTTP. Nothing leaves the machine.

## Quick start (uv)

```bash
uv sync                        # core deps + dev group
uv run pytest                  # tests
uv run aegis demo              # API with in-memory sample data on 127.0.0.1:8765
curl localhost:8765/context/recent
curl -X POST localhost:8765/context/query -H 'content-type: application/json' \
     -d '{"query_text":"what was I reading about fastapi"}'

uv sync --extra capture        # mss + pillow for screenshots
uv run aegis run               # capture daemon + IPC server (writes ~/.aegis)
uv sync --extra hud            # PyQt5 + pynput
uv run aegis hud               # Ctrl+Shift+Space overlay
uv run aegis query "cardiac model"
```

## Layout

| Module | Role |
|---|---|
| `storage.py` | Tier-2 SQLite store, vectorised cosine search, digest rollups (schema from the spec) |
| `ringbuffer.py` | Tier-1 5-minute in-RAM buffer |
| `backends.py` | Embedders (hash default, ONNX MiniLM optional) and vision (none / Ollama Moondream) |
| `capture.py` | Screenshot + active window (Windows/macOS/X11) |
| `daemon.py` | capture -> vision -> embed -> store loop, skips unchanged frames |
| `ipc_server.py` | `/health`, `/context/recent`, `/context/query` (loopback-only) |
| `hud.py` | Floating overlay + global hotkey |

Config lives in `~/.aegis/config.json` (`aegis init` writes defaults). Set `vision_backend: "ollama"`
and `embedder: "onnx"` for real Moondream2 + MiniLM.

## Fixes versus the spec's code
- `HTTPException(status_code=500, status.HTTP_500..., detail=...)` was a syntax error.
- The server called `similarity_search`, but the engine defines `vector_similarity_search`; result tuple
  indexes (`res[2]`, `res[3]`...) didn't match what the engine returned. Results are now dicts.
- `/context/query` embedded a **random vector**; it now embeds the query text.
- Search looped per row in Python; it's now one matrix multiply.
- The spec's `sqlite-vector` extension load was silently swallowed; it's dropped (brute-force numpy
  is fine up to ~100k rows). 
- HUD read the ring buffer as strings, but entries are structured; it now uses the server's `speakable` text.
- The spec's `Ctrl+Space` vs `Ctrl+Shift+Space` conflict resolved to the latter (the HUD section).

## Not built (out of scope here)
Android AppFunctions / iOS App Intents clients, and the "stream an answer from the local LLM" step.
The HTTP contract they need is in place.
