# Aegis-Context
### The memory that knows you best and never leaves your machine: persistent context for every app and AI you use.

Your computer forgets everything the moment you look away. Aegis doesn't.

Aegis-Context is a persistent, on-device context layer. It quietly watches your workspace, turns every screen state into searchable memory, and brings it back with one keystroke. No window switching, no digging through tabs, no cloud. Ask "what was I looking at five minutes ago?" and just get the answer.

## The problem
Work is a stream of re-orientation. People switch apps ~1,200 times a day, about 4 hours a week spent getting their bearings (HBR, 2022). About 20% of the workweek goes to hunting for information already seen (McKinsey). And every new AI chat starts with zero context.

## Why this changes your day
- Stop re-orienting. Get the thread back instead of rebuilding it.
- Stop searching for what you've already seen. A hunt becomes one query.
- Stop re-explaining yourself to AI. One hotkey, one paste, and your agent starts informed.
- Stay in flow. Recall appears as a floating overlay over whatever you're doing.

Time saved = app switches avoided + searches avoided + re-briefs avoided. Even at a conservative 3 hours a week, that's roughly three work weeks back in year one, and it grows as memory does. (A model with stated assumptions, not a measurement.)

## Why it knows you better than anything else can
Most AI memory is either shallow (a few facts you typed in) or lives in someone else's cloud, which is why you can't hand it everything. Aegis sits on your device, so it can safely capture the whole stream of what you actually do.
- Total recall of your own day: continuous, ambient context, not just what you remembered to save.
- Private enough to capture everything: nothing is uploaded, so there's no reason to hold back.
- Compounding: every hour you work makes the next recall sharper. Older history rolls into digests, so memory grows without bloating.
- Yours and portable: any agent or tool you choose can use it.

## Your context, your machine
Everything stays on your device. The API binds to loopback only (127.0.0.1). Nothing is uploaded, ever.

## How it works
1. Perceive. A capture daemon watches your active window and screen, skips unchanged frames, and optionally runs an on-device vision model (Moondream2 via Ollama) to describe what you're doing.
2. Remember. Every state becomes a vector in tiered local memory: a hot 5-minute buffer in RAM, a vector-searchable SQLite store, and rolling digests that compress older history.
3. Recall. Semantic search across everything you've done. Ask in plain language, get your own history back.
4. Invoke with zero friction. Ctrl+Shift+Space opens a minimal HUD wherever you are. Press Enter and your recent context is on the clipboard, ready for any editor or AI chat.

## Built today
- Full capture, embed and store pipeline with change detection
- Three-tier memory with automatic rollups and retention pruning
- Semantic recall by vector similarity, with optional ONNX MiniLM embeddings
- Local FastAPI bridge: /context/recent, /context/query, /health
- Floating HUD with a global hotkey
- One-command install with uv, and a tested storage and capture core

## Where it's headed
- Insights from your own history: patterns in how you work, what's slipping, what to drop
- Hands-free voice recall ("Hey Google, what was I doing?") via Android AppFunctions and iOS App Intents
- Inline answers from a local LLM
- Safe-context filter: redact and approve before any agent sees anything
- MCP server and connectors for Calendar, Gmail, bCourses and Luma

## The bigger idea
Agents are becoming commodities. The context they run on is the thing that lasts. Aegis is that layer: persistent, portable across every tool, and entirely yours.

## Run it
uv sync --extra capture --extra hud
uv run aegis run      # capture daemon + local API
uv run aegis hud      # Ctrl+Shift+Space
