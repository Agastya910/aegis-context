"""Part 2 ingress: FastAPI loopback bridge for hotkeys, HUD and mobile assistants."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import sse
from .backends import Embedder, embed_query
from .ringbuffer import RingBuffer
from .storage import AegisStorageEngine
from .timeutil import fmt_local, utc_now


@dataclass
class BridgeState:
    storage: AegisStorageEngine
    ring: RingBuffer
    embedder: Embedder
    extras: dict = field(default_factory=dict)
    agent: object = None  # ChatAgent; None = retrieval-only (legacy) mode


class QueryRequest(BaseModel):
    query_text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


def _speakable(items: list[dict]) -> str:
    if not items:
        return "No recent activity recorded."
    return ". ".join(
        f"{i['app']}{': ' + i['window_title'] if i['window_title'] else ''} - {i['summary']}" for i in items
    )


def create_app(state: BridgeState) -> FastAPI:
    app = FastAPI(title="Aegis Ambient IPC Bridge", version="2.0.0")

    def need_agent():
        if state.agent is None:
            raise HTTPException(status_code=503, detail="Chat agent not configured.")
        return state.agent

    @app.get("/health")
    def health():
        out = {"status": "ok", "ring": len(state.ring), "stored": state.storage.count(),
               "frames": state.storage.count()}
        with state.storage._lock:
            out["episodes"] = state.storage.conn.execute("SELECT count(*) FROM episodes").fetchone()[0]
        daemon = state.extras.get("daemon")
        if daemon is not None:
            out["vision_errors"] = daemon.vision_errors
            out["last_capture_local"] = fmt_local(daemon.last_capture, date=True) if daemon.last_capture else None
        llm = getattr(state.agent, "llm", None)
        out["llm"] = llm.describe() if hasattr(llm, "describe") else None
        return out

    @app.get("/context/recent")
    def recent(n: int = 5, within_s: float | None = None):
        items = [dict(i) for i in state.ring.recent(n=n, within_s=within_s)]
        for i in items:
            i["local_time"] = fmt_local(datetime.fromtimestamp(i["ts"], timezone.utc))
        return {"status": "ok", "count": len(items), "recent_context": items, "speakable": _speakable(items)}

    @app.post("/context/query")
    def query(payload: QueryRequest):
        if state.storage is None:
            raise HTTPException(status_code=500, detail="Storage engine not bound.")
        if state.agent is not None:
            out = state.agent.ask(payload.query_text)
            state.agent.store.delete_session(out["session_id"])  # one-shot query: don't keep a chat session
            matches = out["sources"] or [h.as_dict() for h in state.agent.retrieval.search(
                payload.query_text, k=payload.top_k)]
            return {"status": "ok", "answer": out["answer"], "answer_error": out["error"], "matches": matches,
                    "digests": [m for m in matches if m["id"].startswith("D")], "synthesized": out["answer"]}
        qv = embed_query(state.embedder, payload.query_text)
        matches = state.storage.hybrid_search(payload.query_text, qv, top_k=payload.top_k)
        digests = state.storage.search_digests(qv, top_k=1)
        synthesized = " ".join(m["summary"] for m in matches[:payload.top_k]) or (
            digests[0]["summary"] if digests else "No matching context found."
        )
        return {"status": "ok", "answer": None, "answer_error": None, "matches": matches, "digests": digests,
                "synthesized": synthesized}

    @app.post("/chat")
    def chat(payload: ChatRequest):
        agent = need_agent()

        def gen():
            try:
                for kind, data in agent.run(payload.session_id, payload.message):
                    yield sse(kind, data)
            except Exception as exc:  # never leave the client hanging without a terminal event
                yield sse("error", {"text": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/chat/sessions")
    def sessions():
        return need_agent().store.list_sessions()

    @app.post("/chat/sessions")
    def new_session():
        return {"session_id": need_agent().store.create_session()}

    @app.get("/chat/sessions/{session_id}")
    def session(session_id: str):
        store = need_agent().store
        if not store.exists(session_id):
            raise HTTPException(status_code=404, detail="No such session.")
        return {"session_id": session_id, "messages": store.get_messages(session_id)}

    @app.delete("/chat/sessions/{session_id}")
    def delete_session(session_id: str):
        need_agent().store.delete_session(session_id)
        return {"status": "ok"}

    @app.get("/memory/source/{source_id}")
    def source(source_id: str):
        return {"id": source_id.upper(), "text": need_agent().retrieval.source(source_id)}

    @app.get("/memory/timeline")
    def timeline(minutes: int = 60):
        r = need_agent().retrieval
        now = utc_now()
        rows, total = r.timeline(now - timedelta(minutes=max(1, min(minutes, 7 * 1440))), now, limit=200)
        return {"total": total, "items": [h.as_dict() for h in rows]}

    return app


def start_ipc_server(state: BridgeState, host: str = "127.0.0.1", port: int = 8765) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Aegis IPC must bind to loopback only")
    uvicorn.run(create_app(state), host=host, port=port, log_level="warning")
