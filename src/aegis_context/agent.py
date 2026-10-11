"""The chat agent: time-aware, tool-calling RAG over screen memory, with persistent conversation memory."""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Iterator

from .chat_store import ChatStore
from .llm import LLMError
from .prompts import SESSION_SUMMARY, SYSTEM
from .retrieval import Retrieval
from .timeutil import find_range_in_text, fmt_local, to_local, tz_label, utc_now
from .tools import TOOLS, ToolContext, dispatch, format_search_row, format_timeline_row

log = logging.getLogger("aegis.agent")
_CITE_GROUP = re.compile(r"\[([^\]]{1,80})\]")
_CITE_ID = re.compile(r"\b([EFD]\d+)\b")
_MEMO_ECHO = re.compile(r"\s*\(context used:[^)]*\)", re.I)

Event = tuple[str, dict]


def cited_ids(text: str) -> list[str]:
    out: list[str] = []
    for group in _CITE_GROUP.findall(text or ""):
        for sid in _CITE_ID.findall(group):
            if sid not in out:
                out.append(sid)
    return out


class ChatAgent:
    def __init__(self, retrieval: Retrieval, ring, chat_store: ChatStore, llm, cfg=None, clock=utc_now, tz=None):
        self.retrieval, self.ring, self.store, self.llm = retrieval, ring, chat_store, llm
        self.clock, self.tz = clock, tz
        self.history_turns = getattr(cfg, "history_turns", 10)
        self.history_char_budget = getattr(cfg, "history_char_budget", 12000)
        self.max_rounds = getattr(cfg, "agent_max_rounds", 4)
        self.tool_result_chars = getattr(cfg, "tool_result_chars", 6000)

    # ---- prompt -----------------------------------------------------------
    def system_prompt(self, session_id: str) -> str:
        now = self.clock()
        nl = to_local(now, self.tz)
        first, _ = self.retrieval.coverage()
        live = "nothing captured in the last few minutes"
        if self.ring is not None:
            recent = self.ring.recent(n=1)
            if recent:
                i = recent[-1]
                live = f"{i['app']} | {i['window_title']}: {i['summary']}"
        summary = self.store.get_summary(session_id)
        return SYSTEM.format(
            now_local=nl.strftime("%Y-%m-%d %H:%M"), weekday=nl.strftime("%A"), tz_label=tz_label(self.tz),
            coverage_start_local=fmt_local(first, date=True, tz=self.tz) if first else "(no memory yet)",
            known_apps=", ".join(self.retrieval.known_apps()) or "none yet", live_line=live,
            session_summary_block=f"Earlier in this conversation: {summary}\n" if summary else "",
        )

    def _history(self, session_id: str) -> list[dict]:
        msgs = self.store.get_messages(session_id)[:-1]  # the current user message is appended separately
        out = []
        for m in msgs[-self.history_turns:]:
            out.append({"role": m["role"], "content": m["content"]})
            if m["role"] == "assistant" and m["memo"]:  # separate note so the model doesn't imitate it in answers
                out.append({"role": "system", "content": f"Memory note for the answer above (internal, never repeat "
                                                         f"this format): context used: {m['memo']}"})
        return out

    # ---- run ----------------------------------------------------------------
    def run(self, session_id: str | None, user_text: str) -> Iterator[Event]:
        sid = self.store.ensure(session_id)
        yield "session", {"session_id": sid}
        self.store.add_message(sid, "user", user_text)
        ctx = ToolContext(self.retrieval, self.clock(), self.tz, self.tool_result_chars)
        if self.llm is None:
            yield from self._finish(sid, ctx, self._offline_answer(user_text, ctx, "No LLM configured"))
            return
        messages = [{"role": "system", "content": self.system_prompt(sid)}, *self._history(sid),
                    {"role": "user", "content": user_text}]
        full = ""
        try:
            for rnd in range(self.max_rounds + 1):
                final_round = rnd == self.max_rounds
                if final_round:
                    messages.append({"role": "system", "content": "Answer now using the data above."})
                text, calls = "", []
                for ev in self.llm.stream(messages, None if final_round else TOOLS):
                    if ev.type == "token":
                        if not text and full and not full.endswith(("\n", " ")):
                            full += "\n\n"
                            yield "token", {"text": "\n\n"}
                        text += ev.text
                        full += ev.text
                        yield "token", {"text": ev.text}
                    elif ev.type == "tool_calls":
                        calls = ev.tool_calls
                    elif ev.type == "status":
                        yield "status", {"text": ev.text}
                if not calls:
                    break
                messages.append({"role": "assistant", "content": text, "tool_calls": calls})
                for call in calls:
                    fn = call.get("function") or {}
                    name, args = fn.get("name", ""), fn.get("arguments") or {}
                    result, status = dispatch(name, args, ctx)
                    yield "status", {"text": status}
                    messages.append({"role": "tool", "tool_name": name, "content": result})
        except LLMError as exc:
            if not full:
                yield "status", {"text": "LLM unavailable, showing raw memory results"}
                yield from self._finish(sid, ctx, self._offline_answer(user_text, ctx, str(exc)))
                return
            yield "error", {"text": f"LLM stopped mid-answer: {exc}"}
        yield from self._finish(sid, ctx, full, streamed=True)

    def _finish(self, sid: str, ctx: ToolContext, text: str, streamed: bool = False) -> Iterator[Event]:
        if not streamed and text:
            yield "token", {"text": text}
        items = []
        for cid in cited_ids(text):
            h = ctx.seen.get(cid) or self.retrieval._hit_for(cid, 0.0, "", [])
            if h:
                ctx.seen[cid] = h
                items.append(h.as_dict(self.tz))
        if items:
            yield "sources", {"items": items}
        memo_parts = list(dict.fromkeys(ctx.consulted))
        if items:
            memo_parts.append("cited " + ", ".join(f"{i['id']} ({i['start_local']}"
                                                   + (f"-{i['end_local']}" if i['end_local'] != i['start_local']
                                                      else "") + ")" for i in items))
        mid = self.store.add_message(sid, "assistant", _MEMO_ECHO.sub("", text).strip(), "; ".join(memo_parts))
        yield "done", {"message_id": mid, "session_id": sid}
        self._maybe_compact(sid)

    # ---- fallback / maintenance ---------------------------------------------------
    def _offline_answer(self, user_text: str, ctx: ToolContext, why: str) -> str:
        rng = find_range_in_text(user_text, ctx.now, ctx.tz)
        if rng:
            rows, total = self.retrieval.timeline(rng[0], rng[1], limit=20)
            body = "\n".join(format_timeline_row(h, ctx) for h in rows) or "No captured activity in that period."
        else:
            rows = self.retrieval.search(user_text, k=5)
            body = "\n".join(format_search_row(h, ctx) for h in rows) or "Nothing matching in memory."
        for h in rows:
            ctx.seen[h.source_id] = h
        cites = " ".join(f"[{h.source_id}]" for h in rows)
        log.warning("offline answer (%s)", why)
        return f"LLM offline — raw memory results:\n\n{body}" + (f"\n\n{cites}" if cites else "")

    def _maybe_compact(self, sid: str) -> None:
        msgs = self.store.get_messages(sid)
        if sum(len(m["content"]) for m in msgs) <= self.history_char_budget or self.llm is None:
            return
        older = msgs[:-self.history_turns]
        if not older:
            return

        def work():
            try:
                convo = "\n".join(f"{m['role']}: {m['content'][:800]}" for m in older)
                prev = self.store.get_summary(sid)
                text, _ = self.llm.complete([{"role": "user", "content": SESSION_SUMMARY.format(
                    messages=(f"(previous summary: {prev})\n" if prev else "") + convo)}])
                if text.strip():
                    self.store.set_summary(sid, text.strip())
            except Exception as exc:
                log.warning("session summary failed: %s", exc)

        threading.Thread(target=work, daemon=True).start()

    def ask(self, text: str, session_id: str | None = None) -> dict:
        """Non-streaming convenience: run a turn and collect the result."""
        out = {"answer": "", "sources": [], "status": [], "session_id": None, "error": None}
        for kind, data in self.run(session_id, text):
            if kind == "token":
                out["answer"] += data["text"]
            elif kind == "sources":
                out["sources"] = data["items"]
            elif kind == "status":
                out["status"].append(data["text"])
            elif kind == "session":
                out["session_id"] = data["session_id"]
            elif kind == "error":
                out["error"] = data["text"]
        return out


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
