"""`aegis eval`: a scripted multi-turn conversation against the synthetic day, using the real configured LLM."""
from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

from .agent import ChatAgent
from .backends import make_embedder
from .chat_store import ChatStore
from .jobs import MemoryJobs
from .llm import make_chat_client
from .retrieval import Retrieval
from .storage import AegisStorageEngine
from .synthetic_day import NOW, TZ, build


def _has(*words):
    return lambda a, st: all(w.lower() in a.lower() for w in words)


def _any(*words):
    return lambda a, st: any(w.lower() in a.lower() for w in words)


def _ordered(first: tuple, second: tuple):
    def check(a, st):
        low = a.lower()
        i = min((low.find(w) for w in first if w in low), default=-1)
        j = min((low.find(w) for w in second if w in low), default=-1)
        return i >= 0 and j >= 0 and i < j
    return check


CONVERSATION = [
    ("hi", lambda a, st: not st and len(a) > 0, "greets, no tool calls"),
    ("what was I doing at 10:10?", _any("stripe", "webhook"), "stripe / webhook"),
    ("and before that?", _any("locked", "storage.py", "sqlite"), "database is locked / storage.py"),
    ("summarize my afternoon", _ordered(("android", "mobile"), ("attention",)),
     "mobile app … attention, in time order"),
    ("what was the first thing I did today?", _any("priya", "budget"), "priya / budget"),
    ("what did I ask claude to do?", _has("create the separate mobile app"), "quotes the typed prompt"),
    ("what did I watch on youtube?", _has("attention is all you need"), "attention is all you need"),
    ("when is the budget review?", lambda a, st: "thursday" in a.lower() and re.search(r"\b2\s*(pm|:00)|14:00", a.lower())
     is not None, "thursday 2pm"),
    ("what did I do yesterday?", lambda a, st: re.search(
        r"\bno\b|nothing|don't have|do not have|isn't any|not (captured|available|recorded)|only (have|covers)|"
        r"starts|begins", a.lower()) is not None, "says there is no data for yesterday"),
    ("how can i optimize my time spent on tasks?", lambda a, st: any("Measuring your time" in s for s in st)
     and re.search(r"\d+\s*(min|%|h\b|switch)", a.lower()) is not None, "measured time report + concrete advice"),
    ("get only the safe info to share from this", lambda a, st: any("safe context pack" in s for s in st)
     and "priya" not in a.lower() and "auc 0.91" not in a.lower(), "sanitized pack, no private names/content"),
    ("find an agent that could help me plan my week", lambda a, st: any("Darwin" in s for s in st),
     "Darwin search with a generic query"),
]


def run_eval(cfg, summarize: bool = True, out=print) -> bool:
    llm = make_chat_client(cfg)
    if llm is None:
        out("No chat LLM configured (chat_backend / .env); nothing to evaluate.")
        return False
    with tempfile.TemporaryDirectory() as d:
        db = AegisStorageEngine(Path(d) / "eval.db")
        embedder = make_embedder(cfg)
        build(db, embedder)
        if summarize:
            out("summarizing episodes with the real LLM…")
            jobs = MemoryJobs(db, embedder, llm, cfg, clock=lambda: NOW, tz=TZ)
            while jobs.summarize_episodes(limit=20):
                pass
        darwin = None
        if getattr(cfg, "darwin_enabled", False):
            from .darwin import DarwinClient

            darwin = DarwinClient(cfg.darwin_url)
        agent = ChatAgent(Retrieval(db, embedder, clock=lambda: NOW, tz=TZ), None, ChatStore(db), llm, cfg,
                          clock=lambda: NOW, tz=TZ, darwin=darwin)
        sid, passed = None, 0
        out(f"model: {llm.describe()}  ·  frozen now: Sun 2026-10-11 16:00 (UTC-07:00)\n")
        for i, (q, check, expect) in enumerate(CONVERSATION, 1):
            t0, first, answer, steps, err = time.time(), None, "", [], None
            for kind, data in agent.run(sid, q):
                if kind == "session":
                    sid = data["session_id"]
                elif kind == "token":
                    first = first or time.time() - t0
                    answer += data["text"]
                elif kind == "status":
                    steps.append(data["text"])
                elif kind == "error":
                    err = data["text"]
            total = time.time() - t0
            ok = err is None and check(answer, steps)
            passed += ok
            out(f"{'PASS' if ok else 'FAIL'} {i}. {q}  [{(first or 0):.1f}s first token, {total:.1f}s total]")
            out(f"     expect: {expect}")
            for s in steps:
                out(f"     · {s}")
            out("     " + " ".join(answer.split())[:400] + ("" if not err else f"  !! {err}"))
        db.close()
    out(f"\n{passed}/{len(CONVERSATION)} passed")
    return passed == len(CONVERSATION)
