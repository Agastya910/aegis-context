"""Ask the live Aegis server a fixed set of questions in one chat session; print answers and timings.

    uv run python scripts/probe_live.py [--base http://127.0.0.1:8765]
"""
from __future__ import annotations

import argparse
import sys
import time

import requests

from aegis_context.hud import parse_sse_lines

QUESTIONS = [
    "hi",
    "what was I doing 10 minutes ago?",
    "and before that?",
    "summarize my last hour",
    "what did I look at in chrome?",
    "what was the first thing I did today?",
    "what did I ask claude to do?",
]


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    a = ap.parse_args()
    sid = None
    for q in QUESTIONS:
        t0, first, answer, steps, sources = time.time(), None, "", [], []
        with requests.post(f"{a.base}/chat", json={"message": q, "session_id": sid}, stream=True,
                           timeout=(5, 180)) as r:
            r.raise_for_status()
            r.encoding = "utf-8"
            for ev, data in parse_sse_lines(r.iter_lines(decode_unicode=True)):
                if ev == "session":
                    sid = data["session_id"]
                elif ev == "token":
                    first = first or time.time() - t0
                    answer += data["text"]
                elif ev == "status":
                    steps.append(data["text"])
                elif ev == "sources":
                    sources = [f"{s['id']} {s['start_local']} {s['app']}" for s in data["items"]]
                elif ev == "error":
                    answer += f"\n!! {data['text']}"
        print(f"\n### {q}   ({(first or 0):.1f}s first token, {time.time() - t0:.1f}s total)")
        for s in steps:
            print(f"  · {s}")
        print(answer.strip())
        if sources:
            print("  sources:", "; ".join(sources))


if __name__ == "__main__":
    main()
