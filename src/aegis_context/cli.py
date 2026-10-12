"""`aegis` command line: run | serve | demo | chat | query | timeline | hud | init | migrate | reembed | eval."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta

from .backends import VisionResult, make_embedder, make_vision
from .config import Config
from .ipc_server import BridgeState, start_ipc_server
from .ringbuffer import RingBuffer
from .storage import AegisStorageEngine


def _build(cfg: Config):
    storage = AegisStorageEngine(cfg.vectors_db)
    ring = RingBuffer(cfg.ring_window_s)
    return storage, ring, make_embedder(cfg)


def _make_agent(cfg: Config, storage, ring, embedder):
    from .agent import ChatAgent
    from .chat_store import ChatStore
    from .llm import make_chat_client
    from .retrieval import Retrieval

    darwin = None
    if cfg.darwin_enabled:
        from .darwin import DarwinClient

        darwin = DarwinClient(cfg.darwin_url, api_key=cfg.darwin_api_key)
    return ChatAgent(Retrieval(storage, embedder), ring, ChatStore(storage), make_chat_client(cfg), cfg,
                     darwin=darwin)


def _demo_seed(storage, ring, embedder) -> None:
    from .ingest import Ingestor
    from .timeutil import utc_now

    samples = [
        ("code.exe", "Visual Studio Code", "storage.py - aegis", "Editing the SQLite vector search function in storage.py"),
        ("chrome.exe", "Google Chrome", "FastAPI docs - Request Body", "Reading FastAPI documentation about request bodies"),
        ("slack.exe", "Slack", "#ml-team", "Chatting in Slack about cardiac disease prediction model results"),
    ]
    ing = Ingestor(storage, embedder, ring)
    now = utc_now()
    for i, (bid, app, title, summ) in enumerate(samples):
        ing.ingest(bid, app, "demo", title, VisionResult(summ, "", "", summ), now - timedelta(minutes=10 - 4 * i))
    storage.close_open_episodes()


def _chat_repl(agent) -> None:
    print("Aegis chat — ask about anything you've had on screen. Ctrl+C or 'exit' to quit, '/new' for a new chat.")
    sid = None
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q in ("exit", "quit"):
            return
        if q == "/new":
            sid = None
            print("(new chat)")
            continue
        if not q:
            continue
        print("aegis> ", end="", flush=True)
        for kind, data in agent.run(sid, q):
            if kind == "session":
                sid = data["session_id"]
            elif kind == "token":
                print(data["text"], end="", flush=True)
            elif kind == "status":
                print(f"\n  · {data['text']}", end="\naegis> ", flush=True)
            elif kind == "sources":
                print("\n  sources: " + ", ".join(f"{s['id']} {s['start_local']} {s['app']}" for s in data["items"]),
                      end="")
            elif kind == "error":
                print(f"\n  ! {data['text']}", end="")
        print()


def main(argv: list[str] | None = None) -> None:
    # Window titles can hold any Unicode; don't crash on legacy Windows console codepages.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    p = argparse.ArgumentParser(prog="aegis")
    p.add_argument("--home", help="data dir (default ~/.aegis)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="capture daemon + IPC server")
    sub.add_parser("serve", help="IPC server only (no capture)")
    sub.add_parser("demo", help="IPC server with in-memory sample data (touches no files)")
    sub.add_parser("chat", help="chat with your screen memory in the terminal")
    q = sub.add_parser("query", help="one-shot question (or raw search with --raw)")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--raw", action="store_true", help="show retrieval hits only, no LLM")
    t = sub.add_parser("timeline", help="print recent activity episodes")
    t.add_argument("--minutes", type=int, default=60)
    sub.add_parser("hud", help="floating HUD + global hotkey")
    sub.add_parser("init", help="write default config.json")
    sub.add_parser("migrate", help="upgrade the memory db schema and backfill episodes/chunks")
    sub.add_parser("reembed", help="recompute stored vectors with the configured embedder")
    sub.add_parser("redact-existing", help="scrub secrets (API keys, tokens) from stored rows")
    ins = sub.add_parser("insights", help="where your time went: categories, switches, focus blocks, tips")
    ins.add_argument("--hours", type=float, default=0, help="look back N hours (default: since midnight)")
    sh = sub.add_parser("share", help="print a sanitized, paste-ready context pack")
    sh.add_argument("--minutes", type=int, default=30)
    sh.add_argument("--focus", default=None)
    sub.add_parser("eval", help="run the scripted chat evaluation against a synthetic day (uses the real LLM)")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if a.cmd in ("run", "serve", "demo") else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s")
    cfg = Config.load(a.home)

    if a.cmd == "init":
        cfg.save()
        print(f"wrote {cfg.home}/config.json")
        return
    if a.cmd == "hud":
        from .hud import run_hud

        run_hud(base=f"http://{cfg.host}:{cfg.port}", hide_from_capture=cfg.hud_hidden_from_capture)
        return
    if a.cmd == "eval":
        from .evals import run_eval

        sys.exit(0 if run_eval(cfg) else 1)

    if a.cmd == "demo":
        cfg.embedder = "hash"
        storage, ring, embedder = AegisStorageEngine(":memory:"), RingBuffer(cfg.ring_window_s), make_embedder(cfg)
    else:
        storage, ring, embedder = _build(cfg)

    if a.cmd == "migrate" or (a.cmd == "run" and storage.needs_backfill):
        from .ingest import Ingestor

        print("migrating memory to schema v1 (backup written next to the db)…")
        print(f"backfilled {Ingestor(storage, embedder, None, cfg).backfill()} frames")
        if a.cmd == "migrate":
            return
    if a.cmd == "redact-existing":
        print(f"redacted {storage.redact_existing(embedder)} rows")
        return
    if a.cmd == "reembed":
        print(f"re-embedded {storage.reembed(embedder)} rows with {cfg.embedder} (dim {embedder.dim})")
        return
    if a.cmd == "timeline":
        from .retrieval import Retrieval
        from .timeutil import fmt_local, utc_now

        rows, total = Retrieval(storage, embedder).timeline(utc_now() - timedelta(minutes=a.minutes), utc_now(),
                                                            limit=200)
        for h in rows:
            print(f"[{h.source_id}] {fmt_local(h.start)}-{fmt_local(h.end)} {h.app} | {h.title}\n    {h.activity}"
                  + (f"\n    typed: {h.typed}" if h.typed else ""))
        print(f"({total} episodes)")
        return

    if a.cmd in ("insights", "share"):
        from .insights import format_report, time_report
        from .retrieval import Retrieval
        from .safe import build_pack
        from .timeutil import parse_when, utc_now

        r = Retrieval(storage, embedder)
        if a.cmd == "share":
            print(build_pack(r, a.minutes, a.focus, private_terms=cfg.private_terms).text)
        else:
            start = utc_now() - timedelta(hours=a.hours) if a.hours else parse_when("today")
            print(format_report(time_report(r, start, utc_now())))
        return

    agent = _make_agent(cfg, storage, ring, embedder)
    if a.cmd == "chat":
        _chat_repl(agent)
        return
    if a.cmd == "query":
        if a.raw or agent.llm is None:
            for h in agent.retrieval.search(a.text, k=a.k):
                d = h.as_dict()
                print(f"{d['score']:.4f}  [{d['id']}] {d['date_local']}  {d['app']}: {d['summary']}")
            return
        out = agent.ask(a.text)
        agent.store.delete_session(out["session_id"])
        print(out["answer"])
        if out["sources"]:
            print("\nsources: " + ", ".join(f"{s['id']} {s['start_local']} {s['app']}" for s in out["sources"]))
        return

    state = BridgeState(storage, ring, embedder, agent=agent)
    state.extras["darwin_api_key"] = cfg.darwin_api_key
    if a.cmd == "demo":
        _demo_seed(storage, ring, embedder)
    if a.cmd == "run":
        from .capture import active_window, grab_frame, idle_seconds
        from .filters import hud_visible
        from .daemon import CaptureDaemon

        from .jobs import MemoryJobs

        jobs = MemoryJobs(storage, embedder, agent.llm, cfg)
        daemon = CaptureDaemon(
            storage, ring, embedder, make_vision(cfg), grab_frame, active_window,
            cfg.capture_interval_s, cfg.digest_after_s, cfg=cfg, idle=idle_seconds, jobs=jobs,
            hud_visible=hud_visible,
        )
        daemon.start()
        jobs.start()
        state.extras["daemon"] = daemon
    llm = agent.llm.describe() if agent.llm else "none (raw retrieval answers)"
    print(f"Aegis on http://{cfg.host}:{cfg.port}  (open it in a browser for the web app)  ·  chat model: {llm}")
    start_ipc_server(state, cfg.host, cfg.port)


if __name__ == "__main__":
    main()
