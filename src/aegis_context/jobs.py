"""Background memory jobs: episode summaries, hourly/daily digests (Tier-3), tiered retention."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

from .backends import embed_many
from .prompts import DIGEST, EPISODE_SUMMARY
from .redact import redact
from .timeutil import db_fmt, fmt_local, parse_db, to_local, utc_now

log = logging.getLogger("aegis.jobs")


class MemoryJobs:
    def __init__(self, storage, embedder, llm=None, cfg=None, clock=utc_now, tz=None):
        self.st, self.embedder, self.llm, self.clock, self.tz = storage, embedder, llm, clock, tz
        self.raw_days = getattr(cfg, "raw_retention_days", 7)
        self.episode_days = getattr(cfg, "episode_retention_days", 90)
        self.digest_days = getattr(cfg, "retention_days", 365)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _q(self, sql, args=()):
        with self.st._lock:
            return self.st.conn.execute(sql, args).fetchall()

    def _x(self, sql, args=()):
        with self.st._lock:
            cur = self.st.conn.execute(sql, args)
            self.st.conn.commit()
            return cur

    def _llm(self, prompt: str) -> str:
        text, _ = self.llm.complete([{"role": "user", "content": prompt}])
        return text.strip()

    # ---- episode summaries ------------------------------------------------
    def summarize_episodes(self, limit: int = 5) -> int:
        cutoff = db_fmt(self.clock() - timedelta(seconds=60))
        rows = self._q("""SELECT e.episode_id, e.app_id, a.display_name, e.norm_title, e.start_at, e.end_at,
                                 e.summary_attempts
                          FROM episodes e JOIN applications a USING (app_id)
                          WHERE e.summary_status = 'pending' AND e.end_at <= ? AND e.frame_count > 0
                          ORDER BY e.start_at LIMIT ?""", (cutoff, limit))
        done = 0
        for eid, app_id, app, title, s, e, attempts in rows:
            frames = self._q("SELECT semantic_summary, typed_text, raw_ocr_text FROM screen_contexts "
                             "WHERE episode_id = ? ORDER BY context_id", (eid,))
            acts = list(dict.fromkeys(f[0] for f in frames if f[0]))
            typed = list(dict.fromkeys(f[1] for f in frames if f[1]))
            texts = [f[2] for f in frames if f[2]]
            fallback = " / ".join(acts[:3]) or f"{app}: {title}"
            summary = ""
            if self.llm is not None and attempts < 3:
                excerpt = ""
                if texts:
                    longest = max(texts, key=len)
                    excerpt = texts[-1][:1500] + ("\n…\n" + longest[:1500] if longest != texts[-1] else "")
                ps, pe = parse_db(s), parse_db(e)
                mins = max(1, round((pe - ps).total_seconds() / 60))
                prompt = EPISODE_SUMMARY.format(
                    app=app, title=title, start=fmt_local(ps, date=True, tz=self.tz), end=fmt_local(pe, tz=self.tz),
                    duration=f"{mins} min", activities="\n".join(f"- {a}" for a in acts[:12]) or "-",
                    typed="\n".join(f'- "{t}"' for t in typed[-5:]) or "-", ocr_excerpt=excerpt or "-")
                try:
                    summary = redact(self._llm(redact(prompt)[:6000]))
                except Exception as exc:
                    log.warning("episode %s summary failed: %s", eid, exc)
                    self._x("UPDATE episodes SET summary_attempts = summary_attempts + 1 WHERE episode_id = ?",
                            (eid,))
                    continue
            summary = summary or fallback
            self._x("UPDATE episodes SET summary = ?, summary_status = 'done' WHERE episode_id = ?", (summary, eid))
            self.st.delete_chunks("episode", episode_id=eid)
            text = f"{app} | {title}\n{summary}"
            self.st.add_chunks("episode", [text], embed_many(self.embedder, [text]), s, app_id, episode_id=eid)
            done += 1
        return done

    # ---- digests ----------------------------------------------------------
    def _insert_digest(self, start: datetime, end: datetime, text: str, level: str) -> int:
        [vec] = embed_many(self.embedder, [text])
        cur = self._x("INSERT INTO historical_digests (start_time, end_time, macro_summary, digest_embedding, level) "
                      "VALUES (?, ?, ?, ?, ?)", (db_fmt(start), db_fmt(end), text, vec.astype("float32").tobytes(),
                                                 level))
        did = cur.lastrowid
        self.st.add_chunks("digest", [text], [vec], start, None, digest_id=did)
        return did

    def _local_floor(self, dt: datetime, unit: str) -> datetime:
        loc = to_local(dt, self.tz)
        loc = loc.replace(minute=0, second=0, microsecond=0)
        if unit == "day":
            loc = loc.replace(hour=0)
        return loc.astimezone(timezone.utc)

    def hourly_digests(self, limit: int = 3) -> int:
        now = self.clock()
        since = db_fmt(now - timedelta(days=7))
        eps = self._q("SELECT start_at FROM episodes WHERE summary_status = 'done' AND start_at >= ?", (since,))
        hours = sorted({self._local_floor(parse_db(r[0]), "hour") for r in eps})
        have = {r[0] for r in self._q("SELECT start_time FROM historical_digests WHERE level = 'hour'")}
        made = 0
        for h in hours:
            end = h + timedelta(hours=1)
            if end > now or db_fmt(h) in have:
                continue
            busy = self._q("SELECT 1 FROM episodes WHERE summary_status IN ('open', 'pending') "
                           "AND start_at < ? AND end_at >= ? LIMIT 1", (db_fmt(end), db_fmt(h)))
            if busy:
                continue
            rows = self._q("""SELECT e.start_at, e.end_at, a.display_name, e.norm_title, e.summary
                              FROM episodes e JOIN applications a USING (app_id)
                              WHERE e.summary_status = 'done' AND e.start_at >= ? AND e.start_at < ?
                              ORDER BY e.start_at""", (db_fmt(h), db_fmt(end)))
            lines = "\n".join(f"- {fmt_local(r[0], tz=self.tz)}-{fmt_local(r[1], tz=self.tz)} {r[2]} | {r[3]}: {r[4]}"
                              for r in rows)
            text = self._digest_text(h, end, lines)
            if text:
                self._insert_digest(h, end, text, "hour")
                made += 1
            if made >= limit:
                break
        return made

    def daily_digests(self) -> int:
        now = self.clock()
        hours = self._q("SELECT start_time, macro_summary FROM historical_digests WHERE level = 'hour' "
                        "ORDER BY start_time")
        days: dict[datetime, list[str]] = {}
        for st, txt in hours:
            days.setdefault(self._local_floor(parse_db(st), "day"), []).append(
                f"- {fmt_local(st, tz=self.tz)}: {txt}")
        have = {r[0] for r in self._q("SELECT start_time FROM historical_digests WHERE level = 'day'")}
        made = 0
        for d, lines in sorted(days.items()):
            end = d + timedelta(days=1)
            if end > now or db_fmt(d) in have:
                continue
            text = self._digest_text(d, end, "\n".join(lines))
            if text:
                self._insert_digest(d, end, text, "day")
                made += 1
        return made

    def _digest_text(self, start: datetime, end: datetime, lines: str) -> str:
        if not lines.strip():
            return ""
        if self.llm is None:
            return lines[:1500]
        try:
            return redact(self._llm(DIGEST.format(start=fmt_local(start, date=True, tz=self.tz),
                                                  end=fmt_local(end, date=True, tz=self.tz), episode_lines=lines)))
        except Exception as exc:
            log.warning("digest failed: %s", exc)
            return ""

    # ---- retention ----------------------------------------------------------
    def retention(self) -> dict:
        now = self.clock()
        raw_cut = db_fmt(now - timedelta(days=self.raw_days))
        ep_cut = db_fmt(now - timedelta(days=self.episode_days))
        dg_cut = db_fmt(now - timedelta(days=self.digest_days))
        with self.st._lock:
            c = self.st.conn
            n_chunks = c.execute("""DELETE FROM chunks WHERE kind IN ('meta', 'ocr') AND context_id IN
                                    (SELECT context_id FROM screen_contexts WHERE captured_at < ?)""",
                                 (raw_cut,)).rowcount
            c.execute("UPDATE screen_contexts SET raw_ocr_text = '', embedding_blob = NULL "
                      "WHERE captured_at < ? AND raw_ocr_text != ''", (raw_cut,))
            old_eps = [r[0] for r in c.execute("SELECT episode_id FROM episodes WHERE end_at < ?", (ep_cut,))]
            if old_eps:
                marks = ",".join("?" * len(old_eps))
                c.execute(f"DELETE FROM screen_contexts WHERE episode_id IN ({marks})", old_eps)
                c.execute(f"DELETE FROM episodes WHERE episode_id IN ({marks})", old_eps)
            n_dg = c.execute("DELETE FROM historical_digests WHERE end_time < ?", (dg_cut,)).rowcount
            c.commit()
        return {"chunks": n_chunks, "episodes": len(old_eps), "digests": n_dg}

    # ---- loop ---------------------------------------------------------------
    def run_once(self) -> dict:
        out = {}
        for name, fn in (("episodes", self.summarize_episodes), ("hourly", self.hourly_digests),
                         ("daily", self.daily_digests), ("retention", self.retention)):
            try:
                out[name] = fn()
            except Exception as exc:
                log.exception("job %s failed", name)
                out[name] = f"error: {exc}"
        return out

    def start(self, interval_s: float = 60) -> None:
        def loop():
            while not self._stop.wait(interval_s):
                self.run_once()
        self._thread = threading.Thread(target=loop, name="aegis-jobs", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
