"""Retrieval over the memory store: hybrid (vector + FTS5, RRF-fused) search, timelines, sources, day summaries.

Source ids: F<n> = frame (screen_contexts), E<n> = episode, D<n> = digest.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .backends import embed_query
from .storage import _snippet, _terms
from .timeutil import db_fmt, fmt_local, parse_db, utc_now

APP_ALIASES = {
    "vscode": ["code"], "vs code": ["code"], "visual studio code": ["code"],
    "terminal": ["windowsterminal"], "windows terminal": ["windowsterminal"], "powershell": ["windowsterminal"],
    "claude code": ["windowsterminal", "claude"], "claude": ["claude", "windowsterminal"],
    "edge": ["msedge"], "word": ["winword"], "outlook": ["outlook", "olk"], "teams": ["ms-teams"],
    "browser": ["chrome", "msedge", "firefox", "brave"], "google chrome": ["chrome"],
    "file explorer": ["explorer"],
}


@dataclass
class Hit:
    source_id: str
    start: datetime
    end: datetime
    app: str
    title: str
    activity: str
    typed: str = ""
    snippet: str = ""
    score: float = 0.0
    episode_id: int | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self, tz=None) -> dict:
        return {"id": self.source_id, "start_local": fmt_local(self.start, tz=tz),
                "end_local": fmt_local(self.end, tz=tz), "date_local": fmt_local(self.start, date=True, tz=tz),
                "timestamp": db_fmt(self.start), "app": self.app, "title": self.title, "summary": self.activity,
                "typed": self.typed, "snippet": self.snippet, "score": round(self.score, 5)}


def fts_query(text: str) -> str | None:
    terms = _terms(text)
    return " OR ".join(f'"{t}"' for t in terms) or None


def rrf(*rankings: list[int], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ids in rankings:
        for rank, i in enumerate(ids):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
    return scores


class VectorCache:
    """In-RAM matrix of chunk vectors, refreshed incrementally (works across processes)."""

    def __init__(self, storage):
        self.storage = storage
        self.reset()

    def reset(self) -> None:
        self.ids = np.zeros(0, dtype=np.int64)
        self.mat: np.ndarray | None = None
        self.times = np.zeros(0, dtype=np.int64)
        self.app_ids = np.zeros(0, dtype=np.int64)
        self.max_id = 0
        self.dim = 0

    def refresh(self, dim: int) -> None:
        st = self.storage
        with st._lock:
            n = st.conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        if dim != self.dim or n < len(self.ids):
            self.reset()
            self.dim = dim
        with st._lock:
            rows = st.conn.execute(
                "SELECT chunk_id, captured_at, COALESCE(app_id, -1), embedding FROM chunks "
                "WHERE chunk_id > ? AND embedding IS NOT NULL ORDER BY chunk_id", (self.max_id,)).fetchall()
        if not rows:
            return
        self.max_id = rows[-1][0]
        rows = [r for r in rows if len(r[3]) == dim * 4]
        if not rows:
            return
        m = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)
        self.ids = np.concatenate([self.ids, np.array([r[0] for r in rows], dtype=np.int64)])
        self.times = np.concatenate([self.times, np.array([int(parse_db(r[1]).timestamp()) for r in rows])])
        self.app_ids = np.concatenate([self.app_ids, np.array([r[2] for r in rows], dtype=np.int64)])
        self.mat = m if self.mat is None else np.vstack([self.mat, m])

    def top(self, q: np.ndarray, n: int, start=None, end=None, app_ids=None) -> list[int]:
        if self.mat is None or not len(self.ids):
            return []
        mask = np.ones(len(self.ids), dtype=bool)
        if start is not None:
            mask &= self.times >= int(start.timestamp())
        if end is not None:
            mask &= self.times <= int(end.timestamp())
        if app_ids:
            mask &= np.isin(self.app_ids, list(app_ids))
        idx = np.nonzero(mask)[0]
        if not len(idx):
            return []
        qn = q / (np.linalg.norm(q) + 1e-10)
        sims = self.mat[idx] @ qn
        order = np.argsort(-sims)[:n]
        return [int(self.ids[idx[i]]) for i in order]


class Retrieval:
    def __init__(self, storage, embedder, clock=utc_now, tz=None):
        self.storage, self.embedder, self.clock, self.tz = storage, embedder, clock, tz
        self.cache = VectorCache(storage)

    def _q(self, sql: str, args=()) -> list:
        with self.storage._lock:
            return self.storage.conn.execute(sql, args).fetchall()

    # ---- apps -------------------------------------------------------------
    def known_apps(self, days: int = 7) -> list[str]:
        since = db_fmt(self.clock() - timedelta(days=days))
        return [r[0] for r in self._q(
            """SELECT a.display_name FROM episodes e JOIN applications a USING (app_id)
               WHERE e.end_at >= ? GROUP BY a.app_id ORDER BY sum(e.frame_count) DESC LIMIT 25""", (since,))]

    def resolve_apps(self, names) -> set[int]:
        if not names:
            return set()
        if isinstance(names, str):
            names = [names]
        apps = self._q("SELECT app_id, lower(display_name), lower(bundle_identifier) FROM applications")
        out: set[int] = set()
        for name in names:
            n = name.lower().strip().removesuffix(".exe")
            wanted = APP_ALIASES.get(n, []) + [n]
            for aid, disp, bundle in apps:
                stem = bundle.removesuffix(".exe")
                if any(w in (disp, stem) for w in wanted) or any(len(w) >= 3 and (w in disp or w in stem)
                                                                 for w in wanted):
                    out.add(aid)
        return out

    def coverage(self) -> tuple[datetime | None, datetime | None]:
        r = self._q("SELECT min(captured_at), max(captured_at) FROM screen_contexts")[0]
        return (parse_db(r[0]) if r[0] else None, parse_db(r[1]) if r[1] else None)

    # ---- episodes ---------------------------------------------------------
    def _live_description(self, episode_id: int) -> tuple[str, str]:
        rows = self._q("SELECT semantic_summary, typed_text FROM screen_contexts WHERE episode_id = ? "
                       "ORDER BY context_id", (episode_id,))
        acts = list(dict.fromkeys(r[0] for r in rows if r[0]))
        typed = next((r[1] for r in reversed(rows) if r[1]), "")
        if len(acts) > 3:
            acts = [acts[0], acts[len(acts) // 2], acts[-1]]
        return " / ".join(acts), typed

    def _episode_hit(self, row, score: float = 0.0) -> Hit:
        eid, s, e, app, title, summary, _status, frames = row
        live, typed = self._live_description(eid)
        return Hit(f"E{eid}", parse_db(s), parse_db(e), app, title, summary or live, typed[:300], "",
                   score, eid, {"frames": frames})

    def timeline(self, start: datetime, end: datetime, apps=None, limit: int = 40, order: str = "asc"
                 ) -> tuple[list[Hit], int]:
        sql = """SELECT e.episode_id, e.start_at, e.end_at, a.display_name, e.norm_title, e.summary,
                        e.summary_status, e.frame_count
                 FROM episodes e JOIN applications a USING (app_id)
                 WHERE e.end_at >= ? AND e.start_at <= ? AND e.frame_count > 0"""
        args: list = [db_fmt(start), db_fmt(end)]
        if apps:
            ids = self.resolve_apps(apps)
            if not ids:
                return [], 0
            sql += f" AND e.app_id IN ({','.join('?' * len(ids))})"
            args += list(ids)
        sql += " ORDER BY e.start_at " + ("DESC" if order == "desc" else "ASC")
        rows = self._q(sql, args)
        return [self._episode_hit(r) for r in rows[:limit]], len(rows)

    # ---- search -----------------------------------------------------------
    def search(self, query: str, start: datetime | None = None, end: datetime | None = None, apps=None,
               k: int = 8) -> list[Hit]:
        app_ids = self.resolve_apps(apps) if apps else None
        if apps and not app_ids:
            return []
        qv = embed_query(self.embedder, query)
        self.cache.refresh(len(qv))
        vec_ids = self.cache.top(qv, 100, start, end, app_ids)
        fts_ids: list[int] = []
        fq = fts_query(query)
        if fq:
            sql = ("SELECT c.chunk_id FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
                   "WHERE chunks_fts MATCH ?")
            args: list = [fq]
            if start is not None:
                sql += " AND c.captured_at >= ?"
                args.append(db_fmt(start))
            if end is not None:
                sql += " AND c.captured_at <= ?"
                args.append(db_fmt(end))
            if app_ids:
                sql += f" AND c.app_id IN ({','.join('?' * len(app_ids))})"
                args += list(app_ids)
            sql += " ORDER BY bm25(chunks_fts) LIMIT 100"
            fts_ids = [r[0] for r in self._q(sql, args)]
        scores = rrf(vec_ids, fts_ids)
        if not scores:
            return []
        ids = list(scores)
        rows = self._q(f"SELECT chunk_id, kind, context_id, episode_id, digest_id, captured_at, text FROM chunks "
                       f"WHERE chunk_id IN ({','.join('?' * len(ids))})", ids)
        now = self.clock()
        agg: dict[str, dict] = {}  # source -> chunk scores + best screen-text chunk
        for cid, kind, ctx, ep, dg, cap, text in rows:
            s = scores[cid]
            if start is None and end is None:
                age_h = max(0.0, (now - parse_db(cap)).total_seconds() / 3600)
                s += 0.002 * math.exp(-age_h / 48)
            src = f"F{ctx}" if ctx else f"E{ep}" if ep else f"D{dg}"
            a = agg.setdefault(src, {"scores": [], "ocr": None})
            a["scores"].append(s)
            if kind == "ocr" and (a["ocr"] is None or s > a["ocr"][0]):
                a["ocr"] = (s, text)
        best = {src: (max(a["scores"]) + 0.1 * (sum(a["scores"]) - max(a["scores"])),
                      a["ocr"][1] if a["ocr"] else "") for src, a in agg.items()}
        terms = _terms(query)
        hits: list[Hit] = []
        per_episode: dict[int, int] = {}
        seen_acts: set = set()
        for src, (s, text) in sorted(best.items(), key=lambda kv: -kv[1][0]):
            h = self._hit_for(src, s, text, terms)
            if not h:
                continue
            if h.episode_id is not None and src.startswith("F"):
                if per_episode.get(h.episode_id, 0) >= 2:
                    continue
                per_episode[h.episode_id] = per_episode.get(h.episode_id, 0) + 1
            key = (h.app, " ".join(h.activity.lower().split())[:120], h.typed[:80])
            if key in seen_acts:
                continue
            seen_acts.add(key)
            hits.append(h)
            if len(hits) >= k:
                break
        return hits

    def _hit_for(self, src: str, score: float, text: str, terms: list[str]) -> Hit | None:
        n = int(src[1:])
        if src[0] == "F":
            r = self._q("""SELECT c.captured_at, a.display_name, c.window_title, c.semantic_summary, c.typed_text,
                                  c.episode_id, c.norm_title
                           FROM screen_contexts c JOIN applications a USING (app_id) WHERE c.context_id = ?""", (n,))
            if not r:
                return None
            cap, app, title, summ, typed, ep, norm = r[0]
            snippet = _snippet(text, terms, 600) if text else ""
            t = parse_db(cap)
            return Hit(src, t, t, app, norm or title or "", summ, typed[:400], snippet, score, ep)
        if src[0] == "E":
            r = self._q("""SELECT e.episode_id, e.start_at, e.end_at, a.display_name, e.norm_title, e.summary,
                                  e.summary_status, e.frame_count
                           FROM episodes e JOIN applications a USING (app_id) WHERE e.episode_id = ?""", (n,))
            return self._episode_hit(r[0], score) if r else None
        r = self._q("SELECT start_time, end_time, macro_summary FROM historical_digests WHERE digest_id = ?", (n,))
        if not r:
            return None
        return Hit(src, parse_db(r[0][0]), parse_db(r[0][1]), "", "digest", r[0][2], "", "", score, None)

    # ---- sources / days ---------------------------------------------------
    def source(self, source_id: str, max_chars: int = 6000) -> str:
        sid = source_id.strip().strip("[]#").upper()
        if not re.fullmatch(r"[FED]\d+", sid):
            return f"Unknown source id {source_id!r}. Use ids like F12, E3 or D1 from earlier results."
        n = int(sid[1:])
        if sid[0] == "F":
            r = self._q("""SELECT c.captured_at, a.display_name, c.window_title, c.semantic_summary, c.typed_text,
                                  c.entities, c.raw_ocr_text
                           FROM screen_contexts c JOIN applications a USING (app_id) WHERE c.context_id = ?""", (n,))
            if not r:
                return f"No source {sid}."
            cap, app, title, summ, typed, ent, ocr = r[0]
            out = (f"[{sid}] {fmt_local(cap, date=True, tz=self.tz)} {app} | {title}\nactivity: {summ}\n"
                   + (f"typed: \"{typed}\"\n" if typed else "") + (f"entities: {ent}\n" if ent else "")
                   + f"screen text:\n{ocr or '(none captured)'}")
            return out[:max_chars]
        if sid[0] == "E":
            r = self._q("""SELECT e.episode_id, e.start_at, e.end_at, a.display_name, e.norm_title, e.summary,
                                  e.summary_status, e.frame_count
                           FROM episodes e JOIN applications a USING (app_id) WHERE e.episode_id = ?""", (n,))
            if not r:
                return f"No source {sid}."
            h = self._episode_hit(r[0])
            frames = self._q("SELECT typed_text, raw_ocr_text FROM screen_contexts WHERE episode_id = ? "
                             "ORDER BY context_id", (n,))
            typed = list(dict.fromkeys(f[0] for f in frames if f[0]))
            texts = [f[1] for f in frames if f[1]]
            longest = max(texts, key=len) if texts else ""
            last = texts[-1] if texts else ""
            out = (f"[{sid}] {fmt_local(h.start, date=True, tz=self.tz)}-{fmt_local(h.end, tz=self.tz)} "
                   f"{h.app} | {h.title}\nsummary: {h.activity}\n"
                   + "".join(f"typed: \"{t}\"\n" for t in typed[-5:])
                   + f"screen text (latest):\n{last}\n"
                   + (f"\nscreen text (longest):\n{longest}" if longest and longest != last else ""))
            return out[:max_chars]
        r = self._q("SELECT start_time, end_time, macro_summary, level FROM historical_digests WHERE digest_id = ?",
                    (n,))
        if not r:
            return f"No source {sid}."
        return f"[{sid}] {r[0][3]} digest {fmt_local(r[0][0], date=True, tz=self.tz)}-" \
               f"{fmt_local(r[0][1], tz=self.tz)}\n{r[0][2]}"[:max_chars]

    def day_bounds(self, day: date) -> tuple[datetime, datetime]:
        tzinfo = self.tz or self.clock().astimezone().tzinfo
        s = datetime(day.year, day.month, day.day, tzinfo=tzinfo).astimezone(timezone.utc)
        return s, s + timedelta(days=1)

    def day_summary(self, day: date) -> tuple[str, list[Hit]]:
        s, e = self.day_bounds(day)
        hits: list[Hit] = []
        d = self._q("SELECT digest_id, start_time, end_time, macro_summary FROM historical_digests "
                    "WHERE level = 'day' AND start_time >= ? AND start_time < ?", (db_fmt(s), db_fmt(e)))
        if d:
            h = Hit(f"D{d[0][0]}", parse_db(d[0][1]), parse_db(d[0][2]), "", "day digest", d[0][3])
            return f"[{h.source_id}] {h.activity}", [h]
        lines = []
        for did, st, en, txt in self._q(
                "SELECT digest_id, start_time, end_time, macro_summary FROM historical_digests "
                "WHERE level = 'hour' AND start_time >= ? AND start_time < ? ORDER BY start_time",
                (db_fmt(s), db_fmt(e))):
            h = Hit(f"D{did}", parse_db(st), parse_db(en), "", "hour digest", txt)
            hits.append(h)
            lines.append(f"[{h.source_id}] {fmt_local(h.start, tz=self.tz)}-{fmt_local(h.end, tz=self.tz)} {txt}")
        covered_until = max((h.end for h in hits), default=s)
        eps, total = self.timeline(covered_until, min(e, self.clock()), limit=60)
        for h in eps:
            hits.append(h)
            lines.append(f"[{h.source_id}] {fmt_local(h.start, tz=self.tz)}-{fmt_local(h.end, tz=self.tz)} "
                         f"{h.app} | {h.title}: {h.activity}")
        if total > len(eps):
            lines.append(f"(+{total - len(eps)} more episodes)")
        return ("\n".join(lines) if lines else ""), hits
