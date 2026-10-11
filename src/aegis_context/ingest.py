"""Turns a captured screen state into stored memory: frame + chunks (+FTS, vectors) inside an episode."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from .backends import VisionResult, embed_many
from .chunking import chunk_text
from .filters import normalize_title
from .redact import redact
from .timeutil import parse_db, utc_now

log = logging.getLogger("aegis.ingest")
_WORD = re.compile(r"\w+")


def jaccard(a: str, b: str) -> float:
    sa, sb = set(_WORD.findall((a or "").lower())), set(_WORD.findall((b or "").lower()))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def meta_text(app: str, title: str, activity: str, typed: str, entities: str) -> str:
    return f"{app} | {title}\nActivity: {activity}\nTyped: {typed}\nEntities: {entities}"


class Ingestor:
    def __init__(self, storage, embedder, ring=None, cfg=None):
        self.storage, self.embedder, self.ring = storage, embedder, ring
        self.gap = timedelta(seconds=getattr(cfg, "episode_gap_s", 180))
        self.max_len = timedelta(seconds=getattr(cfg, "episode_max_s", 1800))
        self.dup = getattr(cfg, "text_dup_threshold", 0.92)
        self.chunk_size = getattr(cfg, "chunk_size", 800)
        self.overlap = getattr(cfg, "chunk_overlap", 150)
        self.cur: dict | None = None  # {id, key, start, last}

    # ---- episodes -------------------------------------------------------
    def _episode_for(self, app_id: int, norm: str, ts: datetime) -> tuple[int, bool]:
        c = self.cur
        if c and c["key"] == (app_id, norm) and ts - c["last"] <= self.gap and ts - c["start"] < self.max_len:
            c["last"] = ts
            return c["id"], True
        if c:
            self.storage.close_episode(c["id"])
        eid = self.storage.open_episode(app_id, norm, ts)
        self.cur = {"id": eid, "key": (app_id, norm), "start": ts, "last": ts}
        return eid, False

    def heartbeat(self, ts: datetime | None = None) -> None:
        """Screen unchanged: keep the current episode's end time fresh."""
        if self.cur:
            ts = ts or utc_now()
            if ts - self.cur["last"] <= self.gap:
                self.cur["last"] = ts
                self.storage.touch_episode(self.cur["id"], ts)

    # ---- frames ---------------------------------------------------------
    def _chunks(self, meta: str, text: str) -> tuple[list[str], list]:
        texts = [meta] + chunk_text(text, self.chunk_size, self.overlap)
        return texts, embed_many(self.embedder, texts)

    def ingest(self, bundle_id: str, app: str, platform: str, title: str, res: VisionResult,
               ts: datetime | None = None, image_hash: str | None = None) -> int | None:
        """Store one screen state. Returns the new frame id, or None if it was a near-duplicate."""
        ts = ts or utc_now()
        activity = redact(res.activity) or f"{app}: {title}".strip(": ")
        typed, entities, text = redact(res.typed), redact(res.entities), redact(res.text)
        app_id = self.storage.upsert_app(bundle_id, app, platform)
        norm = normalize_title(title)
        eid, continued = self._episode_for(app_id, norm, ts)
        if self.ring is not None:
            self.ring.add(app, title, activity, ts=ts.timestamp())
        if continued:
            last = self.storage.last_frame(eid)
            if last and jaccard(last["ocr"] or last["summary"], text or activity) >= self.dup:
                self.storage.touch_episode(eid, ts)
                if typed and typed != last["typed"]:  # user kept typing: keep the latest version of the prompt
                    self.storage.update_frame_typed(last["context_id"], typed)
                    meta = meta_text(app, title, last["summary"], typed, last["entities"])
                    [vec] = embed_many(self.embedder, [meta])
                    self.storage.delete_chunks("meta", context_id=last["context_id"])
                    self.storage.add_chunks("meta", [meta], [vec], last["captured_at"], app_id,
                                            context_id=last["context_id"])
                    self.storage.set_frame_embedding(last["context_id"], vec)
                return None
        texts, vecs = self._chunks(meta_text(app, title, activity, typed, entities), text)
        fid = self.storage.insert_frame(app_id, title, norm, text, activity, typed, entities, image_hash, ts, eid,
                                        vecs[0])
        self.storage.add_chunks("meta", texts[:1], vecs[:1], ts, app_id, context_id=fid)
        if len(texts) > 1:
            self.storage.add_chunks("ocr", texts[1:], vecs[1:], ts, app_id, context_id=fid)
        self.storage.touch_episode(eid, ts, add_frame=True)
        return fid

    # ---- migration backfill ---------------------------------------------
    def backfill(self, progress=print) -> int:
        """Give pre-v1 frames episodes + chunks; drop self-capture rows and title-spam digests; redact."""
        st = self.storage
        with st._lock:
            st.conn.execute("""DELETE FROM screen_contexts WHERE window_title LIKE 'Aegis %' AND app_id IN
                               (SELECT app_id FROM applications WHERE lower(display_name) = 'python')""")
            st.conn.execute("DELETE FROM historical_digests")
            st.conn.commit()
        st.redact_existing(None)
        with st._lock:
            rows = st.conn.execute(
                """SELECT c.context_id, c.app_id, a.display_name, c.window_title, c.semantic_summary,
                          c.raw_ocr_text, c.captured_at
                   FROM screen_contexts c JOIN applications a USING (app_id)
                   WHERE c.episode_id IS NULL ORDER BY c.captured_at, c.context_id""").fetchall()
        self.cur = None
        for i, (cid, app_id, app, title, summ, ocr, cap) in enumerate(rows, 1):
            ts = parse_db(cap)
            norm = normalize_title(title or "")
            eid, _ = self._episode_for(app_id, norm, ts)
            texts, vecs = self._chunks(meta_text(app, title or "", summ, "", ""), ocr)
            with st._lock:
                st.conn.execute("UPDATE screen_contexts SET episode_id=?, norm_title=?, embedding_blob=? "
                                "WHERE context_id=?", (eid, norm, vecs[0].astype("float32").tobytes(), cid))
                st.conn.commit()
            st.add_chunks("meta", texts[:1], vecs[:1], cap, app_id, context_id=cid)
            if len(texts) > 1:
                st.add_chunks("ocr", texts[1:], vecs[1:], cap, app_id, context_id=cid)
            st.touch_episode(eid, ts, add_frame=True)
            if progress and (i % 50 == 0 or i == len(rows)):
                progress(f"  backfilled {i}/{len(rows)} frames")
        st.close_open_episodes()
        self.cur = None
        return len(rows)
