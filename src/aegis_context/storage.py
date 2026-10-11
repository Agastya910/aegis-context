"""Tier-2 warm store: SQLite + float32 embedding BLOBs, with Tier-3 rolling digests."""
from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    app_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bundle_identifier TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    os_platform TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screen_contexts (
    context_id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id INTEGER NOT NULL,
    window_title TEXT,
    raw_ocr_text TEXT NOT NULL,
    semantic_summary TEXT NOT NULL,
    captured_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    embedding_blob BLOB,
    FOREIGN KEY (app_id) REFERENCES applications(app_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_contexts_timestamp ON screen_contexts(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_contexts_app ON screen_contexts(app_id);
CREATE TABLE IF NOT EXISTS historical_digests (
    digest_id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    macro_summary TEXT NOT NULL,
    digest_embedding BLOB
);
"""

V1 = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id INTEGER NOT NULL REFERENCES applications(app_id),
    norm_title TEXT NOT NULL,
    start_at DATETIME NOT NULL,
    end_at DATETIME NOT NULL,
    frame_count INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    summary_status TEXT NOT NULL DEFAULT 'open',
    summary_attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(start_at, end_at);
ALTER TABLE screen_contexts ADD COLUMN episode_id INTEGER REFERENCES episodes(episode_id);
ALTER TABLE screen_contexts ADD COLUMN norm_title TEXT NOT NULL DEFAULT '';
ALTER TABLE screen_contexts ADD COLUMN typed_text TEXT NOT NULL DEFAULT '';
ALTER TABLE screen_contexts ADD COLUMN entities TEXT NOT NULL DEFAULT '';
ALTER TABLE screen_contexts ADD COLUMN image_hash TEXT;
CREATE INDEX IF NOT EXISTS idx_contexts_episode ON screen_contexts(episode_id);
ALTER TABLE historical_digests ADD COLUMN level TEXT NOT NULL DEFAULT 'hour';
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    context_id INTEGER REFERENCES screen_contexts(context_id) ON DELETE CASCADE,
    episode_id INTEGER REFERENCES episodes(episode_id) ON DELETE CASCADE,
    digest_id INTEGER REFERENCES historical_digests(digest_id) ON DELETE CASCADE,
    captured_at DATETIME NOT NULL,
    app_id INTEGER,
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_time ON chunks(captured_at);
CREATE INDEX IF NOT EXISTS idx_chunks_context ON chunks(context_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.chunk_id;
END;
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    memo TEXT NOT NULL DEFAULT '',
    created_at DATETIME NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, message_id);
"""
MIGRATIONS = [V1]

_TS = "%Y-%m-%d %H:%M:%S"
_STOP = set(
    "a an and are as at be by do did does for from had has have i in is it me my of on or so that the this "
    "to was we were what when where which who why with you your am about into just can could would should".split()
)


def document_text(app: str, title: str, summary: str, ocr: str, max_ocr: int = 2000) -> str:
    """Text that gets embedded for a screen state: metadata and summary first, then capped screen text."""
    return " ".join(filter(None, [app, title, summary, (ocr or "")[:max_ocr]]))


def _terms(text: str) -> list[str]:
    return [t for t in dict.fromkeys(re.findall(r"[a-z0-9_]+", text.lower())) if t not in _STOP and len(t) > 1]


def _snippet(text: str, terms: list[str], width: int = 700) -> str:
    """The part of the screen text around the first query-term hit (or its start)."""
    if len(text) <= width:
        return text
    low = text.lower()
    hits = [i for i in (low.find(t) for t in terms) if i >= 0]
    start = max(0, min(hits) - width // 3) if hits else 0
    return ("…" if start else "") + text[start:start + width] + "…"


def _fmt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(_TS)


def serialize_vector(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def deserialize_vector(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


class AegisStorageEngine:
    def __init__(self, db_path: str | Path = "aegis_vectors.db"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.needs_backfill = False
        existed = self.db_path != ":memory:" and Path(self.db_path).exists() and Path(self.db_path).stat().st_size > 0
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()
            self._migrate(backup=existed)

    def _migrate(self, backup: bool) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= len(MIGRATIONS):
            return
        if backup:
            dst = Path(self.db_path).with_name(f"{Path(self.db_path).stem}.v{version}.bak.db")
            if not dst.exists():
                self.conn.execute("VACUUM INTO ?", (str(dst),))
        for i in range(version, len(MIGRATIONS)):
            self.conn.executescript(f"BEGIN;\n{MIGRATIONS[i]}\nPRAGMA user_version = {i + 1};\nCOMMIT;")
        self.needs_backfill = version == 0 and backup

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---- writes ---------------------------------------------------------
    def log_screen_state(
        self,
        bundle_id: str,
        app_name: str,
        platform: str,
        window_title: str,
        ocr_text: str,
        summary: str,
        embedding: np.ndarray | None,
        captured_at: datetime | None = None,
    ) -> int:
        blob = serialize_vector(embedding) if embedding is not None else None
        ts = _fmt(captured_at or datetime.now(timezone.utc))
        with self._lock:
            self.conn.execute(
                """INSERT INTO applications (bundle_identifier, display_name, os_platform)
                   VALUES (?, ?, ?)
                   ON CONFLICT(bundle_identifier) DO UPDATE SET display_name=excluded.display_name""",
                (bundle_id, app_name, platform),
            )
            (app_id,) = self.conn.execute(
                "SELECT app_id FROM applications WHERE bundle_identifier = ?", (bundle_id,)
            ).fetchone()
            cur = self.conn.execute(
                """INSERT INTO screen_contexts
                   (app_id, window_title, raw_ocr_text, semantic_summary, captured_at, embedding_blob)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (app_id, window_title, ocr_text, summary, ts, blob),
            )
            self.conn.commit()
            return cur.lastrowid

    # ---- reads ----------------------------------------------------------
    def vector_similarity_search(
        self, query_vector: np.ndarray, top_k: int = 5, since: datetime | None = None
    ) -> list[dict]:
        """Brute-force cosine search, vectorised with one matrix multiply."""
        sql = """SELECT c.context_id, a.display_name, c.window_title, c.semantic_summary,
                        c.captured_at, c.embedding_blob
                 FROM screen_contexts c JOIN applications a USING (app_id)
                 WHERE c.embedding_blob IS NOT NULL"""
        args: list = []
        if since is not None:
            sql += " AND c.captured_at >= ?"
            args.append(_fmt(since))
        with self._lock:
            rows = self.conn.execute(sql, args).fetchall()
        q = np.asarray(query_vector, dtype=np.float32)
        rows = [r for r in rows if len(r[5]) == q.nbytes]  # skip vectors from a different embedder
        if not rows:
            return []
        mat = np.stack([deserialize_vector(r[5]) for r in rows])
        sims = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-10)
        order = np.argsort(-sims)[:top_k]
        return [
            {
                "similarity_score": float(sims[i]),
                "context_id": rows[i][0],
                "app": rows[i][1],
                "window_title": rows[i][2],
                "summary": rows[i][3],
                "timestamp": rows[i][4],
            }
            for i in order
        ]

    def hybrid_search(
        self, query_text: str, query_vector: np.ndarray, top_k: int = 5, since: datetime | None = None,
        candidates: int = 2000,
    ) -> list[dict]:
        """Vector + keyword search over title/summary/screen text, de-duplicated, lightly recency-boosted.

        Pure vector search lets short title-only rows ("Claude: Claude") beat rows whose screen text
        actually contains the answer; keyword overlap on the OCR text and an empty-row penalty fix that.
        """
        sql = """SELECT c.context_id, a.display_name, c.window_title, c.semantic_summary,
                        c.captured_at, c.embedding_blob, c.raw_ocr_text
                 FROM screen_contexts c JOIN applications a USING (app_id)"""
        args: list = []
        if since is not None:
            sql += " WHERE c.captured_at >= ?"
            args.append(_fmt(since))
        sql += " ORDER BY c.captured_at DESC LIMIT ?"
        args.append(candidates)
        with self._lock:
            rows = self.conn.execute(sql, args).fetchall()
        if not rows:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        qn = np.linalg.norm(q) + 1e-10
        terms = _terms(query_text)
        now = datetime.now(timezone.utc)
        scored = []
        for r in rows:
            blob, ocr = r[5], r[6] or ""
            vec = 0.0
            if blob is not None and len(blob) == q.nbytes:
                v = deserialize_vector(blob)
                vec = float(v @ q / (np.linalg.norm(v) * qn + 1e-10))
            hay = f"{r[2] or ''} {r[3]} {ocr}".lower()
            kw = sum(t in hay for t in terms) / len(terms) if terms else 0.0
            score = 0.7 * vec + 0.3 * kw
            if not ocr.strip():
                score *= 0.6  # title-only rows carry little information
            try:
                age_h = (now - datetime.strptime(r[4], _TS).replace(tzinfo=timezone.utc)).total_seconds() / 3600
                score += 0.05 * float(np.exp(-age_h / 24))
            except (TypeError, ValueError):
                pass
            scored.append((score, vec, kw, r))
        scored.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for score, vec, kw, r in scored:
            key = (r[1], " ".join((r[3] or "").lower().split())[:120], (r[6] or "")[:200])
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "similarity_score": score, "vector_score": vec, "keyword_score": kw,
                "context_id": r[0], "app": r[1], "window_title": r[2], "summary": r[3],
                "timestamp": r[4], "screen_text": _snippet(r[6] or "", terms),
            })
            if len(out) >= top_k:
                break
        return out

    def search_digests(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT digest_id, start_time, end_time, macro_summary, digest_embedding "
                "FROM historical_digests WHERE digest_embedding IS NOT NULL"
            ).fetchall()
        q = np.asarray(query_vector, dtype=np.float32)
        rows = [r for r in rows if len(r[4]) == q.nbytes]
        if not rows:
            return []
        mat = np.stack([deserialize_vector(r[4]) for r in rows])
        sims = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-10)
        return [
            {"similarity_score": float(sims[i]), "start": rows[i][1], "end": rows[i][2], "summary": rows[i][3]}
            for i in np.argsort(-sims)[:top_k]
        ]

    def count(self, table: str = "screen_contexts") -> int:
        assert table in {"screen_contexts", "applications", "historical_digests"}
        with self._lock:
            return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ---- maintenance ----------------------------------------------------
    def rollup_digests(self, embedder, older_than_s: int = 3600, block_s: int = 3600, max_chars: int = 1500) -> int:
        """Fold screen_contexts older than `older_than_s` into hour-block digests, then delete them."""
        cutoff = _fmt(datetime.now(timezone.utc) - timedelta(seconds=older_than_s))
        with self._lock:
            rows = self.conn.execute(
                """SELECT c.context_id, c.captured_at, a.display_name, c.window_title, c.semantic_summary
                   FROM screen_contexts c JOIN applications a USING (app_id)
                   WHERE c.captured_at < ? ORDER BY c.captured_at""",
                (cutoff,),
            ).fetchall()
            if not rows:
                return 0
            blocks: dict[int, list] = {}
            for r in rows:
                t = datetime.strptime(r[1], _TS).replace(tzinfo=timezone.utc).timestamp()
                blocks.setdefault(int(t // block_s), []).append(r)
            for key, items in blocks.items():
                seen, parts = set(), []
                for _, _, app, title, summ in items:
                    line = f"{app}: {title or ''} - {summ}".strip(" -:")
                    if line not in seen:
                        seen.add(line)
                        parts.append(line)
                text = " | ".join(parts)[:max_chars]
                self.conn.execute(
                    "INSERT INTO historical_digests (start_time, end_time, macro_summary, digest_embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (items[0][1], items[-1][1], text, serialize_vector(embedder.embed(text))),
                )
            ids = [r[0] for r in rows]
            self.conn.executemany("DELETE FROM screen_contexts WHERE context_id = ?", [(i,) for i in ids])
            self.conn.commit()
            return len(ids)

    def reembed(self, embedder) -> int:
        """Recompute every stored vector with `embedder` (after switching embedding models)."""
        with self._lock:
            ctx = self.conn.execute(
                """SELECT c.context_id, a.display_name, c.window_title, c.semantic_summary, c.raw_ocr_text
                   FROM screen_contexts c JOIN applications a USING (app_id)"""
            ).fetchall()
            dig = self.conn.execute("SELECT digest_id, macro_summary FROM historical_digests").fetchall()
        ctx_vecs = [
            (serialize_vector(embedder.embed(document_text(app, title, summ, ocr))), cid)
            for cid, app, title, summ, ocr in ctx
        ]
        dig_vecs = [(serialize_vector(embedder.embed(text)), did) for did, text in dig]
        with self._lock:
            self.conn.executemany("UPDATE screen_contexts SET embedding_blob = ? WHERE context_id = ?", ctx_vecs)
            self.conn.executemany("UPDATE historical_digests SET digest_embedding = ? WHERE digest_id = ?", dig_vecs)
            self.conn.commit()
        return len(ctx_vecs) + len(dig_vecs)

    # ---- v1: apps, frames, chunks, episodes -----------------------------
    def upsert_app(self, bundle_id: str, app_name: str, platform: str) -> int:
        with self._lock:
            self.conn.execute(
                """INSERT INTO applications (bundle_identifier, display_name, os_platform) VALUES (?, ?, ?)
                   ON CONFLICT(bundle_identifier) DO UPDATE SET display_name=excluded.display_name""",
                (bundle_id, app_name, platform),
            )
            return self.conn.execute(
                "SELECT app_id FROM applications WHERE bundle_identifier = ?", (bundle_id,)).fetchone()[0]

    def insert_frame(self, app_id: int, title: str, norm_title: str, ocr: str, summary: str, typed: str,
                     entities: str, image_hash: str | None, captured_at: datetime, episode_id: int | None,
                     embedding: np.ndarray | None) -> int:
        blob = serialize_vector(embedding) if embedding is not None else None
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO screen_contexts (app_id, window_title, raw_ocr_text, semantic_summary, captured_at,
                       embedding_blob, episode_id, norm_title, typed_text, entities, image_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (app_id, title, ocr, summary, _fmt(captured_at), blob, episode_id, norm_title, typed, entities,
                 image_hash),
            )
            self.conn.commit()
            return cur.lastrowid

    def add_chunks(self, kind: str, texts: list[str], vecs, captured_at: datetime | str, app_id: int | None,
                   context_id: int | None = None, episode_id: int | None = None, digest_id: int | None = None) -> None:
        ts = captured_at if isinstance(captured_at, str) else _fmt(captured_at)
        rows = [(kind, context_id, episode_id, digest_id, ts, app_id, t,
                 serialize_vector(v) if v is not None else None) for t, v in zip(texts, vecs)]
        with self._lock:
            self.conn.executemany(
                """INSERT INTO chunks (kind, context_id, episode_id, digest_id, captured_at, app_id, text, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", rows)
            self.conn.commit()

    def delete_chunks(self, kind: str, context_id: int | None = None, episode_id: int | None = None,
                      digest_id: int | None = None) -> None:
        col, val = (("context_id", context_id) if context_id is not None else
                    ("episode_id", episode_id) if episode_id is not None else ("digest_id", digest_id))
        with self._lock:
            self.conn.execute(f"DELETE FROM chunks WHERE kind = ? AND {col} = ?", (kind, val))
            self.conn.commit()

    def open_episode(self, app_id: int, norm_title: str, ts: datetime) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO episodes (app_id, norm_title, start_at, end_at, frame_count) VALUES (?, ?, ?, ?, 0)",
                (app_id, norm_title, _fmt(ts), _fmt(ts)))
            self.conn.commit()
            return cur.lastrowid

    def touch_episode(self, episode_id: int, ts: datetime, add_frame: bool = False) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE episodes SET end_at = MAX(end_at, ?), frame_count = frame_count + ? WHERE episode_id = ?",
                (_fmt(ts), int(add_frame), episode_id))
            self.conn.commit()

    def close_episode(self, episode_id: int) -> None:
        with self._lock:
            self.conn.execute("UPDATE episodes SET summary_status = 'pending' "
                              "WHERE episode_id = ? AND summary_status = 'open'", (episode_id,))
            self.conn.commit()

    def close_open_episodes(self) -> int:
        with self._lock:
            n = self.conn.execute("UPDATE episodes SET summary_status = 'pending' "
                                  "WHERE summary_status = 'open'").rowcount
            self.conn.commit()
            return n

    def last_frame(self, episode_id: int) -> dict | None:
        with self._lock:
            r = self.conn.execute(
                """SELECT context_id, raw_ocr_text, typed_text, semantic_summary, entities, window_title, captured_at
                   FROM screen_contexts WHERE episode_id = ? ORDER BY context_id DESC LIMIT 1""",
                (episode_id,)).fetchone()
        if not r:
            return None
        return dict(zip(["context_id", "ocr", "typed", "summary", "entities", "title", "captured_at"], r))

    def update_frame_typed(self, context_id: int, typed: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE screen_contexts SET typed_text = ? WHERE context_id = ?", (typed, context_id))
            self.conn.commit()

    def set_frame_embedding(self, context_id: int, vec) -> None:
        with self._lock:
            self.conn.execute("UPDATE screen_contexts SET embedding_blob = ? WHERE context_id = ?",
                              (serialize_vector(vec), context_id))
            self.conn.commit()

    def redact_existing(self, embedder=None) -> int:
        """Scrub secrets from stored frames; re-embed the ones that changed."""
        from .redact import redact

        with self._lock:
            rows = self.conn.execute(
                """SELECT c.context_id, a.display_name, c.window_title, c.semantic_summary, c.raw_ocr_text
                   FROM screen_contexts c JOIN applications a USING (app_id)"""
            ).fetchall()
        changed = []
        for cid, app, title, summ, ocr in rows:
            s2, o2 = redact(summ), redact(ocr)
            if (s2, o2) != (summ, ocr):
                blob = serialize_vector(embedder.embed(document_text(app, title, s2, o2))) if embedder else None
                changed.append((s2, o2, blob, cid))
        with self._lock:
            self.conn.executemany(
                "UPDATE screen_contexts SET semantic_summary=?, raw_ocr_text=?, "
                "embedding_blob=COALESCE(?, embedding_blob) WHERE context_id=?", changed)
            chunks = self.conn.execute("SELECT chunk_id, text FROM chunks").fetchall()
            fixed = [(redact(t), cid) for cid, t in chunks if redact(t) != t]
            self.conn.executemany("UPDATE chunks SET text=? WHERE chunk_id=?", fixed)
            self.conn.executemany("UPDATE chunks_fts SET text=? WHERE rowid=?", fixed)
            self.conn.commit()
        if embedder is not None and fixed:
            from .backends import embed_many

            vecs = embed_many(embedder, [t for t, _ in fixed])
            with self._lock:
                self.conn.executemany("UPDATE chunks SET embedding=? WHERE chunk_id=?",
                                      [(serialize_vector(v), cid) for v, (_, cid) in zip(vecs, fixed)])
                self.conn.commit()
        return len(changed) + len(fixed)

    def prune_frames(self, raw_retention_days: int) -> int:
        """Delete frames older than the raw retention window (never anything younger)."""
        cutoff = _fmt(datetime.now(timezone.utc) - timedelta(days=raw_retention_days))
        with self._lock:
            n = self.conn.execute("DELETE FROM screen_contexts WHERE captured_at < ?", (cutoff,)).rowcount
            self.conn.commit()
            return n

    def prune(self, retention_days: int) -> int:
        cutoff = _fmt(datetime.now(timezone.utc) - timedelta(days=retention_days))
        with self._lock:
            n = self.conn.execute("DELETE FROM historical_digests WHERE end_time < ?", (cutoff,)).rowcount
            self.conn.commit()
            return n
