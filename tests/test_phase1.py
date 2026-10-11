import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aegis_context.backends import HashingEmbedder, VisionResult, parse_vision
from aegis_context.capture import Frame, WindowInfo
from aegis_context.chunking import chunk_text
from aegis_context.daemon import CaptureDaemon
from aegis_context.ingest import Ingestor
from aegis_context.ringbuffer import RingBuffer
from aegis_context.storage import AegisStorageEngine

E = HashingEmbedder(384)
T0 = datetime(2026, 10, 11, 17, 0, tzinfo=timezone.utc)

V0_SCHEMA = """
CREATE TABLE applications (app_id INTEGER PRIMARY KEY AUTOINCREMENT, bundle_identifier TEXT UNIQUE NOT NULL,
  display_name TEXT NOT NULL, os_platform TEXT NOT NULL);
CREATE TABLE screen_contexts (context_id INTEGER PRIMARY KEY AUTOINCREMENT, app_id INTEGER NOT NULL, window_title TEXT,
  raw_ocr_text TEXT NOT NULL, semantic_summary TEXT NOT NULL, captured_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  embedding_blob BLOB, FOREIGN KEY (app_id) REFERENCES applications(app_id) ON DELETE CASCADE);
CREATE TABLE historical_digests (digest_id INTEGER PRIMARY KEY AUTOINCREMENT, start_time DATETIME NOT NULL,
  end_time DATETIME NOT NULL, macro_summary TEXT NOT NULL, digest_embedding BLOB);
INSERT INTO applications VALUES (1, 'code.exe', 'Code', 'windows'), (2, 'python.exe', 'python', 'windows');
INSERT INTO screen_contexts (app_id, window_title, raw_ocr_text, semantic_summary, captured_at) VALUES
  (1, '✳ storage.py', 'def hybrid_search(): pass', 'editing storage', '2026-10-11 17:00:00'),
  (1, '◐ storage.py', 'def hybrid_search(): return 1', 'editing storage', '2026-10-11 17:01:00'),
  (2, 'Aegis  (Esc to hide)', 'hud text', 'hud', '2026-10-11 17:02:00'),
  (1, 'hud.py', 'class HUD', 'editing hud', '2026-10-11 17:30:00');
INSERT INTO historical_digests (start_time, end_time, macro_summary) VALUES ('a', 'b', 'title spam');
"""


class MigrationTests(unittest.TestCase):
    def test_v0_file_migrates_and_backfills(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "aegis_vectors.db"
            c = sqlite3.connect(p)
            c.executescript(V0_SCHEMA)
            c.commit()
            c.close()
            db = AegisStorageEngine(p)
            self.assertTrue(db.needs_backfill)
            self.assertEqual(db.conn.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertTrue((Path(d) / "aegis_vectors.v0.bak.db").exists())
            self.assertEqual(Ingestor(db, E).backfill(progress=None), 3)  # hud row deleted
            self.assertEqual(db.count("historical_digests"), 0)
            eps = db.conn.execute("SELECT norm_title, frame_count, summary_status FROM episodes ORDER BY 1").fetchall()
            self.assertEqual(eps, [("hud.py", 1, "pending"), ("storage.py", 2, "pending")])
            hits = db.conn.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'hybrid_search'").fetchone()
            self.assertGreaterEqual(hits[0], 1)
            db.close()
            db2 = AegisStorageEngine(p)  # idempotent reopen
            self.assertFalse(db2.needs_backfill)
            self.assertEqual(db2.count(), 3)
            db2.close()

    def test_fresh_db_is_v1(self):
        db = AegisStorageEngine(":memory:")
        self.assertEqual(db.conn.execute("PRAGMA user_version").fetchone()[0], 1)
        self.assertFalse(db.needs_backfill)


class ChunkTests(unittest.TestCase):
    def test_chunking(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("short"), ["short"])
        text = "\n\n".join(f"paragraph {i} " + "word " * 60 for i in range(10))
        cs = chunk_text(text, 800, 150)
        self.assertGreater(len(cs), 3)
        self.assertTrue(all(len(c) <= 950 for c in cs))
        self.assertIn("paragraph 9", cs[-1])
        long_line = "x" * 2000
        self.assertTrue(all(len(c) <= 950 for c in chunk_text(long_line)))


class VisionParserTests(unittest.TestCase):
    def test_well_formed(self):
        r = parse_vision("ACTIVITY: Editing storage.py\nTYPED: NONE\nENTITIES: storage.py\nTEXT:\nline 1\nline 2")
        self.assertEqual((r.activity, r.typed, r.entities, r.text), ("Editing storage.py", "", "storage.py", "line 1\nline 2"))

    def test_fenced_bold_missing(self):
        r = parse_vision("```\n**ACTIVITY:** Chatting with Claude\n**TYPED:** \"build the android app\"\n```")
        self.assertEqual(r.activity, "Chatting with Claude")
        self.assertEqual(r.typed, "build the android app")
        self.assertEqual(r.text, "")

    def test_unformatted(self):
        r = parse_vision("The user is reading docs.")
        self.assertEqual(r.activity, "The user is reading docs.")


def W(title="storage.py", app="Code"):
    return WindowInfo(app.lower() + ".exe", app, title, "windows")


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.db, self.ring = AegisStorageEngine(":memory:"), RingBuffer(window_s=10 ** 9)
        self.ing = Ingestor(self.db, E, self.ring)

    def put(self, minutes, res, title="storage.py", app="Code"):
        w = W(title, app)
        return self.ing.ingest(w.bundle_id, w.app_name, w.platform, w.title, res, T0 + timedelta(minutes=minutes))

    def test_near_duplicate_extends_and_updates_typed(self):
        text = "chat history line one two three four five six seven eight nine ten"
        self.assertIsNotNone(self.put(0, VisionResult("chatting", "build the", "", text)))
        self.assertIsNone(self.put(0.5, VisionResult("chatting", "build the android app", "", text)))
        self.assertEqual(self.db.count(), 1)
        self.assertEqual(self.db.conn.execute("SELECT typed_text FROM screen_contexts").fetchone()[0],
                         "build the android app")
        n = self.db.conn.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'android'").fetchone()[0]
        self.assertEqual(n, 1)

    def test_episode_segmentation(self):
        self.put(0, VisionResult("a", "", "", "alpha text"))
        self.put(1, VisionResult("b", "", "", "beta words entirely different"))   # same episode
        self.put(2, VisionResult("c", "", "", "gamma"), title="hud.py")           # new title -> new episode
        self.put(10, VisionResult("d", "", "", "delta"), title="hud.py")          # gap 8 min > 3 -> new episode
        self.put(11, VisionResult("e", "", "", "eps"), title="✳ hud.py")          # spinner ignored -> same
        eps = self.db.conn.execute("SELECT norm_title, frame_count FROM episodes ORDER BY episode_id").fetchall()
        self.assertEqual(eps, [("storage.py", 2), ("hud.py", 1), ("hud.py", 2)])
        st = [r[0] for r in self.db.conn.execute("SELECT summary_status FROM episodes ORDER BY episode_id")]
        self.assertEqual(st, ["pending", "pending", "open"])

    def test_secrets_redacted_on_ingest(self):
        self.put(0, VisionResult("env", "", "", "OLLAMA_API_KEY=0123456789abcdef0123456789abcdef.FakeKeyAAAAbbbbCCCCdddd1"))
        blob = " ".join(r[0] for r in self.db.conn.execute("SELECT text FROM chunks"))
        self.assertNotIn("0123456789ab", blob)


class DhashSkipTests(unittest.TestCase):
    def test_small_pixel_change_skipped(self):
        db = AegisStorageEngine(":memory:")
        frames = iter([Frame(b"a", 0b1111), Frame(b"b", 0b1110), Frame(b"c", (1 << 200) - 1)])

        class V:
            def describe(self, png):
                return "text " + png.decode(), "did " + png.decode()

        d = CaptureDaemon(db, RingBuffer(), E, V(), lambda: next(frames), lambda: W())
        self.assertEqual([d.tick(), d.tick(), d.tick()], [True, False, True])

    def test_idle_ignores_pixel_churn(self):
        db = AegisStorageEngine(":memory:")
        frames = iter([Frame(b"a", 0), Frame(b"c", (1 << 200) - 1)])

        class V:
            def describe(self, png):
                return png.decode(), png.decode()

        d = CaptureDaemon(db, RingBuffer(), E, V(), lambda: next(frames), lambda: W(), idle=lambda: 999)
        self.assertEqual([d.tick(), d.tick()], [True, False])
