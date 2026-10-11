import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from aegis_context.backends import HashingEmbedder
from aegis_context.capture import WindowInfo
from aegis_context.config import Config
from aegis_context.daemon import CaptureDaemon
from aegis_context.ringbuffer import RingBuffer
from aegis_context.storage import AegisStorageEngine, deserialize_vector, serialize_vector

E = HashingEmbedder(384)


def seed(db):
    for app, title, summ in [
        ("VS Code", "storage.py", "editing sqlite vector search code"),
        ("Chrome", "FastAPI docs", "reading fastapi request body documentation"),
        ("Slack", "#ml", "chatting about cardiac disease model"),
    ]:
        db.log_screen_state(app.lower(), app, "test", title, "", summ, E.embed(f"{app} {title} {summ}"))


class StorageTests(unittest.TestCase):
    def test_vector_roundtrip(self):
        v = np.arange(5, dtype=np.float32)
        self.assertTrue(np.array_equal(deserialize_vector(serialize_vector(v)), v))

    def test_search_ranks_relevant_first(self):
        db = AegisStorageEngine(":memory:")
        seed(db)
        res = db.vector_similarity_search(E.embed("fastapi request body docs"), top_k=3)
        self.assertEqual(res[0]["app"], "Chrome")
        self.assertEqual(len(res), 3)
        self.assertGreaterEqual(res[0]["similarity_score"], res[1]["similarity_score"])

    def test_app_upsert_dedupes(self):
        db = AegisStorageEngine(":memory:")
        seed(db)
        seed(db)
        self.assertEqual(db.count("applications"), 3)
        self.assertEqual(db.count(), 6)

    def test_persistent_file_db(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "v.db"
            db = AegisStorageEngine(path)
            seed(db)
            db.close()
            db2 = AegisStorageEngine(path)
            self.assertEqual(db2.count(), 3)
            db2.close()

    def test_since_filter(self):
        db = AegisStorageEngine(":memory:")
        old = datetime.now(timezone.utc) - timedelta(days=2)
        db.log_screen_state("a", "A", "t", "", "", "old thing", E.embed("old thing"), captured_at=old)
        db.log_screen_state("b", "B", "t", "", "", "new thing", E.embed("new thing"))
        res = db.vector_similarity_search(E.embed("thing"), since=datetime.now(timezone.utc) - timedelta(hours=1))
        self.assertEqual([r["app"] for r in res], ["B"])

    def test_rollup_moves_old_rows_to_digest(self):
        db = AegisStorageEngine(":memory:")
        old = datetime.now(timezone.utc) - timedelta(hours=5)
        db.log_screen_state("a", "A", "t", "t1", "", "debugging parser", E.embed("debugging parser"), captured_at=old)
        db.log_screen_state("b", "B", "t", "", "", "fresh", E.embed("fresh"))
        self.assertEqual(db.rollup_digests(E, older_than_s=3600), 1)
        self.assertEqual(db.count(), 1)
        self.assertEqual(db.count("historical_digests"), 1)
        d = db.search_digests(E.embed("debugging parser"))
        self.assertIn("debugging parser", d[0]["summary"])


class RingTests(unittest.TestCase):
    def test_eviction_and_recent(self):
        r = RingBuffer(window_s=60)
        r.add("old", "", "x", ts=time.time() - 120)
        r.add("new", "", "y")
        self.assertEqual([i["app"] for i in r.recent()], ["new"])
        self.assertEqual(len(r.recent(within_s=1)), 1)


class DaemonTests(unittest.TestCase):
    def test_tick_stores_and_skips_duplicates(self):
        db, ring = AegisStorageEngine(":memory:"), RingBuffer()
        frames = iter([b"f1", b"f1", b"f2"])

        class V:
            def describe(self, png):
                return "ocr " + png.decode(), "did " + png.decode()

        d = CaptureDaemon(db, ring, E, V(), lambda: next(frames),
                          lambda: WindowInfo("x", "X", "title", "test"))
        self.assertEqual([d.tick(), d.tick(), d.tick()], [True, False, True])
        self.assertEqual(db.count(), 2)
        self.assertEqual(len(ring), 2)


class EmbedderConfigTests(unittest.TestCase):
    def test_embedder_deterministic_normalised(self):
        a, b = E.embed("hello world"), E.embed("hello world")
        self.assertTrue(np.array_equal(a, b))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)

    def test_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            c = Config.load(d)
            c.port = 9999
            c.save()
            self.assertEqual(Config.load(d).port, 9999)


try:
    from fastapi.testclient import TestClient
    from aegis_context.ipc_server import BridgeState, create_app
    HAVE_API = True
except Exception:  # fastapi/httpx not installed
    HAVE_API = False


@unittest.skipUnless(HAVE_API, "fastapi + httpx required")
class ApiTests(unittest.TestCase):
    def setUp(self):
        db, ring = AegisStorageEngine(":memory:"), RingBuffer()
        seed(db)
        ring.add("Chrome", "FastAPI docs", "reading docs")
        self.c = TestClient(create_app(BridgeState(db, ring, E)))

    def test_health(self):
        self.assertEqual(self.c.get("/health").json()["stored"], 3)

    def test_recent(self):
        j = self.c.get("/context/recent").json()
        self.assertEqual(j["count"], 1)
        self.assertIn("reading docs", j["speakable"])

    def test_query(self):
        j = self.c.post("/context/query", json={"query_text": "cardiac model"}).json()
        self.assertEqual(j["matches"][0]["app"], "Slack")

    def test_query_validation(self):
        self.assertEqual(self.c.post("/context/query", json={"query_text": ""}).status_code, 422)


class HybridSearchTests(unittest.TestCase):
    def test_screen_text_beats_title_only_rows(self):
        db = AegisStorageEngine(":memory:")
        for _ in range(3):
            db.log_screen_state("claude", "claude", "t", "Claude", "", "claude: Claude", E.embed("claude Claude"))
        ocr = "> please build an android app that calls the laptop over tailscale"
        db.log_screen_state("term", "Terminal", "t", "Claude Code", ocr, "User is chatting with Claude in a terminal",
                            E.embed(f"Terminal Claude Code {ocr}"))
        q = "what was I asking claude to build android"
        res = db.hybrid_search(q, E.embed(q), top_k=3)
        self.assertEqual(res[0]["app"], "Terminal")
        self.assertIn("android", res[0]["screen_text"])
        self.assertEqual(len(res), 2)  # the three identical title-only rows collapse into one
