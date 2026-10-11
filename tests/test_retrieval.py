import sys
import unittest
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aegis_context.backends import HashingEmbedder  # noqa: E402
from aegis_context.retrieval import Retrieval, fts_query, rrf  # noqa: E402
from aegis_context.storage import AegisStorageEngine  # noqa: E402
from fixtures.synthetic_day import NOW, TZ, at, build  # noqa: E402

E = HashingEmbedder(384)

EVAL = [
    ("budget review", "outlook"),
    ("forecast spreadsheet", "outlook"),
    ("database is locked", "sqlite_bug"),
    ("stripe webhook signature", "stripe"),
    ("constructEvent replay attacks tolerance", "stripe"),
    ("cardiac model AUC", "slack"),
    ("gradient boosting", "slack"),
    ("android mobile app", "claude"),
    ("tailscale phone", "claude"),
    ("attention is all you need", "youtube"),
    ("pad thai lunch", "lunch"),
    ("standup meeting", "standup"),
    ("QWidget focus keyboard input", "hud"),
    ("webhook handler tests", "webhooks_code"),
    ("google search sqlite fix", "search"),
]


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = AegisStorageEngine(":memory:")
        cls.eps = build(cls.db, E)
        cls.r = Retrieval(cls.db, E, clock=lambda: NOW, tz=TZ)

    def test_recall_at_5(self):
        ok = 0
        for q, name in EVAL:
            hits = self.r.search(q, k=5)
            if self.eps[name] in {h.episode_id for h in hits}:
                ok += 1
            else:
                print("MISS", q, [(h.source_id, h.title) for h in hits])
        self.assertGreaterEqual(ok / len(EVAL), 0.9)

    def test_typed_text_is_returned(self):
        h = self.r.search("what did I ask claude to build", k=3)[0]
        self.assertIn("android phone", h.typed)

    def test_time_filter(self):
        hits = self.r.search("hud.py", start=at("15:40"), end=NOW)
        self.assertTrue(hits)
        self.assertTrue(all(h.start >= at("15:40") for h in hits))

    def test_app_filter_and_aliases(self):
        self.assertTrue(self.r.resolve_apps("vscode"))
        hits = self.r.search("reading", apps=["chrome"], k=10)
        self.assertTrue(hits)
        self.assertTrue(all(h.app == "chrome" for h in hits))
        self.assertEqual(self.r.search("anything", apps=["photoshop"]), [])

    def test_timeline(self):
        rows, total = self.r.timeline(at("09:00"), at("10:20"))
        self.assertEqual([h.episode_id for h in rows][:4],
                         [self.eps[n] for n in ("outlook", "sqlite_bug", "search", "stripe")])
        first, total = self.r.timeline(at("00:00"), NOW, limit=1)
        self.assertEqual((first[0].episode_id, total), (self.eps["outlook"], len(self.eps)))
        last, _ = self.r.timeline(at("00:00"), NOW, limit=1, order="desc")
        self.assertEqual(last[0].episode_id, self.eps["hud2"])
        self.assertIn("forecast", rows[0].typed)

    def test_sources(self):
        e = self.r.source(f"E{self.eps['claude']}")
        self.assertIn("create the separate mobile app", e)
        self.assertIn("13:40", e)
        fid = self.db.conn.execute("SELECT min(context_id) FROM screen_contexts").fetchone()[0]
        self.assertIn("Q3 budget", self.r.source(f"F{fid}"))
        self.assertIn("Unknown", self.r.source("X1"))

    def test_day_summary_and_coverage(self):
        text, hits = self.r.day_summary(at("12:00").astimezone(TZ).date())
        self.assertIn("Outlook", text)
        self.assertEqual(len(hits), len(self.eps))
        text, hits = self.r.day_summary((at("12:00") - timedelta(days=1)).astimezone(TZ).date())
        self.assertEqual((text, hits), ("", []))
        self.assertEqual(self.r.coverage()[0], at("08:55"))

    def test_helpers(self):
        self.assertEqual(fts_query('what is "this": ok-go'), '"ok" OR "go"')
        self.assertIsNone(fts_query("what is this"))
        s = rrf([1, 2, 3], [3, 1])
        self.assertGreater(s[1], s[2])
        self.assertGreater(s[3], s[2])
