import sys
import unittest
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aegis_context.backends import HashingEmbedder  # noqa: E402
from aegis_context.jobs import MemoryJobs  # noqa: E402
from aegis_context.retrieval import Retrieval  # noqa: E402
from aegis_context.storage import AegisStorageEngine  # noqa: E402
from fixtures.synthetic_day import NOW, TZ, build  # noqa: E402

E = HashingEmbedder(384)


class SummLLM:
    model = "fake"

    def __init__(self):
        self.prompts = []

    def complete(self, messages, tools=None):
        p = messages[-1]["content"]
        self.prompts.append(p)
        if p.startswith("Summarize this stretch"):
            title = p.split("Window: ", 1)[1].split(" |", 1)[0]
            return f"LLM summary of {title}", []
        return "LLM digest narrative", []


class JobTests(unittest.TestCase):
    def setUp(self):
        self.db = AegisStorageEngine(":memory:")
        self.eps = build(self.db, E)
        self.llm = SummLLM()

    def jobs(self, now=NOW, llm="default"):
        return MemoryJobs(self.db, E, self.llm if llm == "default" else llm, clock=lambda: now, tz=TZ)

    def drain(self, j):
        while j.summarize_episodes():
            pass

    def test_episode_summaries_become_searchable(self):
        j = self.jobs()
        self.drain(j)
        rows = self.db.conn.execute("SELECT summary_status, summary FROM episodes").fetchall()
        self.assertTrue(all(s == "done" and t.startswith("LLM summary of") for s, t in rows))
        n = self.db.conn.execute("SELECT count(*) FROM chunks WHERE kind = 'episode'").fetchone()[0]
        self.assertEqual(n, len(rows))
        r = Retrieval(self.db, E, clock=lambda: NOW, tz=TZ)
        self.assertIn("LLM summary", r.timeline(NOW - timedelta(hours=8), NOW)[0][0].activity)
        self.assertTrue(any("Typed:" in p and "android phone" in p for p in self.llm.prompts))

    def test_hourly_and_daily_digests_once(self):
        j = self.jobs()
        self.drain(j)
        made = 0
        while (n := j.hourly_digests()):
            made += n
        self.assertGreater(made, 4)
        self.assertEqual(j.hourly_digests(), 0)  # never duplicated
        self.assertEqual(j.daily_digests(), 0)    # day not over yet
        j2 = self.jobs(NOW + timedelta(days=1))
        self.assertEqual(j2.daily_digests(), 1)
        self.assertEqual(j2.daily_digests(), 0)
        r = Retrieval(self.db, E, clock=lambda: NOW + timedelta(days=1), tz=TZ)
        text, hits = r.day_summary((NOW.astimezone(TZ)).date())
        self.assertIn("LLM digest narrative", text)
        self.assertEqual(hits[0].source_id[0], "D")

    def test_no_llm_fallback(self):
        j = self.jobs(llm=None)
        self.drain(j)
        s = self.db.conn.execute(f"SELECT summary FROM episodes WHERE episode_id = {self.eps['stripe']}").fetchone()[0]
        self.assertIn("Stripe docs", s)

    def test_failing_llm_retries_then_falls_back(self):
        class Boom:
            def complete(self, *a, **k):
                raise RuntimeError("down")

        j = self.jobs(llm=Boom())
        for _ in range(4):
            j.summarize_episodes(limit=100)
        st = {r[0] for r in self.db.conn.execute("SELECT summary_status FROM episodes")}
        self.assertEqual(st, {"done"})

    def test_tiered_retention(self):
        frames = self.db.count()
        out = self.jobs(NOW + timedelta(days=8)).retention()
        self.assertGreater(out["chunks"], 0)
        self.assertEqual(self.db.count(), frames)  # metadata kept
        left = self.db.conn.execute("SELECT count(*) FROM chunks WHERE kind IN ('meta','ocr')").fetchone()[0]
        self.assertEqual(left, 0)
        self.assertEqual(self.db.conn.execute("SELECT max(length(raw_ocr_text)) FROM screen_contexts").fetchone()[0], 0)
        out = self.jobs(NOW + timedelta(days=91)).retention()
        self.assertEqual(out["episodes"], len(self.eps))
        self.assertEqual(self.db.count(), 0)
