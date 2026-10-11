import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient  # noqa: E402

from aegis_context.agent import ChatAgent  # noqa: E402
from aegis_context.backends import HashingEmbedder  # noqa: E402
from aegis_context.chat_store import ChatStore  # noqa: E402
from aegis_context.darwin import DarwinClient, format_results  # noqa: E402
from aegis_context.insights import categorize, time_report  # noqa: E402
from aegis_context.ipc_server import BridgeState, create_app  # noqa: E402
from aegis_context.retrieval import Retrieval  # noqa: E402
from aegis_context.ringbuffer import RingBuffer  # noqa: E402
from aegis_context.safe import build_pack, scrub  # noqa: E402
from aegis_context.storage import AegisStorageEngine  # noqa: E402
from aegis_context.tools import ToolContext, dispatch  # noqa: E402
from fixtures.synthetic_day import NOW, TZ, at, build  # noqa: E402

E = HashingEmbedder(384)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = AegisStorageEngine(":memory:")
        cls.eps = build(cls.db, E)
        cls.r = Retrieval(cls.db, E, clock=lambda: NOW, tz=TZ)


class InsightTests(Fixture):
    def test_categorize(self):
        self.assertEqual(categorize("chrome", "Attention Is All You Need explained - YouTube"), "Entertainment")
        self.assertEqual(categorize("chrome", "Inbox (3) - me@x.com - Gmail"), "Communication")
        self.assertEqual(categorize("Code", "hud.py - aegis"), "Coding")
        self.assertEqual(categorize("chrome", "Webhooks | Stripe Documentation"), "Research & browsing")
        self.assertEqual(categorize("ms-teams", "Standup | Microsoft Teams"), "Meetings")
        self.assertEqual(categorize("Notepad", "notes.txt"), "Docs & notes")

    def test_report(self):
        rep = time_report(self.r, at("00:00"), NOW)
        self.assertGreater(rep.tracked_min, 60)
        self.assertIn("Coding", rep.by_category)
        self.assertGreater(rep.switches, 5)
        self.assertGreater(rep.refocus_cost_min, 0)
        self.assertIsNotNone(rep.longest_block)
        self.assertGreaterEqual(rep.by_category.get("Entertainment", 0), 1)
        self.assertTrue(rep.tips)
        self.assertEqual(time_report(self.r, at("05:00"), at("06:00")).tracked_min, 0)


class SafeTests(Fixture):
    def test_scrub(self):
        text, counts = scrub("mail priya@acme.com or call +1 415-555-0134, card 4111 1111 1111 1111, "
                             "https://x.com/reset?token=abc OLLAMA_API_KEY=0123456789abcdef0123456789abcdef.FakeKeyAAAAbbbbCCCCdddd1")
        for leaked in ("priya@acme.com", "555-0134", "4111", "token=abc", "FakeKeyAAAA"):
            self.assertNotIn(leaked, text)
        self.assertIn("https://x.com/reset", text)
        self.assertTrue({"email", "phone", "card", "secret"} <= set(counts))

    def test_pack_withholds_private_categories(self):
        pack = build_pack(self.r, minutes=8 * 60, now=NOW)
        self.assertIn("Communication", pack.withheld)          # Outlook + Slack content withheld
        self.assertNotIn("Priya", pack.text)                   # email content never shared
        self.assertNotIn("AUC 0.91", pack.text)                # chat content never shared
        self.assertNotIn("priya@acme.com", pack.text)
        self.assertIn("hud.py", pack.text)                     # work context is shared
        self.assertIn("sanitized locally", pack.text)
        focused = build_pack(self.r, minutes=8 * 60, focus="stripe webhook", now=NOW)
        self.assertTrue(all("stripe" in ln.lower() or "webhook" in ln.lower()
                            for ln in focused.text.splitlines() if ln.startswith("- ")))


class FakeResp:
    def __init__(self, status, data, headers=None):
        self.status_code, self._data, self.headers = status, data, headers or {}

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, resp):
        self.resp, self.sent = resp, []

    def post(self, url, json=None, timeout=None):
        self.sent.append((url, json))
        return self.resp


DARWIN_OK = {"contractVersion": "search-public-v2", "outcome": "MATCH", "agents": [], "versions": {}, "results": [
    {"rank": 1, "name": "Plan interests in lanes", "description": "Plans your week across interests.",
     "readiness": "ready", "canStartThread": True, "agent": "a1", "capability": "c1"}]}


class DarwinTests(unittest.TestCase):
    def test_only_scrubbed_need_leaves(self):
        sess = FakeSession(FakeResp(200, DARWIN_OK))
        q, res = DarwinClient(session=sess).search("plan my week, email me at me@acme.com " + "x" * 400)
        url, body = sess.sent[0]
        self.assertTrue(url.endswith("/search"))
        self.assertNotIn("me@acme.com", body["query"])
        self.assertLessEqual(len(body["query"]), 300)
        self.assertEqual(set(body), {"query", "numResults"})
        self.assertEqual(res[0]["name"], "Plan interests in lanes")
        self.assertIn("only this query", format_results(q, res))

    def test_rate_limit(self):
        from aegis_context.darwin import DarwinError
        with self.assertRaises(DarwinError):
            DarwinClient(session=FakeSession(FakeResp(429, {}, {"Retry-After": "5"}))).search("x")


class ToolAndApiTests(Fixture):
    def test_new_tools(self):
        ctx = ToolContext(self.r, NOW, TZ, darwin=DarwinClient(session=FakeSession(FakeResp(200, DARWIN_OK))))
        res, st = dispatch("get_time_report", {}, ctx)
        self.assertIn("Context switches", res)
        self.assertIn("Measuring your time", st)
        res, st = dispatch("get_shareable_context", {"minutes": 480}, ctx)
        self.assertIn("SANITIZED PACK", res)
        self.assertNotIn("Priya", res)
        res, st = dispatch("find_agents", {"need": "plan my week"}, ctx)
        self.assertIn("Plan interests in lanes", res)
        self.assertIn("only this text leaves", st)
        res, _ = dispatch("find_agents", {"need": "x"}, ToolContext(self.r, NOW, TZ))
        self.assertIn("disabled", res)

    def test_endpoints(self):
        agent = ChatAgent(self.r, RingBuffer(), ChatStore(self.db), None, clock=lambda: NOW, tz=TZ)
        c = TestClient(create_app(BridgeState(self.db, RingBuffer(), E, agent=agent)))
        j = c.get("/context/safe?minutes=60").json()
        self.assertIn("text", j)
        j = c.get("/insights?hours=24").json()
        self.assertIn("tips", j)


class ContentSensitivityTests(unittest.TestCase):
    def test_signup_and_personal_docs_withheld_and_private_terms(self):
        from aegis_context.backends import VisionResult
        from aegis_context.ingest import Ingestor
        db = AegisStorageEngine(":memory:")
        ing = Ingestor(db, E)
        ing.ingest("chrome.exe", "chrome", "windows", "Alike - Google Chrome",
                   VisionResult("Creating a new account; entered the name Jane Roe and a password", "", "", "x"), at("15:00"))
        ing.ingest("explorer.exe", "explorer", "windows", "Downloads",
                   VisionResult("Browsing files including USvisaJaneRoe.pdf and a passport scan", "", "", "y"), at("15:10"))
        ing.ingest("code.exe", "Code", "windows", "main.py - Jane Roe's project",
                   VisionResult("Editing main.py for Jane Roe's project", "", "", "z"), at("15:20"))
        db.close_open_episodes()
        r = Retrieval(db, E, clock=lambda: NOW, tz=TZ)
        pack = build_pack(r, 120, now=NOW, private_terms=["Jane Roe"])
        self.assertNotIn("Jane", pack.text)
        self.assertNotIn("passport", pack.text)
        self.assertIn("main.py", pack.text)
        self.assertIn("Accounts & security", pack.withheld)
        self.assertIn("Personal documents", pack.withheld)
