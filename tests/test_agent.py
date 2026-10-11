import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aegis_context.agent import ChatAgent, cited_ids  # noqa: E402
from aegis_context.backends import HashingEmbedder  # noqa: E402
from aegis_context.chat_store import ChatStore  # noqa: E402
from aegis_context.llm import FallbackChat, LLMError, LLMEvent  # noqa: E402
from aegis_context.retrieval import Retrieval  # noqa: E402
from aegis_context.storage import AegisStorageEngine  # noqa: E402
from aegis_context.tools import ToolContext, dispatch  # noqa: E402
from fixtures.synthetic_day import NOW, TZ, build  # noqa: E402

E = HashingEmbedder(384)


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


class FakeLLM:
    """Scripted LLM: each stream() call pops the next round (a list of LLMEvents, or an exception)."""

    model = "fake"

    def __init__(self, *rounds):
        self.rounds, self.requests = list(rounds), []

    def stream(self, messages, tools=None):
        self.requests.append(([dict(m) for m in messages], tools))
        r = self.rounds.pop(0) if self.rounds else [LLMEvent("token", "ok"), LLMEvent("done")]
        if isinstance(r, Exception):
            raise r
        yield from r

    def complete(self, messages, tools=None):
        return "summary", []


def tokens(*parts):
    return [LLMEvent("token", p) for p in parts] + [LLMEvent("done")]


def tool_round(*calls):
    return [LLMEvent("tool_calls", tool_calls=list(calls)), LLMEvent("done")]


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.db = AegisStorageEngine(":memory:")
        self.eps = build(self.db, E)
        self.r = Retrieval(self.db, E, clock=lambda: NOW, tz=TZ)
        self.store = ChatStore(self.db)

    def agent(self, llm):
        return ChatAgent(self.r, None, self.store, llm, clock=lambda: NOW, tz=TZ)

    def test_tool_loop_events_memo_and_followup(self):
        stripe = self.eps["stripe"]
        llm = FakeLLM(
            tool_round(call("get_activity_timeline", start="2026-10-11T10:00", end="2026-10-11T10:20")),
            tokens("At 10:05 you were reading the Stripe ", f"webhook docs [E{stripe}]."),
        )
        ag = self.agent(llm)
        events = list(ag.run(None, "what was I doing around 10:10?"))
        kinds = [k for k, _ in events]
        self.assertEqual(kinds[0], "session")
        self.assertEqual(kinds[1], "status")
        self.assertIn("10:00-10:20", events[1][1]["text"])
        self.assertEqual(kinds[-2:], ["sources", "done"])
        self.assertEqual(events[-2][1]["items"][0]["id"], f"E{stripe}")
        # the tool result reached the model, in local time
        tool_msg = llm.requests[1][0][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertIn("Webhooks | Stripe Documentation", tool_msg["content"])
        self.assertIn("10:05", tool_msg["content"])
        sid = events[0][1]["session_id"]
        msgs = self.store.get_messages(sid)
        self.assertIn("timeline 10:00-10:20", msgs[-1]["memo"])
        self.assertIn(f"cited E{stripe}", msgs[-1]["memo"])
        # follow-up turn sees the previous answer + its memo
        llm2 = FakeLLM(tokens("Before that you were searching Google."))
        list(self.agent(llm2).run(sid, "and before that?"))
        sent = llm2.requests[0][0]
        self.assertTrue(any(m["role"] == "system" and "context used: timeline 10:00-10:20" in m["content"]
                            for m in sent))
        self.assertEqual(sent[-1], {"role": "user", "content": "and before that?"})

    def test_greeting_needs_no_tools(self):
        events = list(self.agent(FakeLLM(tokens("Hi! Ask me about your day."))).run(None, "hi"))
        self.assertEqual([k for k, _ in events], ["session", "token", "done"])

    def test_system_prompt_has_now_and_coverage(self):
        sid = self.store.create_session()
        p = self.agent(FakeLLM()).system_prompt(sid)
        self.assertIn("2026-10-11 16:00", p)
        self.assertIn("UTC-07:00", p)
        self.assertIn("08:55", p)
        self.assertIn("Code", p)

    def test_offline_fallback(self):
        out = self.agent(FakeLLM(LLMError("down"))).ask("stripe webhook signature")
        self.assertTrue(out["answer"].startswith("LLM offline"))
        self.assertIn("Stripe", out["answer"])
        self.assertTrue(out["sources"])
        out = self.agent(None).ask("what did I do in the last 30 minutes")
        self.assertIn("hud.py", out["answer"])

    def test_round_limit_forces_answer(self):
        rounds = [tool_round(call("search_memory", query="x"))] * 4 + [tokens("final")]
        llm = FakeLLM(*rounds)
        out = self.agent(llm).ask("loop forever")
        self.assertEqual(out["answer"], "final")
        self.assertIsNone(llm.requests[-1][1])  # last round had no tools

    def test_sessions_persist(self):
        out = self.agent(FakeLLM(tokens("hello"))).ask("hi there")
        self.assertEqual(self.store.list_sessions()[0]["title"], "hi there")
        self.assertEqual([m["role"] for m in self.store.get_messages(out["session_id"])], ["user", "assistant"])


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = AegisStorageEngine(":memory:")
        cls.eps = build(cls.db, E)
        cls.r = Retrieval(cls.db, E, clock=lambda: NOW, tz=TZ)

    def ctx(self):
        return ToolContext(self.r, NOW, TZ)

    def test_dispatch_variants(self):
        c = self.ctx()
        res, status = dispatch("get_activity_timeline", {"start": "today", "end": "now", "limit": 1}, c)
        self.assertIn("Inbox - priya@acme.com", res)
        self.assertIn("more episodes", res)
        res, _ = dispatch("get_activity_timeline", '{"start": "2026-10-10T09:00", "end": "2026-10-10T10:00"}', c)
        self.assertIn("No captured activity", res)
        self.assertIn("Memory covers", res)
        res, _ = dispatch("search_memory", {"query": "cardiac", "app": "photoshop"}, c)
        self.assertIn("No app matching", res)
        res, _ = dispatch("search_memory", {"query": "cardiac model"}, c)
        self.assertIn("AUC 0.91", res)
        res, _ = dispatch("get_screen_text", {"source_id": f"[E{self.eps['claude']}]"}, c)
        self.assertIn("create the separate mobile app", res)
        res, _ = dispatch("get_day_summary", {"date": "today"}, c)
        self.assertIn("Outlook", res)
        res, _ = dispatch("get_activity_timeline", {"start": "today", "end": "now"}, c)
        self.assertIn("Inbox - priya", res)   # whole range visible, no truncation
        self.assertIn("hud.py", res.split("\n")[-2] + res.split("\n")[-3])
        self.assertTrue(dispatch("nope", {}, c)[0].startswith("ERROR"))
        self.assertTrue(dispatch("search_memory", "{not json", c)[0].startswith("ERROR"))
        self.assertTrue(res.startswith("<<SCREEN MEMORY"))

    def test_cited_ids(self):
        self.assertEqual(cited_ids("a [E12, F3] b [D1] c [E12] [note]"), ["E12", "F3", "D1"])


class FallbackTests(unittest.TestCase):
    def test_switches_before_first_token(self):
        fc = FallbackChat(FakeLLM(LLMError("cloud down")), FakeLLM(tokens("local answer")))
        fc.fallback.model = "qwen"
        evs = list(fc.stream([], None))
        self.assertEqual(evs[0].type, "status")
        self.assertIn("qwen", evs[0].text)
        self.assertEqual("".join(e.text for e in evs if e.type == "token"), "local answer")
