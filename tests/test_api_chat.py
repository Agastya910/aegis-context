import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient  # noqa: E402

from aegis_context.agent import ChatAgent  # noqa: E402
from aegis_context.backends import HashingEmbedder  # noqa: E402
from aegis_context.chat_store import ChatStore  # noqa: E402
from aegis_context.ipc_server import BridgeState, create_app  # noqa: E402
from aegis_context.retrieval import Retrieval  # noqa: E402
from aegis_context.ringbuffer import RingBuffer  # noqa: E402
from aegis_context.storage import AegisStorageEngine  # noqa: E402
from fixtures.synthetic_day import NOW, TZ, build  # noqa: E402
from test_agent import FakeLLM, call, tokens, tool_round  # noqa: E402

E = HashingEmbedder(384)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        ev = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        out.append((ev, data))
    return out


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        db = AegisStorageEngine(":memory:")
        self.eps = build(db, E)
        self.llm = FakeLLM()
        r = Retrieval(db, E, clock=lambda: NOW, tz=TZ)
        agent = ChatAgent(r, RingBuffer(), ChatStore(db), self.llm, clock=lambda: NOW, tz=TZ)
        self.c = TestClient(create_app(BridgeState(db, RingBuffer(), E, agent=agent)))

    def test_chat_stream_and_sessions(self):
        e = self.eps["youtube"]
        self.llm.rounds = [tool_round(call("search_memory", query="youtube video")),
                           tokens("You watched “Attention Is All You Need explained” ", f"[E{e}].")]
        r = self.c.post("/chat", json={"message": "what did I watch on youtube?"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        evs = parse_sse(r.text)
        kinds = [k for k, _ in evs]
        self.assertEqual(kinds[0], "session")
        self.assertIn("status", kinds)
        self.assertEqual(kinds[-1], "done")
        answer = "".join(d["text"] for k, d in evs if k == "token")
        self.assertIn("Attention Is All You Need", answer)  # unicode survives the stream
        sid = evs[0][1]["session_id"]
        self.assertEqual(self.c.get("/chat/sessions").json()[0]["session_id"], sid)
        msgs = self.c.get(f"/chat/sessions/{sid}").json()["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        new = self.c.post("/chat/sessions").json()["session_id"]
        self.assertNotEqual(new, sid)
        self.c.delete(f"/chat/sessions/{new}")
        self.assertEqual(self.c.get(f"/chat/sessions/{new}").status_code, 404)

    def test_memory_endpoints(self):
        src = self.c.get(f"/memory/source/E{self.eps['slack']}").json()
        self.assertIn("gradient boosting", src["text"])
        self.assertIn("items", self.c.get("/memory/timeline?minutes=60").json())

    def test_context_query_backcompat_with_agent(self):
        self.llm.rounds = [tokens(f"Dana posted AUC 0.91 [E{self.eps['slack']}].")]
        j = self.c.post("/context/query", json={"query_text": "cardiac model"}).json()
        for key in ("status", "matches", "digests", "synthesized", "answer"):
            self.assertIn(key, j)
        self.assertEqual(j["matches"][0]["app"], "slack")
        self.assertEqual(self.c.get("/chat/sessions").json(), [])  # one-shot query leaves no session

    def test_validation_and_health(self):
        self.assertEqual(self.c.post("/chat", json={"message": ""}).status_code, 422)
        h = self.c.get("/health").json()
        self.assertGreater(h["episodes"], 0)


class NoAgentTests(unittest.TestCase):
    def test_chat_needs_agent(self):
        c = TestClient(create_app(BridgeState(AegisStorageEngine(":memory:"), RingBuffer(), E)))
        self.assertEqual(c.post("/chat", json={"message": "hi"}).status_code, 503)
