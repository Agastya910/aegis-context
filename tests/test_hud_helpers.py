import unittest

from aegis_context.agent import sse
from aegis_context.hud import linkify, parse_sse_lines, render_markdown


class HudHelperTests(unittest.TestCase):
    def test_sse_roundtrip(self):
        wire = sse("session", {"session_id": "s1"}) + sse("token", {"text": "héllo “x”"}) + sse("done", {})
        evs = list(parse_sse_lines(wire.split("\n")))
        self.assertEqual(evs, [("session", {"session_id": "s1"}), ("token", {"text": "héllo “x”"}), ("done", {})])

    def test_linkify(self):
        self.assertEqual(linkify("see [E12, F3] and [D1]; not [note]"),
                         "see [E12](aegis:E12) [F3](aegis:F3) and [D1](aegis:D1); not [note]")

    def test_render(self):
        md = render_markdown([
            {"role": "user", "text": "what was I doing?", "time": "13:35"},
            {"role": "assistant", "text": "Reading docs [E2].", "steps": ["Checking your timeline 13:20-13:30"],
             "sources": [{"id": "E2", "start_local": "13:22", "app": "chrome"}]},
        ])
        self.assertIn("> what was I doing?", md)
        self.assertIn("[E2](aegis:E2)", md)
        self.assertIn("*· Checking your timeline", md)
        self.assertIn("Sources:", md)
