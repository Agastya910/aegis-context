import unittest
from datetime import datetime, timedelta, timezone

from aegis_context.backends import HashingEmbedder
from aegis_context.capture import WindowInfo
from aegis_context.daemon import CaptureDaemon
from aegis_context.filters import HUD_TITLE, normalize_title, skip_reason
from aegis_context.redact import redact
from aegis_context.ringbuffer import RingBuffer
from aegis_context.storage import AegisStorageEngine
from aegis_context.timeutil import fmt_local, parse_range, parse_when

E = HashingEmbedder(384)
TZ = timezone(timedelta(hours=-7))
NOW = datetime(2026, 10, 11, 20, 30, tzinfo=timezone.utc)  # 13:30 local


def L(h, m, day=11):
    return datetime(2026, 10, day, h, m, tzinfo=TZ).astimezone(timezone.utc)


class RetentionTests(unittest.TestCase):
    def test_prune_keeps_recent_frames(self):
        db = AegisStorageEngine(":memory:")
        now = datetime.now(timezone.utc)
        db.log_screen_state("a", "A", "t", "", "", "two hours", E.embed("x"), captured_at=now - timedelta(hours=2))
        db.log_screen_state("b", "B", "t", "", "", "eight days", E.embed("y"), captured_at=now - timedelta(days=8))
        self.assertEqual(db.prune_frames(7), 1)
        self.assertEqual(db.count(), 1)


class FilterTests(unittest.TestCase):
    def test_normalize_title(self):
        a = normalize_title("✳ Aegis context Windows setup")
        self.assertEqual(a, normalize_title("◐ Aegis context Windows setup"))
        self.assertEqual(a, "Aegis context Windows setup")
        self.assertEqual(normalize_title("(3) Slack | #ml"), "Slack | #ml")

    def test_skip(self):
        self.assertTrue(skip_reason(WindowInfo("python.exe", "python", HUD_TITLE, "windows")))
        self.assertTrue(skip_reason(WindowInfo("1password.exe", "1Password", "Vault", "windows")))
        self.assertTrue(skip_reason(WindowInfo("msedge.exe", "msedge", "New tab - [InPrivate]", "windows")))
        self.assertIsNone(skip_reason(WindowInfo("code.exe", "Code", "storage.py", "windows")))

    def test_daemon_never_captures_hud(self):
        db, ring = AegisStorageEngine(":memory:"), RingBuffer()
        called = []

        class V:
            def describe(self, png):
                called.append(1)
                return "", "x"

        d = CaptureDaemon(db, ring, E, V(), lambda: b"png",
                          lambda: WindowInfo("python.exe", "python", HUD_TITLE, "windows"))
        self.assertFalse(d.tick())
        self.assertEqual((db.count(), called), (0, []))


class RedactTests(unittest.TestCase):
    def test_patterns(self):
        cases = {
            "OLLAMA_API_KEY=0123456789abcdef0123456789abcdef.FakeKeyAAAAbbbbCCCCdddd1": "0123456789ab",
            "key fedcba9876543210fedcba9876543210.FakeKeyZZZZyyyyXXXXwwww2 here": "fedcba98765",
            "> 0123456789 abcdef0123456789abcdef.FakeKeyAAAAbbbbCCCC dddd1`": "FakeKeyAAAA",  # OCR-garbled
            "sk-ant-api03-abcdefghijklmnopqrstuvwxyz": "abcdefghijkl",
            "AKIAABCDEFGHIJKLMNOP": "AKIAABCD",
            "ghp_" + "a" * 36: "aaaaaaaa",
            "password: hunter22": "hunter22",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3In0.abcdefghijklmnop": "eyJhbGci",
        }
        for text, secret in cases.items():
            self.assertNotIn(secret, redact(text), text)
            self.assertIn("[REDACTED", redact(text))

    def test_no_false_positives(self):
        for text in ["commit 3f2a9c1e8b7d6a5f4e3d2c1b0a9f8e7d6c5b4a39", "id 123e4567-e89b-12d3-a456-426614174000",
                     "Reading the FastAPI request body docs"]:
            self.assertEqual(redact(text), text)

    def test_redact_existing(self):
        db = AegisStorageEngine(":memory:")
        db.log_screen_state("a", "A", "t", "", "OLLAMA_API_KEY=0123456789abcdef0123456789abcdef.FakeKeyAAAAbbbbCCCCdddd1",
                            "env file", E.embed("x"))
        self.assertEqual(db.redact_existing(E), 1)
        ocr = db.conn.execute("SELECT raw_ocr_text FROM screen_contexts").fetchone()[0]
        self.assertNotIn("0123456789ab", ocr)


class TimeTests(unittest.TestCase):
    def w(self, s):
        return parse_when(s, NOW, TZ)

    def test_parse_when(self):
        cases = {
            "2026-10-11T13:20": L(13, 20), "2026-10-11 13:20": L(13, 20), "2026-10-11T13:20:00": L(13, 20),
            "13:20": L(13, 20), "1:20pm": L(13, 20), "3pm": L(15, 0), "9am": L(9, 0), "12am": L(0, 0),
            "now": NOW, "today": L(0, 0), "yesterday": L(0, 0, 10), "yesterday 15:00": L(15, 0, 10),
            "10 minutes ago": NOW - timedelta(minutes=10), "2 hours ago": NOW - timedelta(hours=2),
            "an hour ago": NOW - timedelta(hours=1), "45 mins ago": NOW - timedelta(minutes=45),
            "this morning": L(6, 0), "this afternoon": L(12, 0), "today 9:30": L(9, 30), "at 10:05": L(10, 5),
            "2026-10-11T20:20:00Z": datetime(2026, 10, 11, 20, 20, tzinfo=timezone.utc),
        }
        for s, want in cases.items():
            self.assertEqual(self.w(s), want, s)
        self.assertIsNone(self.w("banana"))

    def test_parse_range(self):
        s, e = parse_range("13:25", "13:26", NOW, TZ)  # tiny window widens to ±3 min
        self.assertEqual((s, e), (L(13, 25) + timedelta(seconds=30) - timedelta(minutes=3),
                                  L(13, 25) + timedelta(seconds=30) + timedelta(minutes=3)))
        s, e = parse_range("13:30", "12:30", NOW, TZ)  # swapped
        self.assertEqual((s, e), (L(12, 30), L(13, 30)))
        s, e = parse_range("12:00", "18:00", NOW, TZ)  # clamp to now
        self.assertEqual(e, NOW)
        s, e = parse_range(None, None, NOW, TZ)
        self.assertEqual((s, e), (NOW - timedelta(hours=1), NOW))

    def test_fmt_local(self):
        self.assertEqual(fmt_local(L(13, 22), tz=TZ), "13:22")
        self.assertEqual(fmt_local("2026-10-11 20:22:00", tz=TZ), "13:22")
