"""A scripted local day (UTC-07:00, 2026-10-11) with known facts, ingested through the real pipeline."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aegis_context.backends import VisionResult
from aegis_context.ingest import Ingestor

TZ = timezone(timedelta(hours=-7))
NOW = datetime(2026, 10, 11, 16, 0, tzinfo=TZ).astimezone(timezone.utc)

# name: (app, bundle, title, [(HH:MM, activity, typed, entities, text), ...])
DAY = {
    "outlook": ("Outlook", "olk.exe", "Inbox - priya@acme.com - Outlook", [
        ("08:55", "Reading an email from Priya Sharma about the Q3 budget review", "", "Priya Sharma, Q3 budget",
         "From: Priya Sharma\nSubject: Q3 budget review moved\nHi, the Q3 budget review has been moved to Thursday "
         "2pm in room 4B. Please bring the forecast spreadsheet."),
        ("09:04", "Replying to Priya's email about the Q3 budget review", "Thanks Priya, I'll bring the forecast.",
         "Priya Sharma", "Reply: Thanks Priya, I'll bring the forecast.\nOn Sat, Priya Sharma wrote: the Q3 budget "
         "review has been moved to Thursday 2pm"),
    ]),
    "sqlite_bug": ("Code", "code.exe", "storage.py - aegis-context - Visual Studio Code", [
        ("09:10", "Debugging a 'database is locked' error in storage.py", "", "storage.py, sqlite3",
         "Traceback (most recent call last):\n  File \"storage.py\", line 120, in log_screen_state\n"
         "sqlite3.OperationalError: database is locked"),
        ("09:22", "Adding a threading lock around SQLite writes in storage.py", "", "storage.py",
         "class AegisStorageEngine:\n    def __init__(self):\n        self._lock = threading.RLock()\n"
         "        self.conn.execute('PRAGMA journal_mode = WAL')"),
        ("09:35", "Running pytest; storage tests now pass", "", "pytest", "===== 14 passed in 0.9s ====="),
    ]),
    "search": ("chrome", "chrome.exe", "sqlite database is locked python - Google Search", [
        ("09:40", "Searching Google for how to fix sqlite 'database is locked' in Python", "", "Google",
         "sqlite3.OperationalError: database is locked - Stack Overflow\nUse WAL mode and a single writer"),
    ]),
    "stripe": ("chrome", "chrome.exe", "Webhooks | Stripe Documentation", [
        ("10:05", "Reading the Stripe docs on verifying webhook signatures", "", "Stripe, stripe.com",
         "Verify webhook signatures\nStripe signs the webhook events it sends to your endpoints by including a "
         "signature in each event's Stripe-Signature header. Use stripe.Webhook.construct_event (constructEvent)."),
        ("10:15", "Reading about webhook signature replay attacks and tolerance in the Stripe docs", "",
         "Stripe", "Preventing replay attacks: the timestamp is included in the signed payload, default tolerance "
         "of five minutes"),
    ]),
    "webhooks_code": ("Code", "code.exe", "webhooks.py - payments - Visual Studio Code", [
        ("10:30", "Implementing Stripe webhook signature verification in webhooks.py", "", "webhooks.py",
         "def handle(request):\n    event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)"),
        ("10:55", "Writing tests for the webhook handler", "", "test_webhooks.py",
         "def test_bad_signature_rejected():\n    assert handle(bad).status_code == 400"),
    ]),
    "slack": ("slack", "slack.exe", "ml-team (Channel) - Acme - Slack", [
        ("11:30", "Reading #ml-team in Slack: Dana posted cardiac model results", "", "Dana, cardiac model",
         "Dana: cardiac model AUC 0.91 on the holdout set, recall 0.84 at threshold 0.5"),
        ("11:40", "Replying to Dana in #ml-team about the cardiac model", "nice! can we try gradient boosting next?",
         "Dana", "Dana: cardiac model AUC 0.91 on the holdout set\nYou: nice! can we try gradient boosting next?"),
    ]),
    "lunch": ("chrome", "chrome.exe", "DoorDash - Thai Basil", [
        ("12:15", "Ordering lunch on DoorDash: Pad Thai from Thai Basil", "", "DoorDash, Thai Basil",
         "Thai Basil\nPad Thai (chicken) $14.50\nCheckout - deliver to 500 Market St"),
    ]),
    "claude": ("WindowsTerminal", "windowsterminal.exe", "Claude Code", [
        ("13:40", "Asking Claude Code to build an Android app for the aegis project",
         "create the separate mobile app now that i can download in my android phone", "Claude Code, aegis",
         "> create the separate mobile app now that i can download in my android phone\n"
         "Claude: I can build the phone side one of two ways"),
        ("13:48", "Reading Claude's answer about a Tailscale phone-to-laptop design", "", "Tailscale",
         "Phone to laptop: an HTTP shortcut that calls the laptop over Tailscale, with a token"),
    ]),
    "hud": ("Code", "code.exe", "hud.py - aegis-context - Visual Studio Code", [
        ("14:10", "Fixing the HUD focus bug in hud.py (window not receiving keyboard input)", "", "hud.py, PyQt5",
         "class HUD(QWidget):\n    def toggle(self):\n        _force_foreground(int(self.winId()))"),
        ("14:25", "Adding a scrollable QTextBrowser chat transcript to the HUD", "", "hud.py",
         "self.chat = QTextBrowser()\nself.chat.verticalScrollBar()"),
    ]),
    "standup": ("ms-teams", "ms-teams.exe", "Standup | Microsoft Teams", [
        ("14:45", "In the daily standup meeting on Teams with Priya and Dana", "", "Priya, Dana, Teams",
         "Daily standup - Priya Sharma, Dana Lee - 12 min\nBlocked: waiting on budget numbers"),
    ]),
    "youtube": ("chrome", "chrome.exe", "Attention Is All You Need explained - YouTube", [
        ("15:20", "Watching a YouTube video explaining the 'Attention Is All You Need' transformer paper", "",
         "YouTube, transformers", "Attention Is All You Need explained\nmulti-head self-attention, positional "
         "encoding\n1.2M views"),
    ]),
    "hud2": ("Code", "code.exe", "hud.py - aegis-context - Visual Studio Code", [
        ("15:50", "Styling the HUD chat bubbles in hud.py", "", "hud.py",
         "QTextBrowser{background:rgb(26,26,38)}"),
    ]),
}


def at(hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime(2026, 10, 11, h, m, tzinfo=TZ).astimezone(timezone.utc)


def build(storage, embedder, ring=None) -> dict[str, int]:
    """Ingest the day; returns {name: episode_id}."""
    ing = Ingestor(storage, embedder, ring)
    eps: dict[str, int] = {}
    for name, (app, bundle, title, frames) in DAY.items():
        prev = None
        for hhmm, act, typed, ent, text in frames:
            t = at(hhmm)
            while prev and t - prev > timedelta(minutes=2):  # the real daemon heartbeats while the screen is steady
                prev += timedelta(minutes=2)
                ing.heartbeat(prev)
            ing.ingest(bundle, app, "windows", title, VisionResult(act, typed, ent, text), t)
            prev = t
        eps[name] = ing.cur["id"]
    storage.close_open_episodes()
    return eps
