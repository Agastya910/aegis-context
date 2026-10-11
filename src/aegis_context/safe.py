"""Privacy firewall for anything that leaves Aegis: clipboard context packs and outbound queries (e.g. Darwin).

Three layers, all local:
1. Secret redaction (redact.py): API keys, tokens, passwords, private keys.
2. PII scrubbing: emails, phone numbers, card numbers, IPs, long account-like numbers, URL query strings.
3. Sensitive-category withholding: email, chat, meetings, social, finance and account/sign-in activity is reduced to
   "<category>, N min" — its content (who wrote what) never goes into a pack.
Raw screen text (OCR) is never included in a pack; only activity summaries and window titles.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .insights import SENSITIVE_CATEGORIES, categorize
from .redact import redact
from .retrieval import Retrieval
from .timeutil import fmt_local, utc_now

_PII: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("phone", re.compile(r"(?<![\w.])(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})[ .-]?\d{3,4}[ .-]?\d{3,4}(?![\w.])")),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("number", re.compile(r"\b\d{7,}\b")),
]
# content-level sensitivity (the window title alone can look harmless, e.g. a signup form on a normal site)
_CONTENT_RULES = [
    ("Accounts & security", re.compile(r"(?i)sign(ing)?[ -]?up|creat\w* (a |an )?(new )?account|log(ging)?[ -]?in\b|"
                                       r"sign(ing)?[ -]?in\b|password|passkey|verification code|two-factor|2fa|"
                                       r"api key|access token|credentials")),
    ("Personal documents", re.compile(r"(?i)visa|passport|social security|\bssn\b|tax (return|form)|\bw-?2\b|"
                                      r"medical|diagnos|prescription|bank statement|salary|payslip|pay stub|"
                                      r"offer letter|r[ée]sum[ée]|\bcv\b|immigration|lease agreement")),
]
_URL_QUERY = re.compile(r"(https?://[^\s?#]+)[?#][^\s]*")
_INBOX_COUNT = re.compile(r"\(\d[\d,]*\)")


def scrub(text: str) -> tuple[str, dict[str, int]]:
    """Redact secrets + PII. Returns (clean text, counts by kind)."""
    counts: dict[str, int] = {}
    if not text:
        return text, counts
    before = text
    text = redact(text)
    if text != before:
        counts["secret"] = text.count("[REDACTED")
    text = _URL_QUERY.sub(r"\1", text)
    for kind, pat in _PII:
        text, n = pat.subn(f"[{kind}]", text)
        if n:
            counts[kind] = counts.get(kind, 0) + n
    return text, counts


@dataclass
class SafePack:
    text: str
    included: int = 0
    withheld: dict[str, float] = field(default_factory=dict)  # category -> minutes
    scrubbed: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"text": self.text, "included": self.included, "withheld": self.withheld, "scrubbed": self.scrubbed}


def build_pack(r: Retrieval, minutes: int = 30, focus: str | None = None, now: datetime | None = None,
               max_items: int = 15, private_terms: list[str] | None = None) -> SafePack:
    """A paste-ready, sanitized summary of recent activity (for an AI assistant, a teammate, a standup)."""
    now = now or r.clock() or utc_now()
    start = now - timedelta(minutes=max(1, minutes))
    rows, _ = r.timeline(start, now, limit=500)
    pack = SafePack("")
    lines: list[str] = []
    focus_l = (focus or "").lower().strip()
    for h in rows:
        cat = categorize(h.app, h.title)
        mins = max(1.0, (h.end - h.start).total_seconds() / 60)
        if cat not in SENSITIVE_CATEGORIES:
            for rule_cat, pat in _CONTENT_RULES:
                if pat.search(f"{h.title} {h.activity} {h.typed}"):
                    cat = rule_cat
                    break
        if cat in SENSITIVE_CATEGORIES or cat == "Personal documents":
            pack.withheld[cat] = pack.withheld.get(cat, 0) + mins
            continue
        title, c1 = scrub(_INBOX_COUNT.sub("", h.title or ""))
        act, c2 = scrub(h.activity or "")
        blob = f"{h.app} {title} {act}".lower()
        if focus_l and not any(w in blob for w in focus_l.split()):
            continue
        for c in (c1, c2):
            for k, v in c.items():
                pack.scrubbed[k] = pack.scrubbed.get(k, 0) + v
        for term in private_terms or []:
            if term.strip():
                pat = re.compile(re.escape(term.strip()), re.I)
                title, n1 = pat.subn("[private]", title)
                act, n2 = pat.subn("[private]", act)
                if n1 + n2:
                    pack.scrubbed["private term"] = pack.scrubbed.get("private term", 0) + n1 + n2
        act = " ".join(act.split())
        if len(act) > 200:
            act = act[:200] + "…"
        span = fmt_local(h.start, tz=r.tz) + (f"–{fmt_local(h.end, tz=r.tz)}" if mins >= 2 else "")
        line = f"- {span} · {h.app} · {title.strip()}: {act}"
        if lines and lines[-1].split(": ", 1)[-1] == act:  # collapse repeats
            continue
        lines.append(line)
    if len(lines) > max_items:
        lines = lines[-max_items:]
    pack.included = len(lines)
    head = f"## My recent context (last {minutes} min, shared from Aegis; sanitized locally)"
    body = "\n".join(lines) if lines else "- (no shareable activity in this period)"
    foot = []
    if pack.withheld:
        foot.append("Withheld (private): " + ", ".join(f"{c} {m:.0f} min" for c, m in pack.withheld.items()))
    if pack.scrubbed:
        foot.append("Scrubbed: " + ", ".join(f"{n} {k}" for k, n in pack.scrubbed.items()))
    pack.text = "\n".join([head, body] + ([""] + foot if foot else []))
    return pack
