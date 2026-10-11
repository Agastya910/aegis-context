"""Time insights: where your time went, how fragmented it was, and when you focus best.

All computed locally from activity episodes. The model (documented in docs/ARCHITECTURE.md):
- an episode's attention time = min(next episode start - its start, its own span + 2 min, 30 min); bigger gaps = away
- a *focus block* = consecutive episodes in the same category with < 2 min between them
- *deep work* = focus blocks >= 25 min (one classic pomodoro); focus ratio = deep-work time / tracked time
- *context switches* = transitions between different categories; switch rate = switches per tracked hour
- *refocus cost* = sum over switches of min(`refocus_min`, time spent after the switch): default 2 min, a
  conservative attention-residue estimate that can never exceed the time actually tracked
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .retrieval import Retrieval
from .timeutil import fmt_local, to_local

# (category, app regex, title regex) — first match wins; title-only rules come first so a browser tab on YouTube
# counts as entertainment, not "browsing".
CATEGORY_RULES: list[tuple[str, str, str]] = [
    ("Accounts & security", r"1password|bitwarden|keepass",
     r"sign ?up|log ?in|sign ?in|password|passkey|2fa|two-factor|verification code|manage access|/settings/access|"
     r"account settings|security settings|reset your"),
    ("Meetings", r"zoom|ms-teams|teams|webex|meet", r"meet\.google|zoom meeting|standup|\bcall\b|huddle"),
    ("Entertainment", r"spotify|netflix|steam|vlc", r"youtube|netflix|twitch|prime video|reddit|9gag|tiktok"),
    ("Social", r"instagram|facebook|twitter|x\.com", r"instagram|facebook|twitter|linkedin|/ x$| on x\b"),
    ("Communication", r"outlook|olk|thunderbird|slack|discord|whatsapp|telegram|signal|messages|mail",
     r"gmail|inbox|outlook|slack|whatsapp|discord|messenger|mail\b"),
    ("Finance & personal admin", r"bank|paypal|venmo", r"bank|paypal|venmo|invoice|tax|doordash|uber eats|amazon\.|checkout"),
    ("AI assistants", r"^claude$|chatgpt|copilot|cursor", r"claude|chatgpt|gemini|copilot|perplexity"),
    ("Coding", r"^code$|code - insiders|pycharm|idea|webstorm|windowsterminal|powershell|cmd|terminal|wt|devenv|vim",
     r"\.py\b|\.ts\b|\.js\b|github|gitlab|stack overflow|stackoverflow|localhost|pull request"),
    ("Docs & notes", r"notepad|obsidian|notion|onenote|winword|word|excel|powerpnt|acrobat|acrord32",
     r"google docs|google sheets|notion|\.pdf\b|confluence|docs\b"),
    ("Research & browsing", r"chrome|msedge|firefox|brave|safari|opera|arc", r".*"),
]
# never shared in context packs beyond "<category>, N min": conversations name people
SENSITIVE_CATEGORIES = {"Communication", "Meetings", "Social", "Finance & personal admin", "Accounts & security"}
DISTRACTION_CATEGORIES = {"Entertainment", "Social"}


def categorize(app: str, title: str) -> str:
    a, t = (app or "").lower(), (title or "").lower()
    for cat, app_re, title_re in CATEGORY_RULES:  # title signals first (a YouTube tab is entertainment)
        if cat != "Research & browsing" and title_re and re.search(title_re, t):
            return cat
    for cat, app_re, _ in CATEGORY_RULES:
        if re.search(app_re, a):
            return cat
    return "Other"


@dataclass
class Block:
    category: str
    start: datetime
    end: datetime
    minutes: float
    apps: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)


@dataclass
class TimeReport:
    start: datetime
    end: datetime
    tracked_min: float = 0.0
    by_category: dict[str, float] = field(default_factory=dict)
    by_app: dict[str, float] = field(default_factory=dict)
    top_titles: list[tuple[str, float]] = field(default_factory=list)
    switches: int = 0
    switches_per_hour: float = 0.0
    refocus_cost_min: float = 0.0
    deep_blocks: list[Block] = field(default_factory=list)
    longest_block: Block | None = None
    focus_ratio: float = 0.0
    distraction_min: float = 0.0
    best_focus_hour: str | None = None
    tips: list[str] = field(default_factory=list)


def time_report(r: Retrieval, start: datetime, end: datetime, refocus_min: float = 2.0, deep_min: float = 25.0
                ) -> TimeReport:
    rows, _ = r.timeline(start, end, limit=5000)
    rep = TimeReport(start, end)
    if not rows:
        return rep
    rows.sort(key=lambda h: h.start)
    spans: list[tuple] = []  # (category, start, minutes, hit)
    for i, h in enumerate(rows):
        nxt = rows[i + 1].start if i + 1 < len(rows) else min(end, h.end + timedelta(minutes=2))
        own = (h.end - h.start) + timedelta(minutes=2)
        mins = max(0.5, min((nxt - h.start), own, timedelta(minutes=30)).total_seconds() / 60)
        spans.append((categorize(h.app, h.title), h.start, mins, h))
    cat_t: dict[str, float] = defaultdict(float)
    app_t: dict[str, float] = defaultdict(float)
    title_t: dict[str, float] = defaultdict(float)
    hour_focus: dict[str, float] = defaultdict(float)
    blocks: list[Block] = []
    prev_cat = None
    refocus = 0.0
    for cat, s, mins, h in spans:
        cat_t[cat] += mins
        app_t[h.app] += mins
        if h.title:
            title_t[f"{h.app} | {h.title}"] += mins
        b = blocks[-1] if blocks else None
        if b and b.category == cat and (s - b.end) <= timedelta(minutes=2):
            b.end = max(b.end, s + timedelta(minutes=mins))
            b.minutes += mins
        else:
            blocks.append(Block(cat, s, s + timedelta(minutes=mins), mins))
            b = blocks[-1]
        if h.app not in b.apps:
            b.apps.append(h.app)
        if h.title and h.title not in b.titles:
            b.titles.append(h.title)
        b.ids.append(h.source_id)
        if prev_cat is not None and cat != prev_cat:
            rep.switches += 1
            refocus += min(refocus_min, mins)
        prev_cat = cat
    rep.tracked_min = sum(cat_t.values())
    rep.by_category = dict(sorted(cat_t.items(), key=lambda kv: -kv[1]))
    rep.by_app = dict(sorted(app_t.items(), key=lambda kv: -kv[1])[:8])
    rep.top_titles = sorted(title_t.items(), key=lambda kv: -kv[1])[:6]
    hours = rep.tracked_min / 60
    rep.switches_per_hour = rep.switches / hours if hours > 0 else 0.0
    rep.refocus_cost_min = refocus
    productive = [b for b in blocks if b.category not in DISTRACTION_CATEGORIES | {"Communication", "Social"}]
    rep.deep_blocks = sorted([b for b in productive if b.minutes >= deep_min], key=lambda b: -b.minutes)
    rep.longest_block = max(productive, key=lambda b: b.minutes) if productive else None
    rep.focus_ratio = sum(b.minutes for b in rep.deep_blocks) / rep.tracked_min if rep.tracked_min else 0.0
    rep.distraction_min = sum(cat_t[c] for c in DISTRACTION_CATEGORIES)
    for b in productive:
        hour_focus[to_local(b.start, r.tz).strftime("%H:00")] += b.minutes
    rep.best_focus_hour = max(hour_focus, key=hour_focus.get) if hour_focus else None
    rep.tips = _tips(rep)
    return rep


def _tips(rep: TimeReport) -> list[str]:
    tips = []
    comm = rep.by_category.get("Communication", 0)
    if rep.switches_per_hour > 12:
        tips.append(f"You switched context {rep.switches_per_hour:.0f}×/hour (up to ≈{rep.refocus_cost_min:.0f} min lost to "
                    "refocusing). Batch email/chat into 2–3 fixed slots and close them in between.")
    if rep.longest_block and rep.longest_block.minutes < 25:
        tips.append(f"Your longest uninterrupted stretch was {rep.longest_block.minutes:.0f} min. Protect one 50-min block"
                    + (f" around {rep.best_focus_hour}, your most focused hour." if rep.best_focus_hour else "."))
    elif rep.deep_blocks:
        tips.append(f"You had {len(rep.deep_blocks)} deep-work block(s); the best started at "
                    f"{fmt_local(rep.deep_blocks[0].start)}. Schedule your hardest task in that slot again.")
    if rep.tracked_min and comm / rep.tracked_min > 0.25:
        tips.append(f"Communication took {comm / rep.tracked_min:.0%} of your time. Turn off notifications during focus "
                    "blocks and reply in batches.")
    if rep.distraction_min >= 15:
        tips.append(f"{rep.distraction_min:.0f} min went to entertainment/social. Fine as a break — keep it after a "
                    "focus block rather than in the middle of one.")
    ai = rep.by_category.get("AI assistants", 0)
    if ai >= 15:
        tips.append(f"{ai:.0f} min with AI assistants: paste an Aegis context pack instead of re-explaining your setup "
                    "each time.")
    return tips


def format_report(rep: TimeReport, tz=None) -> str:
    if not rep.tracked_min:
        return f"No tracked activity between {fmt_local(rep.start, True, tz)} and {fmt_local(rep.end, True, tz)}."
    def m(x):
        return f"{x / 60:.1f} h" if x >= 90 else f"{x:.0f} min"
    lines = [f"Time report {fmt_local(rep.start, True, tz)} – {fmt_local(rep.end, False, tz)} · tracked {m(rep.tracked_min)}",
             "By category: " + ", ".join(f"{c} {m(v)} ({v / rep.tracked_min:.0%})" for c, v in rep.by_category.items()),
             "Top apps: " + ", ".join(f"{a} {m(v)}" for a, v in rep.by_app.items()),
             "Top windows: " + "; ".join(f"{t} ({m(v)})" for t, v in rep.top_titles),
             f"Context switches: {rep.switches} ({rep.switches_per_hour:.1f}/h) · est. refocus cost {m(rep.refocus_cost_min)}",
             f"Deep-work blocks (≥25 min): {len(rep.deep_blocks)} · focus ratio {rep.focus_ratio:.0%}"]
    for b in rep.deep_blocks[:3]:
        lines.append(f"  - {fmt_local(b.start, False, tz)}–{fmt_local(b.end, False, tz)} {b.category} ({m(b.minutes)}): "
                     f"{', '.join(b.titles[:2])} [{', '.join(b.ids[:3])}]")
    if rep.longest_block:
        b = rep.longest_block
        lines.append(f"Longest focus: {m(b.minutes)} of {b.category} from {fmt_local(b.start, False, tz)} [{b.ids[0]}]")
    if rep.best_focus_hour:
        lines.append(f"Most focused hour: {rep.best_focus_hour}")
    lines.append(f"Entertainment/social: {m(rep.distraction_min)}")
    if rep.tips:
        lines.append("Data-backed suggestions:")
        lines += [f"  - {t}" for t in rep.tips]
    return "\n".join(lines)


def report_dict(rep: TimeReport, tz=None) -> dict:
    return {
        "start_local": fmt_local(rep.start, True, tz), "end_local": fmt_local(rep.end, True, tz),
        "tracked_min": round(rep.tracked_min, 1), "by_category": {k: round(v, 1) for k, v in rep.by_category.items()},
        "by_app": {k: round(v, 1) for k, v in rep.by_app.items()}, "switches": rep.switches,
        "switches_per_hour": round(rep.switches_per_hour, 1), "refocus_cost_min": round(rep.refocus_cost_min, 1),
        "deep_blocks": [{"category": b.category, "start_local": fmt_local(b.start, False, tz),
                         "end_local": fmt_local(b.end, False, tz), "minutes": round(b.minutes, 1), "ids": b.ids[:5]}
                        for b in rep.deep_blocks],
        "focus_ratio": round(rep.focus_ratio, 3), "distraction_min": round(rep.distraction_min, 1),
        "best_focus_hour": rep.best_focus_hour, "tips": rep.tips, "text": format_report(rep, tz),
    }
