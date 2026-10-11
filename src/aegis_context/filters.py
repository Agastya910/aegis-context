"""Which foreground windows never get captured, and title normalization for change detection."""
from __future__ import annotations

import os
import re

HUD_TITLE = "Aegis HUD"

DEFAULT_EXCLUDE_APPS = ["1password", "bitwarden", "keepass", "keepassxc", "lockapp", "logonui"]
DEFAULT_EXCLUDE_TITLES = ["(?i)inprivate", "(?i)incognito", "(?i)private browsing"]

_LEAD = re.compile(r"^[^\w(\[]+")
_COUNTER = re.compile(r"^\(\d+\)\s*")


def normalize_title(title: str) -> str:
    t = (title or "").strip()
    for _ in range(3):
        t = _COUNTER.sub("", _LEAD.sub("", t)).strip()
    return " ".join(t.split())


def skip_reason(win, cfg=None) -> str | None:
    title = win.title or ""
    if title.startswith(HUD_TITLE):
        return "aegis hud"
    if getattr(win, "pid", 0) and win.pid == os.getpid():
        return "aegis process"
    stem = (win.bundle_id or win.app_name or "").lower().removesuffix(".exe")
    apps = getattr(cfg, "exclude_apps", None) or DEFAULT_EXCLUDE_APPS
    if stem in {a.lower() for a in apps}:
        return f"excluded app {stem}"
    pats = getattr(cfg, "exclude_title_patterns", None) or DEFAULT_EXCLUDE_TITLES
    for p in pats:
        if re.search(p, title):
            return "excluded title"
    return None
