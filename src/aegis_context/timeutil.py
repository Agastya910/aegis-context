"""Time helpers: store UTC, show local. Also parses the loose time expressions LLM tools send."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone, tzinfo

DB_FMT = "%Y-%m-%d %H:%M:%S"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def db_fmt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(DB_FMT)


def parse_db(s: str) -> datetime:
    return datetime.strptime(s, DB_FMT).replace(tzinfo=timezone.utc)


def to_local(dt: datetime, tz: tzinfo | None = None) -> datetime:
    return dt.astimezone(tz) if tz else dt.astimezone()


def fmt_local(dt: datetime | str, date: bool = False, tz: tzinfo | None = None) -> str:
    if isinstance(dt, str):
        dt = parse_db(dt)
    loc = to_local(dt, tz)
    return loc.strftime("%a %b %d %H:%M") if date else loc.strftime("%H:%M")


def tz_label(tz: tzinfo | None = None) -> str:
    off = to_local(utc_now(), tz).strftime("%z")  # e.g. -0700
    return f"UTC{off[:3]}:{off[3:]}" if off else "UTC"


_UNITS = {"min": 60, "minute": 60, "mins": 60, "minutes": 60, "m": 60,
          "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
          "day": 86400, "days": 86400, "d": 86400,
          "sec": 1, "secs": 1, "second": 1, "seconds": 1, "s": 1}
_PARTS = {"morning": 6, "afternoon": 12, "evening": 17, "night": 20}


def _local_midnight(now_local: datetime, days_back: int = 0) -> datetime:
    d = now_local.date() - timedelta(days=days_back)
    return datetime(d.year, d.month, d.day, tzinfo=now_local.tzinfo)


def _clock(s: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(?::\d{2})?\s*(am|pm)?", s.strip())
    if not m or (m.group(2) is None and m.group(3) is None):
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return (h, mi) if h < 24 and mi < 60 else None


def parse_when(s: str | None, now: datetime | None = None, tz: tzinfo | None = None) -> datetime | None:
    """Loose local-time parser. Returns aware UTC, or None if unparseable."""
    if not s:
        return None
    now = now or utc_now()
    nl = to_local(now, tz)
    raw = s.strip()
    t = raw.lower()
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):  # ISO date/datetime; naive = local
        try:
            d = datetime.fromisoformat(raw.replace("Z", "+00:00").replace("z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=nl.tzinfo)
            return d.astimezone(timezone.utc)
        except ValueError:
            pass
    if t in ("now", "right now", "current"):
        return now
    m = re.fullmatch(r"(?:about |around )?(\d+(?:\.\d+)?)\s*([a-z]+)\s+ago", t)
    if m and m.group(2) in _UNITS:
        return now - timedelta(seconds=float(m.group(1)) * _UNITS[m.group(2)])
    m = re.fullmatch(r"(?:an?|one)\s+(hour|minute|day)\s+ago", t)
    if m:
        return now - timedelta(seconds=_UNITS[m.group(1)])
    day_back = 0
    rest = t
    if rest.startswith("today"):
        rest = rest[5:].strip()
    elif rest.startswith("yesterday"):
        day_back, rest = 1, rest[9:].strip()
    elif rest.startswith("this "):
        rest = rest[5:].strip()
    rest = rest.removeprefix("at ").strip()
    base = _local_midnight(nl, day_back)
    if not rest:
        return base.astimezone(timezone.utc)
    if rest in _PARTS:
        return (base + timedelta(hours=_PARTS[rest])).astimezone(timezone.utc)
    c = _clock(rest)
    if c:
        return (base + timedelta(hours=c[0], minutes=c[1])).astimezone(timezone.utc)
    return None


def parse_range(start: str | None, end: str | None, now: datetime | None = None,
                tz: tzinfo | None = None) -> tuple[datetime, datetime]:
    """Resolve a (start, end) pair to an aware-UTC range: defaults, swap, clamp to now, widen tiny windows."""
    now = now or utc_now()
    e = parse_when(end, now, tz) or now
    s = parse_when(start, now, tz) or (e - timedelta(hours=1))
    if s > e:
        s, e = e, s
    if e > now:
        e = now
    if s > e:
        s = e - timedelta(minutes=5)
    if e - s < timedelta(minutes=4):
        mid = s + (e - s) / 2
        s, e = mid - timedelta(minutes=3), min(now, mid + timedelta(minutes=3))
    return s, e


def parse_day(s: str | None, now: datetime | None = None, tz: tzinfo | None = None) -> date:
    now = now or utc_now()
    nl = to_local(now, tz)
    t = (s or "today").strip().lower()
    if t == "today":
        return nl.date()
    if t == "yesterday":
        return nl.date() - timedelta(days=1)
    try:
        return date.fromisoformat(t[:10])
    except ValueError:
        d = parse_when(t, now, tz)
        return to_local(d, tz).date() if d else nl.date()


def find_range_in_text(text: str, now: datetime | None = None, tz: tzinfo | None = None
                       ) -> tuple[datetime, datetime] | None:
    """Best-effort time range from a free-text question (used by the no-LLM fallback)."""
    now = now or utc_now()
    t = text.lower()
    m = re.search(r"(\d+)\s*(minutes?|mins?|hours?|hrs?)\s+ago", t)
    if m:
        p = parse_when(f"{m.group(1)} {m.group(2)} ago", now, tz)
        return parse_range(db_iso(p - timedelta(minutes=5), tz), db_iso(p + timedelta(minutes=5), tz), now, tz)
    m = re.search(r"last\s+(\d+)?\s*(minutes?|mins?|hours?|hrs?)", t)
    if m:
        n = int(m.group(1) or 1)
        return now - timedelta(seconds=n * _UNITS[m.group(2)]), now
    if "yesterday" in t:
        s = parse_when("yesterday", now, tz)
        return s, s + timedelta(days=1)
    if "today" in t or "this morning" in t:
        return parse_when("today", now, tz), now
    return None


def db_iso(dt: datetime, tz: tzinfo | None = None) -> str:
    """Local ISO string (no offset), the format tools speak."""
    return to_local(dt, tz).strftime("%Y-%m-%dT%H:%M:%S")
