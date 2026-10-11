"""Automate my most repetitive task: local memory picks the task, Darwin finds an agent for it.

1. Measure (local): group recent episodes into repeated tasks (same app + normalized window stem); cost =
   runs x average minutes. The top one is the automation candidate.
2. Describe (LLM, generic): turn that task's episode summaries into a capability sentence + steps with no names,
   companies, emails or file contents; then the privacy firewall scrubs it again.
3. Find (Darwin v3): only that generic description leaves the device; Darwin returns ranked agents with reasons,
   uncertainties and readiness, or a clarifying question, or a no-match reason.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import timedelta

from .darwin import DarwinError, search_v3
from .insights import SENSITIVE_CATEGORIES, categorize
from .retrieval import Retrieval
from .safe import scrub

_STEM_STRIP = re.compile(r"\(\d[\d,]*\)|\b\d+\b|[-–|·]\s*(google chrome|microsoft edge|visual studio code|notepad)\s*$",
                         re.I)

DESCRIBE = """You help a user automate a task they repeat on their computer. From these observations of the SAME
task done several times, write what an external AI agent would need to do it for them.
Rules: be generic. No person names, company names, emails, URLs, file contents or numbers from the screen.
Reply in exactly this format:
CAPABILITY: <one sentence, e.g. "Extract vendor, date and amount from PDF receipts into an expense spreadsheet">
STEPS:
- <step>
- <step>
(3-6 steps)

Task: {app} | {title} ({runs} runs, {avg:.0f} min each on average)
Observations:
{observations}"""


def _stem(app: str, title: str) -> str:
    t = _STEM_STRIP.sub("", title or "").strip(" -|·")
    return f"{app.lower()}|{' '.join(t.lower().split())[:60]}"


def repetitive_tasks(r: Retrieval, days: int = 7, limit: int = 5) -> list[dict]:
    now = r.clock()
    rows, _ = r.timeline(now - timedelta(days=days), now, limit=5000)
    groups: dict[str, list] = defaultdict(list)
    for h in rows:
        if categorize(h.app, h.title) in SENSITIVE_CATEGORIES:
            continue
        groups[_stem(h.app, h.title)].append(h)
    tasks = []
    for key, hs in groups.items():
        hs.sort(key=lambda h: h.start)
        # a "run" = episodes of this task separated by >= 10 min of doing something else
        runs, last_end = [], None
        for h in hs:
            if last_end is None or h.start - last_end > timedelta(minutes=10):
                runs.append([h])
            else:
                runs[-1].append(h)
            last_end = h.end
        mins = [max(1.0, (rn[-1].end - rn[0].start).total_seconds() / 60) for rn in runs]
        total = sum(mins)
        tasks.append({"key": key, "app": hs[0].app, "title": hs[-1].title, "runs": len(runs),
                      "total_min": round(total, 1), "avg_min": round(total / len(runs), 1),
                      "score": round(len(runs) * total / len(runs), 1),
                      "ids": [h.source_id for h in hs][-8:],
                      "observations": list(dict.fromkeys(h.activity for h in hs if h.activity))[-8:]})
    tasks.sort(key=lambda t: (-(t["runs"] >= 2), -t["score"]))
    return tasks[:limit]


def describe_task(task: dict, llm=None) -> tuple[str, list[str]]:
    """Generic capability + steps for a task (LLM if available, else a template)."""
    if llm is not None:
        try:
            text, _ = llm.complete([{"role": "user", "content": DESCRIBE.format(
                app=task["app"], title=task["title"], runs=task["runs"], avg=task["avg_min"],
                observations="\n".join(f"- {o[:300]}" for o in task["observations"]))}])
            cap = re.search(r"CAPABILITY:\s*(.+)", text)
            steps = [s.strip(" -*") for s in re.findall(r"^\s*[-*]\s*(.+)$", text, re.M)]
            if cap:
                return scrub(cap.group(1).strip())[0], [scrub(s)[0] for s in steps[:6]]
        except Exception:
            pass
    return (f"Automate a repeated {task['app']} task: {scrub(task['title'])[0]}",
            [scrub(o[:160])[0] for o in task["observations"][:4]])


def automate(r: Retrieval, darwin_key: str, llm=None, task_text: str | None = None, days: int = 7) -> dict:
    """End to end. If task_text is given (the user's own words), skip detection and use it."""
    if task_text and task_text.strip():
        task = {"app": "", "title": task_text.strip(), "runs": None, "avg_min": None, "total_min": None,
                "ids": [], "observations": [task_text.strip()], "source": "user"}
        capability, steps = scrub(task_text.strip())[0], []
        if llm is not None:
            capability, steps = describe_task({**task, "app": "any", "runs": 1, "avg_min": 0}, llm)
    else:
        tasks = repetitive_tasks(r, days)
        if not tasks:
            return {"ok": False, "error": "No repeated tasks in memory yet. Use Aegis for a while, or describe a task."}
        task = {**tasks[0], "source": "measured", "alternatives": [
            {k: t[k] for k in ("app", "title", "runs", "avg_min", "total_min")} for t in tasks[1:]]}
        capability, steps = describe_task(task, llm)
    context = "Steps the user currently does by hand:\n" + "\n".join(f"- {s}" for s in steps) if steps else ""
    out = {"ok": True, "task": {k: v for k, v in task.items() if k != "observations"},
           "capability": capability, "steps": steps}
    if not darwin_key:
        out.update(ok=False, error="Set DARWIN_API_KEY in .env to search Darwin.")
        return out
    try:
        out["darwin"] = search_v3(darwin_key, capability, context)
    except DarwinError as exc:
        out.update(ok=False, error=str(exc))
    return out
