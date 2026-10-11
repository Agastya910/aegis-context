"""Darwin (darwin.so) agent discovery: find external agents for what you're doing — without sharing your memory.

Only a short, generic description of the need is sent (scrubbed of secrets and PII, capped at 300 chars).
Uses Darwin's public anonymous Search (POST https://api.darwin.so/api/v2/search); no account or key needed.
Acting on a result (threads, auth, payment) happens on darwin.so with the user's explicit approval.
"""
from __future__ import annotations

import requests

from .safe import scrub

DARWIN_API = "https://api.darwin.so/api/v2"
BROWSE_URL = "https://darwin.so"


class DarwinError(Exception):
    pass


class DarwinClient:
    def __init__(self, base: str = DARWIN_API, timeout: float = 20, session=None):
        self.base, self.timeout = base.rstrip("/"), timeout
        self.session = session or requests.Session()

    @staticmethod
    def sanitize(need: str) -> str:
        clean, _ = scrub(" ".join((need or "").split()))
        return clean[:300]

    def search(self, need: str, objective: str | None = None, n: int = 5) -> tuple[str, list[dict]]:
        """Returns (the exact query sent, results)."""
        query = self.sanitize(need)
        if not query:
            raise DarwinError("empty need")
        body: dict = {"query": query, "numResults": max(1, min(n, 10))}
        if objective:
            body["objective"] = self.sanitize(objective)[:200]
        try:
            r = self.session.post(f"{self.base}/search", json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise DarwinError(f"Darwin unreachable: {exc}") from exc
        if r.status_code == 429:
            raise DarwinError(f"Darwin rate limit; retry in {r.headers.get('Retry-After', '60')} s")
        if r.status_code >= 400:
            raise DarwinError(f"Darwin HTTP {r.status_code}")
        data = r.json()
        results = []
        for item in data.get("results", [])[:n]:
            results.append({
                "rank": item.get("rank"), "name": item.get("name", ""),
                "description": " ".join((item.get("description") or "").split()),
                "readiness": item.get("readiness", ""), "can_start_thread": bool(item.get("canStartThread")),
                "agent": item.get("agent"), "capability": item.get("capability"),
            })
        return query, results


def format_results(query: str, results: list[dict]) -> str:
    if not results:
        return f'Darwin found no agents for "{query}". Browse manually at {BROWSE_URL}.'
    lines = [f'Darwin results for "{query}" (sent: only this query):']
    for r in results:
        desc = r["description"][:280] + ("…" if len(r["description"]) > 280 else "")
        ready = "ready to start" if r["can_start_thread"] else (r["readiness"] or "listed")
        lines.append(f"{r['rank']}. {r['name']} — {desc} ({ready})")
    lines.append(f"Open {BROWSE_URL} to start a thread with one (requires the user's approval there).")
    return "\n".join(lines)
