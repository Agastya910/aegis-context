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
    def __init__(self, base: str = DARWIN_API, timeout: float = 45, session=None, api_key: str = ""):
        self.base, self.timeout, self.api_key = base.rstrip("/"), timeout, api_key
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
        if self.api_key:  # v3: ranked agents with reasons
            d = search_v3(self.api_key, query, objective or "", session=self.session, timeout=self.timeout)
            if d["status"] == "needs_input":
                return query, [{"rank": "?", "name": "Darwin needs more detail", "description": d["question"] or "",
                                "readiness": "", "can_start_thread": False}]
            return query, [{"rank": m["rank"], "name": m["name"], "readiness": m["readiness"],
                            "can_start_thread": m["can_start_thread"],
                            "description": (m["reasons"][0] if m["reasons"] else m["description"])}
                           for m in d["matches"]]
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


# ---- v3 (authenticated) ---------------------------------------------------------
DARWIN_V3 = "https://api.darwin.so/api/v3"


def search_v3(api_key: str, query: str, context: str = "", session=None, timeout: float = 45) -> dict:
    """Darwin v3 Search with an app key. Returns a normalized dict:
    {status: completed|needs_input|no_match, search_id, query, question, no_match_reason, overview, matches[]}.
    Only `query` and `context` are sent; both are scrubbed locally first."""
    q, _ = scrub(" ".join((query or "").split()))
    ctx, _ = scrub(context or "")
    body: dict = {"query": q[:500]}
    if ctx.strip():
        body["context"] = [{"type": "text", "text": ctx[:2000]}]
    s = session or requests.Session()
    try:
        r = s.post(f"{DARWIN_V3}/search", json=body, timeout=timeout,
                   headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"})
    except requests.RequestException as exc:
        raise DarwinError(f"Darwin unreachable: {exc}") from exc
    if r.status_code >= 400:
        raise DarwinError(f"Darwin HTTP {r.status_code}")
    d = r.json()
    resp = d.get("response") or {}
    matches = [{
        "rank": a.get("rank"), "name": a.get("name", ""), "agent_id": a.get("agentId"),
        "capability_id": a.get("capabilityId"),
        "description": " ".join((a.get("description") or "").split())[:400],
        "reasons": a.get("reasons") or [], "uncertainties": a.get("uncertainties") or [],
        "readiness": a.get("readiness", ""), "can_start_thread": bool(a.get("canStartThread")),
        "required_setup": a.get("requiredSetup"),
        "connection_methods": [c.get("type") or c.get("kind") or str(c)[:40] for c in (a.get("connectionMethods") or [])
                               if isinstance(c, dict)],
    } for a in (resp.get("agents") or [])[:5]]
    return {"status": d.get("status"), "search_id": d.get("searchId"), "query": body["query"],
            "context_sent": ctx[:2000], "question": resp.get("question"),
            "no_match_reason": resp.get("noMatchReason"), "overview": resp.get("overview"), "matches": matches}
