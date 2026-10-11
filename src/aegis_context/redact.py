"""Secret redaction for anything stored or sent to an LLM."""
from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("ollama", re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9]{24}\b")),
    # OCR'd keys come back garbled (spaces, swapped chars): long hex-ish run, a dot, long alnum run
    ("ollama", re.compile(r"\b[0-9a-f]{4}[0-9a-f ]{18,44}[0-9a-f]\.\s?[A-Za-z0-9][A-Za-z0-9 ]{14,30}[A-Za-z0-9]")),
    ("api_key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}")),
    ("aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("slack", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("google", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
]
_ASSIGN = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD)[A-Z0-9_]*)(\s*[:=]\s*[\"']?)([^\s\"']{6,})"
)


def redact(text: str) -> str:
    if not text:
        return text
    for kind, pat in _PATTERNS:
        text = pat.sub(f"[REDACTED:{kind}]", text)
    return _ASSIGN.sub(lambda m: m.group(1) + m.group(2) + (
        m.group(3) if m.group(3).startswith("[REDACTED") else "[REDACTED:secret]"), text)
