"""Split screen text into overlapping ~800-char passages for retrieval."""
from __future__ import annotations

import re


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    # units: paragraphs -> lines -> hard splits, each <= size
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        for line in para.splitlines() or [para]:
            line = line.rstrip()
            while len(line) > size:
                cut = line.rfind(" ", 0, size)
                cut = cut if cut > size // 2 else size
                units.append(line[:cut])
                line = line[cut:].lstrip()
            if line:
                units.append(line)
        units.append("")  # paragraph boundary marker
    chunks: list[str] = []
    cur = ""
    for u in units:
        piece = u if u else ""
        candidate = (cur + "\n" + piece) if cur else piece
        if len(candidate) > size and cur:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            sp = tail.find(" ")
            tail = tail[sp + 1:] if 0 <= sp < len(tail) - 1 else tail
            cur = (tail + "\n" + piece).strip() if tail and len(tail) + len(piece) < size else piece
        else:
            cur = candidate
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]
