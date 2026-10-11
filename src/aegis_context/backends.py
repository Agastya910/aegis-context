"""Pluggable perception + embedding backends. Defaults run fully offline with no models."""
from __future__ import annotations

import base64
import hashlib
import io
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class VisionBackend(Protocol):
    def describe(self, image_png: bytes) -> tuple[str, str]:
        """Return (ocr_text, semantic_summary)."""


_TOKEN = re.compile(r"[a-z0-9_]+")


class HashingEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams), L2-normalised.

    No model download needed; good enough for keyword-ish recall and for tests.
    Swap for OnnxEmbedder (all-MiniLM-L6-v2) for true semantic search.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(vec)
        return vec / n if n else vec


class OnnxEmbedder:
    """all-MiniLM-L6-v2 (or compatible) via onnxruntime. Install with `uv sync --extra onnx`."""

    def __init__(self, model_path: str, tokenizer_path: str, dim: int = 384, max_len: int = 256):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.dim = dim
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(length=None)
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def embed(self, text: str) -> np.ndarray:
        enc = self.tok.encode(text or " ")
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0]  # (1, seq, dim)
        m = mask[..., None].astype(np.float32)
        vec = (out * m).sum(1)[0] / max(m.sum(), 1.0)
        n = np.linalg.norm(vec)
        return (vec / n if n else vec).astype(np.float32)


class OllamaEmbedder:
    """Embeddings from a local Ollama model (e.g. nomic-embed-text, 768-dim)."""

    def __init__(self, url: str = "http://127.0.0.1:11434", model: str = "nomic-embed-text", timeout: float = 120):
        import requests

        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self._session = requests.Session()
        # nomic-embed-text is trained with task prefixes; without them retrieval quality drops noticeably.
        nomic = model.startswith("nomic-embed")
        self._doc_prefix, self._query_prefix = ("search_document: ", "search_query: ") if nomic else ("", "")
        self.dim = len(self._raw("dimension probe"))

    def _raw(self, text: str) -> list[float]:
        r = self._session.post(
            f"{self.url}/api/embed", json={"model": self.model, "input": text or " "}, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]

    def _norm(self, raw: list[float]) -> np.ndarray:
        vec = np.asarray(raw, dtype=np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    def embed(self, text: str) -> np.ndarray:
        return self._norm(self._raw(self._doc_prefix + (text or "")))

    def embed_query(self, text: str) -> np.ndarray:
        return self._norm(self._raw(self._query_prefix + (text or "")))

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        r = self._session.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": [self._doc_prefix + (t or " ") for t in texts]},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return [self._norm(v) for v in r.json()["embeddings"]]


def embed_query(embedder: Embedder, text: str) -> np.ndarray:
    """Query-side embedding; falls back to `embed` for symmetric embedders."""
    fn = getattr(embedder, "embed_query", None)
    return fn(text) if fn else embedder.embed(text)


def embed_many(embedder, texts: list[str]) -> list[np.ndarray]:
    """Batch document embedding; one HTTP call for Ollama, a loop otherwise."""
    if not texts:
        return []
    fn = getattr(embedder, "embed_many", None)
    return list(fn(texts)) if fn else [embedder.embed(t) for t in texts]


@dataclass
class VisionResult:
    activity: str = ""
    typed: str = ""
    entities: str = ""
    text: str = ""


VISION_PROMPT = """You are capturing the user's screen for their private memory log.
Foreground app: {app}. Window title: {title}.
Reply in exactly this format:
ACTIVITY: <1-2 sentences: what the user is doing; name the specific file, web page, document, chat, person or error>
TYPED: <text the user is typing or has just typed into an input box or prompt, verbatim; NONE if none>
ENTITIES: <comma-separated files, sites/domains, people, projects, products, error messages; NONE if none>
TEXT:
<all legible text on screen, top to bottom, as plain Markdown>"""

_SECTION = re.compile(r"^\s*[#*\-\s]*\**\s*(ACTIVITY|TYPED|ENTITIES|TEXT)\s*\**\s*:\s*\**\s*(.*)$", re.I)


def parse_vision(raw: str) -> VisionResult:
    """Tolerant parser for the sectioned vision reply (handles fences, bold headers, missing sections)."""
    raw = re.sub(r"^```[a-zA-Z]*\s*$", "", raw or "", flags=re.M)
    out: dict[str, list[str]] = {}
    cur = None
    for line in raw.splitlines():
        m = _SECTION.match(line)
        if m and (m.group(1).upper() not in out):
            cur = m.group(1).upper()
            out[cur] = [m.group(2)] if m.group(2).strip() else []
        elif cur:
            out[cur].append(line)
    if not out:  # model ignored the format: treat everything as activity/text
        txt = raw.strip()
        return VisionResult(activity=txt.split("\n")[0][:300], text=txt)

    def get(k: str, multiline: bool = False) -> str:
        v = "\n".join(out.get(k, [])).strip() if multiline else " ".join(
            x.strip() for x in out.get(k, [])).strip()
        v = v.strip("*").strip()
        return "" if v.upper() in ("NONE", "N/A", "NONE.") else v

    return VisionResult(get("ACTIVITY"), get("TYPED", True).strip('"'), get("ENTITIES"), get("TEXT", True))


class NullVision:
    """No VLM configured: records only window metadata."""

    def describe(self, image_png: bytes) -> tuple[str, str]:
        return "", ""


class OllamaVision:
    """Moondream2 (or any vision model) through a local Ollama server."""

    def __init__(self, url: str = "http://127.0.0.1:11434", model: str = "moondream", timeout: float = 60, api_key: str = ""):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _ask(self, img_b64: str, prompt: str) -> str:
        import requests

        r = requests.post(
            f"{self.url}/api/generate",
            json={"model": self.model, "prompt": prompt, "images": [img_b64], "stream": False},
            headers=self.headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def analyze(self, image_png: bytes, app: str = "", title: str = "") -> VisionResult:
        """One call: activity + typed text + entities + screen text."""
        b64 = base64.b64encode(image_png).decode()
        return parse_vision(self._ask(b64, VISION_PROMPT.format(app=app or "unknown", title=title or "")))

    def describe(self, image_png: bytes) -> tuple[str, str]:
        r = self.analyze(image_png)
        return r.text, r.activity


def make_embedder(cfg) -> Embedder:
    if cfg.embedder == "onnx":
        return OnnxEmbedder(cfg.onnx_model_path, cfg.onnx_tokenizer_path, cfg.embedding_dim)
    if cfg.embedder == "ollama":
        return OllamaEmbedder(cfg.ollama_url, cfg.embed_model)
    return HashingEmbedder(cfg.embedding_dim)


def make_vision(cfg) -> VisionBackend:
    if cfg.vision_backend == "ollama":
        return OllamaVision(cfg.vision_url or cfg.ollama_url, cfg.vision_model, api_key=cfg.vision_api_key)
    return NullVision()


def png_bytes(img) -> bytes:  # PIL image -> PNG bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
