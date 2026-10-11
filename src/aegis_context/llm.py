"""Ollama /api/chat client (local or ollama.com) with streaming, tool calls, retries and a local fallback."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests

log = logging.getLogger("aegis.llm")


class LLMError(Exception):
    pass


class _Retryable(Exception):
    pass


@dataclass
class LLMEvent:
    type: str  # "token" | "tool_calls" | "status" | "done"
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)


class OllamaChatClient:
    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 120, think: bool | None = False):
        self.url, self.model, self.timeout, self.think = url.rstrip("/"), model, timeout, think
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.session = requests.Session()

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[LLMEvent]:
        body: dict = {"model": self.model, "messages": messages, "stream": True}
        if self.think is not None:
            body["think"] = self.think
        if tools:
            body["tools"] = tools
        started = False
        for attempt in range(3):
            try:
                calls: list[dict] = []
                with self.session.post(f"{self.url}/api/chat", json=body, headers=self.headers, stream=True,
                                       timeout=(10, self.timeout)) as r:
                    if r.status_code == 429 or r.status_code >= 500:
                        raise _Retryable(f"HTTP {r.status_code}")
                    if r.status_code >= 400:
                        raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                    for line in r.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if "error" in chunk:
                            raise LLMError(str(chunk["error"]))
                        msg = chunk.get("message") or {}
                        if msg.get("content"):
                            started = True
                            yield LLMEvent("token", msg["content"])
                        calls += msg.get("tool_calls") or []
                        if chunk.get("done"):
                            break
                if calls:
                    yield LLMEvent("tool_calls", tool_calls=calls)
                yield LLMEvent("done")
                return
            except requests.Timeout as exc:  # a hung request won't get better by waiting again: fail fast
                raise LLMError(f"{self.model}: timed out after {self.timeout}s") from exc
            except (requests.ConnectionError, _Retryable) as exc:
                if started or attempt == 2:
                    raise LLMError(f"{self.model}: {exc}") from exc
                log.warning("LLM attempt %d failed (%s); retrying", attempt + 1, exc)
                time.sleep((1, 3)[attempt])

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[str, list[dict]]:
        text, calls = "", []
        for ev in self.stream(messages, tools):
            if ev.type == "token":
                text += ev.text
            elif ev.type == "tool_calls":
                calls = ev.tool_calls
        return text, calls


class FallbackChat:
    """Primary model (usually cloud); on failure before the first token, switch to the local fallback."""

    def __init__(self, primary: OllamaChatClient, fallback: OllamaChatClient | None = None):
        self.primary, self.fallback = primary, fallback
        self.model = primary.model

    def describe(self) -> dict:
        return {"primary": self.primary.model, "fallback": self.fallback.model if self.fallback else None}

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[LLMEvent]:
        started = False
        try:
            for ev in self.primary.stream(messages, tools):
                started = started or ev.type == "token"
                yield ev
            return
        except LLMError as exc:
            if started or not self.fallback:
                raise
            log.warning("primary LLM failed: %s", exc)
        yield LLMEvent("status", f"Cloud model unavailable, using local {self.fallback.model}")
        yield from self.fallback.stream(messages, tools)

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[str, list[dict]]:
        text, calls = "", []
        for ev in self.stream(messages, tools):
            if ev.type == "token":
                text += ev.text
            elif ev.type == "tool_calls":
                calls = ev.tool_calls
        return text, calls


def make_chat_client(cfg) -> FallbackChat | None:
    if getattr(cfg, "chat_backend", "ollama") == "none":
        return None
    remote_url = cfg.chat_url or cfg.vision_url
    local = (OllamaChatClient(cfg.ollama_url, cfg.chat_fallback_model, think=None)
             if cfg.chat_fallback_model else None)
    if remote_url and cfg.vision_api_key:
        model = cfg.chat_model or cfg.answer_model or cfg.vision_model
        # read timeout = max silence between stream chunks; cloud normally answers in < 5 s
        return FallbackChat(OllamaChatClient(remote_url, model, cfg.vision_api_key, timeout=30, think=False), local)
    if cfg.chat_model:
        return FallbackChat(OllamaChatClient(cfg.ollama_url, cfg.chat_model, think=None))
    return FallbackChat(local) if local else None
