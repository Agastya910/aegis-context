"""Tier-1 hot store: thread-safe in-RAM rolling window of recent screen states."""
from __future__ import annotations

import threading
import time
from collections import deque


class RingBuffer:
    def __init__(self, window_s: int = 300, max_items: int = 1024):
        self.window_s = window_s
        self._items: deque[dict] = deque(maxlen=max_items)
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        while self._items and now - self._items[0]["ts"] > self.window_s:
            self._items.popleft()

    def add(self, app: str, window_title: str, summary: str, ts: float | None = None) -> dict:
        ts = time.time() if ts is None else ts
        item = {"ts": ts, "app": app, "window_title": window_title, "summary": summary}
        with self._lock:
            last = self._items[-1] if self._items else None
            if last and (last["app"], last["window_title"], last["summary"]) == (app, window_title, summary):
                last["ts"] = ts  # same state again: refresh instead of duplicating
                return last
            self._items.append(item)
            self._evict(ts)
        return item

    def recent(self, n: int | None = None, within_s: float | None = None) -> list[dict]:
        now = time.time()
        with self._lock:
            self._evict(now)
            items = list(self._items)
        if within_s is not None:
            items = [i for i in items if now - i["ts"] <= within_s]
        return items[-n:] if n else items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
