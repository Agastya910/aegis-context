"""Part 1 pipeline: capture -> vision -> embed -> store (+ Tier-1 ring buffer).

Two threads: capture (cheap: window filter + perceptual-hash change detection, never touches the network) and
process (one vision call, redaction, dedupe, chunking, embedding, episodes). A small queue between them drops the
oldest frame when processing falls behind.
"""
from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .backends import Embedder, VisionBackend, VisionResult
from .capture import Frame, WindowInfo, hamming
from .filters import normalize_title, skip_reason
from .ingest import Ingestor
from .ringbuffer import RingBuffer
from .storage import AegisStorageEngine
from .timeutil import utc_now

log = logging.getLogger("aegis.daemon")


@dataclass
class Candidate:
    win: WindowInfo
    frame: Frame
    ts: datetime


class CaptureDaemon:
    def __init__(
        self,
        storage: AegisStorageEngine,
        ring: RingBuffer,
        embedder: Embedder,
        vision: VisionBackend,
        grab: Callable[[], bytes | Frame],
        window: Callable[[], WindowInfo],
        interval_s: float = 5.0,
        digest_after_s: int = 3600,
        cfg=None,
        idle: Callable[[], float] | None = None,
        jobs=None,
        hud_visible: Callable[[], bool] | None = None,
    ):
        self.storage, self.ring, self.embedder, self.vision = storage, ring, embedder, vision
        self.grab, self.window, self.idle = grab, window, idle or (lambda: 0.0)
        self.hud_visible = hud_visible or (lambda: False)
        self.interval_s, self.digest_after_s = interval_s, digest_after_s
        self.cfg, self.jobs = cfg, jobs
        self.raw_retention_days = getattr(cfg, "raw_retention_days", 7)
        self.dhash_threshold = getattr(cfg, "dhash_threshold", 8)
        self.idle_after_s = getattr(cfg, "idle_after_s", 120)
        self.vision_min_interval_s = getattr(cfg, "vision_min_interval_s", 0.0)
        self.ingestor = Ingestor(storage, embedder, ring, cfg)
        self.vision_errors = 0
        self.last_capture: datetime | None = None
        self._q: queue.Queue[Candidate] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_key: tuple | None = None
        self._last_sig: str | None = None
        self._last_dhash: int | None = None
        self._last_vision = 0.0
        self._last_maint = 0.0

    # ---- capture side ---------------------------------------------------
    def capture_once(self) -> Candidate | None:
        win = self.window()
        if skip_reason(win, self.cfg):
            return None
        if self.hud_visible():  # HUD on screen: don't record Aegis's own answers; keep the episode alive
            self.ingestor.heartbeat()
            return None
        got = self.grab()
        frame = got if isinstance(got, Frame) else Frame(got, None)
        key = (win.bundle_id, normalize_title(win.title))
        if frame.dhash is not None and self._last_dhash is not None and key == self._last_key:
            idle = self.idle() >= self.idle_after_s  # user away: ignore pixel churn (video, clock)
            same = idle or hamming(frame.dhash, self._last_dhash) <= self.dhash_threshold
        else:
            sig = hashlib.sha1(frame.png + key[1].encode()).hexdigest()
            same = sig == self._last_sig
            self._last_sig = sig
        if same:
            self.ingestor.heartbeat()
            return None
        self._last_key, self._last_dhash = key, frame.dhash
        return Candidate(win, frame, utc_now())

    # ---- process side ---------------------------------------------------
    def _analyze(self, c: Candidate) -> VisionResult:
        try:
            if hasattr(self.vision, "analyze"):
                return self.vision.analyze(c.frame.png, c.win.app_name, c.win.title)
            ocr, summary = self.vision.describe(c.frame.png)
            return VisionResult(activity=summary or "", text=ocr or "")
        except Exception as exc:
            self.vision_errors += 1
            log.warning("vision failed: %s", exc)
            return VisionResult()

    def process(self, c: Candidate) -> bool:
        res = self._analyze(c)
        self._last_vision = time.time()
        fid = self.ingestor.ingest(c.win.bundle_id, c.win.app_name, c.win.platform, c.win.title, res, c.ts,
                                   f"{c.frame.dhash:064x}" if c.frame.dhash is not None else None)
        self.last_capture = c.ts
        return fid is not None

    def tick(self) -> bool:
        """One synchronous capture+process cycle. Returns True if a new frame was stored."""
        c = self.capture_once()
        return self.process(c) if c else False

    # ---- threads --------------------------------------------------------
    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                c = self.capture_once()
                if c:
                    if self._q.full():
                        try:
                            self._q.get_nowait()
                        except queue.Empty:
                            pass
                    self._q.put_nowait(c)
            except Exception:
                log.exception("capture failed")
            self._stop.wait(self.interval_s)

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                c = self._q.get(timeout=1)
            except queue.Empty:
                c = None
            try:
                if c:
                    wait = self.vision_min_interval_s - (time.time() - self._last_vision)
                    if wait > 0:
                        self._stop.wait(wait)
                    while not self._q.empty():  # behind? only the newest frame matters
                        c = self._q.get_nowait()
                    self.process(c)
                if self.jobs is None and time.time() - self._last_maint > 600:  # MemoryJobs does tiered retention
                    self._last_maint = time.time()
                    n = self.storage.prune_frames(self.raw_retention_days)
                    if n:
                        log.info("pruned %d frames past retention", n)
            except Exception:
                log.exception("processing failed")

    def start(self) -> None:
        self.storage.close_open_episodes()
        for name, fn in (("aegis-capture", self._capture_loop), ("aegis-process", self._process_loop)):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
