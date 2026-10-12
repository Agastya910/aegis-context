"""Desktop floating HUD + global hotkey: a streaming chat with your screen memory. Needs `uv sync --extra hud`."""
from __future__ import annotations

import json
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import requests

from .filters import HUD_TITLE

BASE = "http://127.0.0.1:8765"
_CITE = re.compile(r"\[((?:[EFD]\d+)(?:\s*,\s*[EFD]\d+)*)\]")


def _force_foreground(hwnd: int) -> None:
    """Windows refuses SetForegroundWindow from background processes; borrow the foreground thread's input."""
    if sys.platform != "win32":
        return
    import ctypes

    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    fg_thread = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    me = kernel32.GetCurrentThreadId()
    attached = fg_thread != me and user32.AttachThreadInput(fg_thread, me, True)
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    if attached:
        user32.AttachThreadInput(fg_thread, me, False)


def _exclude_from_capture(hwnd: int) -> None:
    """WDA_EXCLUDEFROMCAPTURE (Win10 2004+): keep the HUD out of the daemon's screenshots."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11)
    except Exception:
        pass


# ---- pure helpers (unit-tested without Qt) ---------------------------------
def parse_sse_lines(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    event, data = "message", []
    for line in lines:
        if line is None:
            continue
        if line == "":
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if data:
        yield event, json.loads("\n".join(data))


def linkify(text: str) -> str:
    """[E12, F3] -> clickable markdown links aegis:E12 / aegis:F3."""
    def rep(m):
        ids = [s.strip() for s in m.group(1).split(",")]
        return " ".join(f"[{i}](aegis:{i})" for i in ids)
    return _CITE.sub(rep, text)


def render_markdown(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        if m["role"] == "user":
            quoted = "\n".join("> " + ln for ln in m["text"].splitlines() or [""])
            parts.append(f"**You** · {m.get('time', '')}\n\n{quoted}")
        elif m["role"] == "info":
            parts.append(m["text"])
        else:
            body = ""
            if m.get("steps"):
                body += "\n".join(f"*· {s}*  " for s in m["steps"]) + "\n\n"
            body += linkify(m["text"]) if m["text"] else ("*thinking…*" if m.get("streaming") else "")
            if m.get("sources"):
                body += "\n\nSources: " + " · ".join(
                    f"[{s['id']}](aegis:{s['id']}) {s['start_local']} {s['app']}" for s in m["sources"])
            if m.get("error"):
                body += f"\n\n⚠ {m['error']}"
            parts.append(f"**Aegis**\n\n{body}")
    return "\n\n---\n\n".join(parts)


def _state_path() -> Path:
    return Path.home() / ".aegis" / "hud_state.json"


def _load_state() -> dict:
    try:
        return json.loads(_state_path().read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        _state_path().parent.mkdir(parents=True, exist_ok=True)
        _state_path().write_text(json.dumps(state))
    except Exception:
        pass


def run_hud(hotkey: str = "<ctrl>+<shift>+<space>", base: str = BASE, hide_from_capture: bool = False) -> None:
    """hide_from_capture=True makes the HUD invisible to ALL screen capture (incl. OBS / screen share);
    by default it stays visible and the daemon simply pauses while the HUD is on screen."""
    from pynput import keyboard
    from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont, QKeySequence
    from PyQt5.QtWidgets import (
        QApplication,
        QDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QShortcut,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )

    STYLE = (
        "QWidget{background:rgb(20,20,30);color:#eee}"
        "QTextBrowser{background:rgb(26,26,38);border:1px solid #333;border-radius:6px;padding:6px}"
        "QLineEdit{background:rgb(30,30,45);border:1px solid #444;border-radius:5px;padding:8px;"
        "color:#00FFCC;font-family:'Cascadia Mono','Consolas',monospace}"
        "QPushButton{background:rgb(45,45,70);border:1px solid #555;border-radius:5px;padding:5px 9px}"
        "QPushButton:hover{background:rgb(60,60,95)} QPushButton:disabled{color:#666}"
    )

    class Bridge(QObject):
        fire = pyqtSignal()
        ev = pyqtSignal(str, object)

    class SourceDialog(QDialog):
        def __init__(self, parent, sid: str, text: str):
            super().__init__(parent)
            self.setWindowTitle(f"{HUD_TITLE} · {sid}")
            self.setStyleSheet(STYLE)
            self.resize(560, 420)
            lay = QVBoxLayout(self)
            view = QTextBrowser()
            view.setPlainText(text)
            lay.addWidget(view)

    class HUD(QWidget):
        def __init__(self, bridge: Bridge):
            super().__init__()
            self.bridge = bridge
            self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
            self.setWindowTitle(HUD_TITLE)  # the daemon skips windows with this title
            self.resize(680, 520)
            self.setMinimumSize(380, 240)
            self.setStyleSheet(STYLE)
            lay = QVBoxLayout(self)
            self.header = QLabel("Ctrl+Shift+Space to toggle · Esc to hide")
            self.header.setStyleSheet("color:#8af")
            self.header.setWordWrap(True)
            self.chat = QTextBrowser()
            self.chat.setOpenLinks(False)
            self.chat.setFont(QFont("Segoe UI", 10))
            self.chat.document().setDefaultStyleSheet("a{color:#7fdcff;text-decoration:none}")
            self.chat.anchorClicked.connect(self.open_source)
            self.status = QLabel("")
            self.status.setStyleSheet("color:#999")
            self.input = QLineEdit()
            self.input.setPlaceholderText("Ask about anything you've had on screen… (Enter on empty = copy safe context)")
            self.input.returnPressed.connect(self.submit)
            row = QHBoxLayout()
            self.buttons = {}
            for text, slot in (("Send", self.submit), ("Stop", self.stop), ("New chat", self.new_chat),
                               ("Recent", self.show_recent), ("Insights", self.insights),
                               ("Find agents", self.find_agents), ("Copy answer", self.copy_answer),
                               ("Copy safe context", self.copy_context)):
                b = QPushButton(text)
                b.clicked.connect(slot)
                row.addWidget(b)
                self.buttons[text] = b
            row.addStretch(1)
            lay.addWidget(self.header)
            lay.addWidget(self.chat, 1)
            lay.addWidget(self.status)
            lay.addLayout(row)
            lay.addWidget(self.input)
            QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.hide)
            QShortcut(QKeySequence(Qt.Key_Up), self.input, activated=self.recall)
            self.messages: list[dict] = []
            self.session_id: str | None = _load_state().get("session_id")
            self.streaming = False
            self.stop_flag = threading.Event()
            self.last_question = ""
            self.context = ""
            self._placed = self._affinity = False
            self._render_pending = False
            bridge.ev.connect(self.on_event)
            self._set_streaming(False)
            if self.session_id:
                self._bg(lambda: ("history", requests.get(f"{base}/chat/sessions/{self.session_id}", timeout=5)
                                  .json()))

        # --- helpers ------------------------------------------------------
        def _bg(self, fn):
            """Run fn() off the GUI thread; it returns (event, data) which is delivered via the signal."""
            def work():
                try:
                    ev, data = fn()
                    self.bridge.ev.emit(ev, data)
                except Exception as exc:
                    self.bridge.ev.emit("bg_error", {"text": str(exc)})
            threading.Thread(target=work, daemon=True).start()

        def _set_streaming(self, on: bool):
            self.streaming = on
            self.buttons["Stop"].setEnabled(on)
            self.buttons["Send"].setEnabled(not on)

        def render(self):
            self._render_pending = False
            sb = self.chat.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 4
            self.chat.setMarkdown(render_markdown(self.messages))
            if at_bottom or self.streaming:
                sb.setValue(sb.maximum())

        def schedule_render(self):
            if not self._render_pending:
                self._render_pending = True
                QTimer.singleShot(80, self.render)

        # --- window behaviour ---------------------------------------------
        def toggle(self):
            if self.isVisible() and self.isActiveWindow():
                self.hide()
                return
            if not self._placed:
                scr = QApplication.primaryScreen().availableGeometry()
                self.move(scr.x() + (scr.width() - self.width()) // 2, scr.y() + int(scr.height() * 0.15))
                self._placed = True
            self.show()
            if hide_from_capture and not self._affinity:
                _exclude_from_capture(int(self.winId()))
                self._affinity = True
            self.raise_()
            self.activateWindow()
            _force_foreground(int(self.winId()))
            self.input.setFocus()
            self._bg(lambda: ("now", requests.get(f"{base}/context/recent", params={"n": 1}, timeout=3).json()))

        def closeEvent(self, e):
            e.ignore()
            self.hide()

        def recall(self):
            if not self.input.text() and self.last_question:
                self.input.setText(self.last_question)

        # --- actions --------------------------------------------------------
        def ask(self, q: str):
            if not self.streaming:
                self.input.setText(q)
                self.submit()

        def insights(self):
            self.ask("How did I spend my time today, and what should I change to get more done?")

        def find_agents(self):
            self.ask("Find agents on Darwin that could help me with what I'm working on right now.")

        def submit(self):
            q = self.input.text().strip()
            if self.streaming:
                return
            if not q:
                self.copy_context()
                return
            self.input.clear()
            self.last_question = q
            now = datetime.now().strftime("%H:%M")
            self.messages.append({"role": "user", "text": q, "time": now})
            self.messages.append({"role": "assistant", "text": "", "steps": [], "sources": [], "streaming": True})
            self._set_streaming(True)
            self.stop_flag.clear()
            self.status.setText("thinking…")
            self.render()
            sid = self.session_id

            def work():
                try:
                    with requests.post(f"{base}/chat", json={"message": q, "session_id": sid}, stream=True,
                                       timeout=(5, 180)) as r:
                        if r.status_code != 200:
                            self.bridge.ev.emit("error", {"text": f"HTTP {r.status_code}: {r.text[:200]}"})
                            return
                        r.encoding = "utf-8"  # SSE has no charset; requests would assume Latin-1
                        for ev, data in parse_sse_lines(r.iter_lines(decode_unicode=True)):
                            if self.stop_flag.is_set():
                                break
                            self.bridge.ev.emit(ev, data)
                except Exception as exc:
                    self.bridge.ev.emit("error", {"text": f"Daemon unreachable at {base}: {exc}"})
                finally:
                    self.bridge.ev.emit("end", {})

            threading.Thread(target=work, daemon=True).start()

        def stop(self):
            self.stop_flag.set()

        def new_chat(self):
            if self.streaming:
                return
            self.messages = []
            self.session_id = None
            _save_state({})
            self.render()
            self.status.setText("new chat")

        def show_recent(self):
            self._bg(lambda: ("recent", requests.get(f"{base}/memory/timeline", params={"minutes": 15},
                                                     timeout=5).json()))

        def copy_answer(self):
            last = next((m["text"] for m in reversed(self.messages) if m["role"] == "assistant" and m["text"]), "")
            QApplication.clipboard().setText(last)
            self.status.setText("answer copied" if last else "nothing to copy")

        def copy_context(self):
            def fetch():  # sanitized pack: secrets/PII scrubbed, email/chat/finance withheld, no raw screen text
                j = requests.get(f"{base}/context/safe", params={"minutes": 30}, timeout=5).json()
                return "context", j
            self._bg(fetch)

        def open_source(self, url):
            sid = url.toString().split(":", 1)[-1]
            self._bg(lambda: ("source", requests.get(f"{base}/memory/source/{sid}", timeout=5).json()))

        # --- events (GUI thread) ------------------------------------------
        def on_event(self, kind: str, data):
            cur = next((m for m in reversed(self.messages) if m["role"] == "assistant"), None)
            if kind == "session":
                self.session_id = data["session_id"]
                _save_state({"session_id": self.session_id})
            elif kind == "status" and cur is not None:
                cur["steps"].append(data["text"])
                self.status.setText(data["text"])
                self.schedule_render()
            elif kind == "token" and cur is not None:
                cur["text"] += data["text"]
                self.schedule_render()
            elif kind == "sources" and cur is not None:
                cur["sources"] = data["items"]
            elif kind == "error":
                if cur is not None and cur.get("streaming"):
                    cur["error"] = data["text"]
                else:
                    self.status.setText(data["text"])
            elif kind == "end":
                if cur is not None:
                    cur["streaming"] = False
                self._set_streaming(False)
                self.status.setText("")
                self.render()
            elif kind == "history":
                self.messages = [{"role": m["role"], "text": m["content"], "time": m.get("time_local", ""),
                                  "steps": [], "sources": []} for m in data.get("messages", [])]
                if not self.messages:
                    self.session_id = None
                self.render()
            elif kind == "now":
                items = data.get("recent_context") or []
                if items:
                    i = items[-1]
                    self.header.setText(f"Now: {i['app']} — {i['window_title']} · {i.get('local_time', '')}")
            elif kind == "recent":
                rows = data.get("items") or []
                lines = [f"- [{r['id']}](aegis:{r['id']}) {r['start_local']}–{r['end_local']} **{r['app']}** "
                         f"{r['title']}: {r['summary']}" for r in rows] or ["- nothing captured in the last 15 min"]
                self.messages.append({"role": "info", "text": "**Last 15 minutes**\n\n" + "\n".join(lines)})
                self.render()
            elif kind == "context":
                self.context = data.get("text", "")
                QApplication.clipboard().setText(self.context)
                held = sum(data.get("withheld", {}).values())
                self.status.setText(f"safe context copied ({data.get('included', 0)} items"
                                    + (f", {held:.0f} min private activity withheld" if held else "") + ")")
            elif kind == "source":
                SourceDialog(self, data.get("id", ""), data.get("text", "")).show()
            elif kind == "bg_error":
                self.status.setText(f"daemon unreachable at {base}")

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    bridge = Bridge()
    hud = HUD(bridge)
    bridge.fire.connect(hud.toggle)
    listener = keyboard.GlobalHotKeys({hotkey: bridge.fire.emit})
    listener.start()
    hud.toggle()  # show once on launch so it's obvious it's running
    sys.exit(app.exec_())
