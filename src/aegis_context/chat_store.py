"""Persistent chat sessions (conversation memory) in the same SQLite file."""
from __future__ import annotations

import uuid

from .timeutil import db_fmt, fmt_local, utc_now


class ChatStore:
    def __init__(self, storage):
        self.st = storage

    def _x(self, sql: str, args=()):
        with self.st._lock:
            cur = self.st.conn.execute(sql, args)
            self.st.conn.commit()
            return cur

    def create_session(self) -> str:
        sid = uuid.uuid4().hex
        now = db_fmt(utc_now())
        self._x("INSERT INTO chat_sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)", (sid, now, now))
        return sid

    def exists(self, session_id: str | None) -> bool:
        return bool(session_id) and bool(
            self._x("SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone())

    def ensure(self, session_id: str | None) -> str:
        return session_id if self.exists(session_id) else self.create_session()

    def list_sessions(self, limit: int = 20) -> list[dict]:
        rows = self._x("SELECT session_id, title, updated_at FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
                       (limit,)).fetchall()
        return [{"session_id": r[0], "title": r[1], "updated_local": fmt_local(r[2], date=True)} for r in rows]

    def get_messages(self, session_id: str) -> list[dict]:
        rows = self._x("SELECT message_id, role, content, memo, created_at FROM chat_messages WHERE session_id = ? "
                       "ORDER BY message_id", (session_id,)).fetchall()
        return [{"message_id": r[0], "role": r[1], "content": r[2], "memo": r[3], "created_at": r[4],
                 "time_local": fmt_local(r[4])} for r in rows]

    def add_message(self, session_id: str, role: str, content: str, memo: str = "") -> int:
        now = db_fmt(utc_now())
        cur = self._x("INSERT INTO chat_messages (session_id, role, content, memo, created_at) VALUES (?, ?, ?, ?, ?)",
                      (session_id, role, content, memo, now))
        self._x("UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        if role == "user":
            self._x("UPDATE chat_sessions SET title = ? WHERE session_id = ? AND title = ''",
                    (content.strip().replace("\n", " ")[:60], session_id))
        return cur.lastrowid

    def get_summary(self, session_id: str) -> str:
        r = self._x("SELECT summary FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
        return r[0] if r else ""

    def set_summary(self, session_id: str, text: str) -> None:
        self._x("UPDATE chat_sessions SET summary = ? WHERE session_id = ?", (text, session_id))

    def set_title(self, session_id: str, text: str) -> None:
        self._x("UPDATE chat_sessions SET title = ? WHERE session_id = ?", (text[:60], session_id))

    def delete_session(self, session_id: str) -> None:
        self._x("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        self._x("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
