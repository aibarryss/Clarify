"""Минимальное постоянное хранилище учебных сессий (SQLite)."""

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .schemas import Message, SessionCreate, SessionCreated, SessionDetail


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "clarify.db"


def database_path() -> Path:
    """Путь БД не зависит от папки, из которой запустили сервер."""
    return Path(os.environ.get("CLARIFY_DB_PATH", str(DEFAULT_DB_PATH)))


def init_messages_table(connection: sqlite3.Connection) -> None:
    # Существующая таблица sessions не меняется; новая создаётся при первом чтении/записи.
    connection.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )"""
    )


def create_session(data: SessionCreate) -> SessionCreated:
    session_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(database_path())) as connection, connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                objective TEXT NOT NULL,
                lesson_notes TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """INSERT INTO sessions (id, topic, objective, lesson_notes, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, data.topic, data.objective, data.lesson_notes, created_at),
        )
    return SessionCreated(
        id=session_id,
        topic=data.topic,
        objective=data.objective,
        has_lesson_notes=bool(data.lesson_notes),
    )


def get_session(session_id: str) -> SessionDetail | None:
    """Читает сохранённый урок без передачи клиенту заметок учителя/ученика."""
    db = database_path()
    if not db.exists():
        return None
    with closing(sqlite3.connect(db)) as connection, connection:
        try:
            row = connection.execute(
                "SELECT id, topic, objective, lesson_notes FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            init_messages_table(connection)
            history = connection.execute(
                """SELECT id, role, text, created_at FROM messages
                   WHERE session_id = ? ORDER BY sequence""",
                (session_id,),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # Неверная схема БД — ошибка сервера, а не «урок не найден».
            raise RuntimeError("Session database is not initialized correctly") from exc
    return SessionDetail(
        id=row[0],
        topic=row[1],
        objective=row[2],
        has_lesson_notes=bool(row[3]),
        messages=[Message(id=m[0], role=m[1], text=m[2], created_at=m[3]) for m in history],
    )


def get_lesson_notes(session_id: str) -> str | None:
    """Заметки только для серверного контекста AI; в JSON клиенту не передаются."""
    with closing(sqlite3.connect(database_path())) as connection, connection:
        row = connection.execute(
            "SELECT lesson_notes FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return row[0] if row else None


def save_exchange(session_id: str, user_text: str, assistant_text: str) -> Message:
    """Записывает вопрос и ответ одной транзакцией после успешного вызова AI."""
    now = datetime.now(timezone.utc).isoformat()
    reply = Message(id=str(uuid4()), role="assistant", text=assistant_text, created_at=now)
    with closing(sqlite3.connect(database_path())) as connection, connection:
        connection.execute("PRAGMA foreign_keys = ON")
        init_messages_table(connection)
        connection.executemany(
            """INSERT INTO messages (id, session_id, role, text, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (str(uuid4()), session_id, "user", user_text, now),
                (reply.id, session_id, reply.role, reply.text, reply.created_at),
            ],
        )
    return reply
