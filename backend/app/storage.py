"""Минимальное постоянное хранилище учебных сессий (SQLite)."""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .schemas import SessionCreate, SessionCreated


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "clarify.db"


def database_path() -> Path:
    """Путь БД не зависит от папки, из которой запустили сервер."""
    return Path(os.environ.get("CLARIFY_DB_PATH", str(DEFAULT_DB_PATH)))


def create_session(data: SessionCreate) -> SessionCreated:
    session_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database_path()) as connection:
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
