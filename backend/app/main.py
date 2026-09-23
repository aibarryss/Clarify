"""API Clarify: проверка сервера и создание учебной сессии."""

from fastapi import FastAPI

from .schemas import SessionCreate, SessionCreated
from .storage import create_session

app = FastAPI(title="Clarify API")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Позволяет проверить, что бэкенд запущен."""
    return {"status": "ok"}


@app.post("/api/sessions", response_model=SessionCreated, status_code=201)
def new_session(data: SessionCreate) -> SessionCreated:
    """Сохраняет тему и цель урока; возвращает ID для будущего чата."""
    return create_session(data)
