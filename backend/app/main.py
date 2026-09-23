"""API Clarify: учебные сессии и диалог с AI."""

import os
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import llm
from .schemas import Message, MessageCreate, SessionCreate, SessionCreated, SessionDetail
from .storage import create_session, get_lesson_notes, get_session, save_exchange

app = FastAPI(title="Clarify API")

# Только явно заданные источники HTML-клиента. Ключи остаются на бэкенде.
origins = []
for value in os.getenv("CLARIFY_CORS_ORIGINS", "").split(","):
    origin = value.strip().rstrip("/")
    if not origin:
        continue
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("CLARIFY_CORS_ORIGINS должен содержать только HTTP(S) Origin без пути")
    origins.append(origin)
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    """Позволяет проверить, что бэкенд запущен."""
    return {"status": "ok"}


@app.post("/api/sessions", response_model=SessionCreated, status_code=201)
def new_session(data: SessionCreate) -> SessionCreated:
    """Сохраняет тему и цель урока; возвращает ID для будущего чата."""
    return create_session(data)


@app.get("/api/sessions/{session_id}", response_model=SessionDetail)
def read_session(session_id: str) -> SessionDetail:
    """Возвращает урок по ID, например после обновления страницы."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Учебная сессия не найдена")
    return session


@app.post("/api/sessions/{session_id}/messages", response_model=Message)
def send_message(session_id: str, data: MessageCreate) -> Message:
    """Получает ответ AI и сохраняет обмен целиком — если AI недоступен, не сохраняет."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Учебная сессия не найдена")

    if data.action == "ask":
        question = data.text  # MessageCreate гарантирует непустой текст.
    else:
        if not session.messages or session.messages[-1].role != "assistant":
            raise HTTPException(status_code=409, detail="Сначала задайте вопрос")
        question = {
            "simplify": "Объясни предыдущий ответ проще",
            "example": "Приведи пример к предыдущему ответу",
        }[data.action]

    try:
        answer = llm.generate_reply(
            topic=session.topic,
            objective=session.objective,
            lesson_notes=get_lesson_notes(session_id),
            history=session.messages[-12:],
            question=question,
        )
    except llm.AIUnavailableError as exc:
        raise HTTPException(status_code=502, detail="AI временно недоступен") from exc
    if not answer or not answer.strip():
        raise HTTPException(status_code=502, detail="AI не вернул ответ")
    return save_exchange(session_id, question, answer.strip())


# HTML и API живут на одном локальном адресе; секреты и SQLite здесь не раздаются.
FRONT_DIR = Path(__file__).resolve().parents[2] / "front"
app.mount("/app", StaticFiles(directory=FRONT_DIR, html=True), name="frontend")


@app.get("/", include_in_schema=False)
def frontend() -> RedirectResponse:
    return RedirectResponse(url="/app/code.html")
