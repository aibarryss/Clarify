"""Формат данных, которые принимает и возвращает API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SessionCreate(BaseModel):
    topic: str = Field(max_length=120)
    objective: str = Field(max_length=300)
    lesson_notes: str | None = Field(default=None, max_length=2000)

    @field_validator("topic", "objective")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Поле не может быть пустым")
        return value

    @field_validator("lesson_notes")
    @classmethod
    def optional_text(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class SessionCreated(BaseModel):
    id: str
    topic: str
    objective: str
    has_lesson_notes: bool


class MessageCreate(BaseModel):
    action: Literal["ask", "simplify", "example"]
    text: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def check_text(self) -> "MessageCreate":
        if self.action == "ask":
            if not self.text or not self.text.strip():
                raise ValueError("Для вопроса нужен текст")
            self.text = self.text.strip()
        elif self.text is not None:
            if self.text.strip():
                raise ValueError("Для этого действия не нужно поле text")
            self.text = None
        return self


class Message(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    text: str
    created_at: str


class SessionDetail(SessionCreated):
    messages: list[Message]
