"""Формат данных, которые принимает и возвращает API."""

from pydantic import BaseModel, Field, field_validator


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
