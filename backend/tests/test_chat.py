"""Быстрые проверки API без внешних сервисов и без реального AI-ключа."""

import asyncio
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app import llm, main
from backend.app.llm import AIUnavailableError
from backend.app.schemas import MessageCreate, SessionCreate


class DotEnvTests(unittest.TestCase):
    """Проверка загрузки ключей без доступа к настоящему локальному .env."""

    def test_project_env_is_loaded_on_app_import_and_overrides_inherited_values(self):
        with tempfile.TemporaryDirectory() as folder:
            env_path = Path(folder) / ".env"
            env_path.write_text("GEMINI_API_KEY=from-file\nGROQ_API_KEY=from-file\n", encoding="utf-8")
            original_env = os.environ.copy()
            try:
                os.environ.pop("GEMINI_API_KEY", None)
                os.environ["GROQ_API_KEY"] = "from-process"
                def load_fixture(path, override):
                    self.assertEqual(path, Path(main.__file__).resolve().parents[2] / ".env")
                    return load_dotenv(env_path, override=override)

                with patch("dotenv.load_dotenv", side_effect=load_fixture) as loader:
                    importlib.reload(main)
                    loader.assert_called_once_with(Path(main.__file__).resolve().parents[2] / ".env", override=True)
                self.assertEqual(os.environ["GEMINI_API_KEY"], "from-file")
                self.assertEqual(os.environ["GROQ_API_KEY"], "from-file")
            finally:
                os.environ.clear()
                os.environ.update(original_env)
                with patch("dotenv.load_dotenv", return_value=False):
                    importlib.reload(main)


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = str(Path(self.folder.name) / "clarify.db")
        self.database_setting = patch.dict(os.environ, {
            "CLARIFY_DB_PATH": self.path,
            "GEMINI_API_KEY": "",
            "GROQ_API_KEY": "",
        })
        self.database_setting.start()
        self.addCleanup(self.database_setting.stop)
        self.addCleanup(self.folder.cleanup)
        self.session = main.new_session(
            SessionCreate(topic=" Math ", objective=" Learn brackets ", lesson_notes="Teacher note")
        )

    def test_create_and_read_session(self):
        detail = main.read_session(self.session.id)
        self.assertEqual(detail.topic, "Math")
        self.assertEqual(detail.objective, "Learn brackets")
        self.assertTrue(detail.has_lesson_notes)
        self.assertEqual(detail.messages, [])
        self.assertNotIn("lesson_notes", detail.model_dump())
        with self.assertRaises(HTTPException) as error:
            main.read_session("unknown")
        self.assertEqual(error.exception.status_code, 404)

    def test_invalid_input(self):
        with self.assertRaises(ValidationError):
            SessionCreate(topic=" ", objective="Learn")
        with self.assertRaises(ValidationError):
            MessageCreate(action="ask", text="   ")
        with self.assertRaises(ValidationError):
            MessageCreate(action="example", text="extra text")

    def test_no_fake_answer_and_no_partial_save(self):
        with self.assertRaises(HTTPException) as error:
            main.send_message(self.session.id, MessageCreate(action="ask", text="Why?"))
        self.assertEqual(error.exception.status_code, 502)
        self.assertEqual(main.read_session(self.session.id).messages, [])
        with self.assertRaises(HTTPException) as error:
            main.send_message(self.session.id, MessageCreate(action="simplify"))
        self.assertEqual(error.exception.status_code, 409)
        with self.assertRaises(HTTPException) as error:
            main.send_message("unknown", MessageCreate(action="ask", text="Why?"))
        self.assertEqual(error.exception.status_code, 404)

    def test_successful_exchange_and_restored_history(self):
        with patch.object(main.llm, "generate_reply", return_value="  Hint?  ") as generate:
            reply = main.send_message(self.session.id, MessageCreate(action="ask", text=" Why? "))
            self.assertEqual(generate.call_args.kwargs["lesson_notes"], "Teacher note")
            self.assertEqual(generate.call_args.kwargs["question"], "Why?")
            self.assertEqual(reply.text, "Hint?")
            main.send_message(self.session.id, MessageCreate(action="simplify"))
            self.assertEqual(generate.call_args.kwargs["history"][-1].text, "Hint?")
        history = main.read_session(self.session.id).messages
        self.assertEqual([m.role for m in history], ["user", "assistant", "user", "assistant"])
        self.assertEqual(history[0].text, "Why?")
        self.assertEqual(history[2].text, "Объясни предыдущий ответ проще")
        self.assertTrue(all(m.id and m.created_at for m in history))

    def test_failure_after_success_keeps_only_complete_exchange(self):
        with patch.object(main.llm, "generate_reply", return_value="First hint"):
            main.send_message(self.session.id, MessageCreate(action="ask", text="First question"))
        with patch.object(main.llm, "generate_reply", side_effect=AIUnavailableError("secret details")):
            with self.assertRaises(HTTPException) as error:
                main.send_message(self.session.id, MessageCreate(action="ask", text="Second question"))
        self.assertEqual(error.exception.status_code, 502)
        self.assertNotIn("secret details", error.exception.detail)
        self.assertEqual(len(main.read_session(self.session.id).messages), 2)


class HTTPContractTests(unittest.TestCase):
    """Проверка настоящего ASGI-контракта без httpx и без внешних запросов."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.env = patch.dict(os.environ, {
            "CLARIFY_DB_PATH": str(Path(self.folder.name) / "http.db"),
            "CLARIFY_CORS_ORIGINS": "http://localhost:5500",
            "GEMINI_API_KEY": "",
            "GROQ_API_KEY": "",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        with patch("dotenv.load_dotenv", return_value=False):
            importlib.reload(main)
        self.addCleanup(self.restore_main_without_dotenv)

    @staticmethod
    def restore_main_without_dotenv():
        with patch("dotenv.load_dotenv", return_value=False):
            importlib.reload(main)

    def request(self, method, path, payload=None, headers=None):
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        request_headers = [(k.lower().encode("ascii"), v.encode("ascii")) for k, v in (headers or {}).items()]
        if payload is not None:
            request_headers.append((b"content-type", b"application/json"))

        async def call():
            sent = []
            received = False

            async def receive():
                nonlocal received
                if not received:
                    received = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return {"type": "http.disconnect"}

            async def send(message):
                sent.append(message)

            await main.app({
                "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                "method": method, "path": path, "raw_path": path.encode("utf-8"),
                "query_string": b"", "root_path": "", "scheme": "http",
                "server": ("localhost", 8000), "client": ("127.0.0.1", 12345),
                "headers": request_headers,
            }, receive, send)
            status = next(m["status"] for m in sent if m["type"] == "http.response.start")
            response_headers = dict(next(m["headers"] for m in sent if m["type"] == "http.response.start"))
            result = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
            if result and b"application/json" in response_headers.get(b"content-type", b""):
                return status, response_headers, json.loads(result)
            return status, response_headers, result

        return asyncio.run(call())

    def test_docs_and_cors_preflight(self):
        status, _, _ = self.request("GET", "/docs")
        self.assertEqual(status, 200)
        status, headers, _ = self.request("GET", "/")
        self.assertEqual(status, 307)
        self.assertEqual(headers[b"location"], b"/app/code.html")
        status, headers, page = self.request("GET", "/app/code.html")
        self.assertEqual(status, 200)
        self.assertIn(b"text/html", headers[b"content-type"])
        self.assertIn(b"const API = '/api'", page)
        self.assertIn(b"renderAssistantMessage(bubble, message.text)", page)
        self.assertIn(b"renderMathInElement(bubble", page)
        self.assertIn(b"strong.textContent = match[1]", page)
        status, headers, _ = self.request("OPTIONS", "/api/sessions", headers={
            "Origin": "http://localhost:5500", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertEqual(status, 200)
        self.assertEqual(headers[b"access-control-allow-origin"], b"http://localhost:5500")
        status, headers, _ = self.request("OPTIONS", "/api/sessions", headers={
            "Origin": "http://untrusted.example", "Access-Control-Request-Method": "POST",
        })
        self.assertEqual(status, 400)
        self.assertNotIn(b"access-control-allow-origin", headers)

    def test_invalid_cors_origin_is_rejected(self):
        with patch("dotenv.load_dotenv", return_value=False):
            with patch.dict(os.environ, {"CLARIFY_CORS_ORIGINS": "*"}):
                with self.assertRaises(ValueError):
                    importlib.reload(main)
            with patch.dict(os.environ, {"CLARIFY_CORS_ORIGINS": "file:///tmp/page.html"}):
                with self.assertRaises(ValueError):
                    importlib.reload(main)
            importlib.reload(main)

    def test_http_session_validation_and_no_provider(self):
        status, _, created = self.request("POST", "/api/sessions", {
            "topic": "Алгебра", "objective": "Раскрыть скобки",
        })
        self.assertEqual(status, 201)
        session_id = created["id"]
        status, _, validation = self.request("POST", f"/api/sessions/{session_id}/messages", {
            "action": "ask", "text": " ",
        })
        self.assertEqual(status, 422)
        self.assertIsInstance(validation["detail"], list)
        self.assertTrue(all(item.get("msg") for item in validation["detail"]))
        status, _, error = self.request("POST", f"/api/sessions/{session_id}/messages", {
            "action": "ask", "text": "Почему 2(x+3) = 2x+6?",
        })
        self.assertEqual(status, 502)
        self.assertEqual(error["detail"], "AI временно недоступен")
        status, _, restored = self.request("GET", f"/api/sessions/{session_id}")
        self.assertEqual(status, 200)
        self.assertEqual(restored["messages"], [])

    def test_http_fallback_and_both_down(self):
        class Response:
            def __init__(self, payload):
                self.body = json.dumps(payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                return self.body[:size]

        status, _, created = self.request("POST", "/api/sessions", {
            "topic": "Алгебра", "objective": "Раскрыть скобки",
        })
        self.assertEqual(status, 201)
        path = f"/api/sessions/{created['id']}"
        calls = []

        def fallback(req, timeout):
            calls.append(req.full_url)
            if "generativelanguage" in req.full_url:
                raise HTTPError(req.full_url, 429, "Rate limit", {}, None)
            return Response({"choices": [{"message": {"content": "Умножь оба слагаемых."}}]})

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test", "GROQ_API_KEY": "test"}):
            with patch.object(llm.request, "urlopen", side_effect=fallback):
                status, _, reply = self.request("POST", path + "/messages", {
                    "action": "ask", "text": "Как раскрыть скобки?",
                })
            self.assertEqual(status, 200)
            self.assertEqual(reply["role"], "assistant")
            self.assertEqual(reply["text"], "Умножь оба слагаемых.")
            self.assertTrue(reply["id"] and reply["created_at"])
            self.assertEqual(len(calls), 2)
            with patch.object(llm.request, "urlopen", side_effect=URLError("down")):
                status, _, _ = self.request("POST", path + "/messages", {
                    "action": "ask", "text": "И дальше?",
                })
        self.assertEqual(status, 502)
        status, _, restored = self.request("GET", path)
        self.assertEqual(status, 200)
        self.assertEqual([item["text"] for item in restored["messages"]], [
            "Как раскрыть скобки?", "Умножь оба слагаемых.",
        ])


if __name__ == "__main__":
    unittest.main()
