"""Контракт провайдеров без внешней сети и без настоящих ключей."""

import json
import os
import tempfile
import unittest
from fastapi import HTTPException
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from backend.app import llm, main
from backend.app.schemas import Message, MessageCreate, SessionCreate


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, size):
        return self.payload[:size]


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-gemini", "GROQ_API_KEY": "test-groq",
            "GEMINI_MODEL": llm.DEFAULT_GEMINI_MODEL, "GROQ_MODEL": llm.DEFAULT_GROQ_MODEL,
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.input = dict(topic="Algebra", objective="Solve", lesson_notes=None, history=[], question="Why?")

    def test_gemini_first_and_lesson_context(self):
        with patch.object(llm.request, "urlopen", return_value=FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": " Hint "}]}}]}
        )) as send:
            self.assertEqual(llm.generate_reply(**self.input), "Hint")
        req = send.call_args.args[0]
        body = json.loads(req.data)
        self.assertIn(llm.DEFAULT_GEMINI_MODEL, req.full_url)
        self.assertEqual(req.get_header("X-goog-api-key"), "test-gemini")
        self.assertEqual(req.get_header("User-agent"), "Clarify-local/1.0")
        self.assertEqual(body["contents"][0]["role"], "user")
        self.assertIn("Algebra", body["contents"][0]["parts"][0]["text"])
        self.assertIn("Why?", body["contents"][0]["parts"][0]["text"])
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], llm.MAX_OUTPUT_TOKENS)
        self.assertNotIn("test-gemini", req.full_url)

    def test_gemini_rate_limit_uses_groq(self):
        calls = []

        def side_effect(req, timeout):
            calls.append(req)
            if len(calls) == 1:
                raise HTTPError(req.full_url, 429, "Too many", {}, None)
            return FakeResponse({"choices": [{"message": {"content": "Backup hint"}}]})

        with patch.object(llm.request, "urlopen", side_effect=side_effect):
            self.assertEqual(llm.generate_reply(**self.input), "Backup hint")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1].get_header("Authorization"), "Bearer test-groq")
        body = json.loads(calls[1].data)
        self.assertEqual(body["model"], llm.DEFAULT_GROQ_MODEL)
        self.assertEqual(body["max_completion_tokens"], llm.MAX_OUTPUT_TOKENS)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertIn("Algebra", body["messages"][-1]["content"])

    def test_auth_error_does_not_hide_behind_groq(self):
        reqs = []

        def side_effect(req, timeout):
            reqs.append(req)
            raise HTTPError(req.full_url, 401, "Bad key", {}, None)

        with patch.object(llm.request, "urlopen", side_effect=side_effect):
            with self.assertRaises(llm.AIUnavailableError) as error:
                llm.generate_reply(**self.input)
        self.assertEqual(len(reqs), 1)
        self.assertNotIn("test-gemini", str(error.exception))

    def test_gemini_model_not_found_does_not_use_groq(self):
        def not_found(req, timeout):
            raise HTTPError(req.full_url, 404, "Model unavailable", {}, None)

        with patch.object(llm.request, "urlopen", side_effect=not_found) as send:
            with self.assertRaises(llm.AIUnavailableError):
                llm.generate_reply(**self.input)
        self.assertEqual(send.call_count, 1)

    def test_gemini_safety_refusal_does_not_use_groq(self):
        blocked = {"promptFeedback": {"blockReason": "SAFETY"}}
        with patch.object(llm.request, "urlopen", return_value=FakeResponse(blocked)) as send:
            with self.assertRaises(llm.AIUnavailableError):
                llm.generate_reply(**self.input)
        self.assertEqual(send.call_count, 1)

    def test_gemini_candidate_safety_refusal_does_not_use_groq(self):
        blocked = {"candidates": [{"finishReason": "SAFETY"}]}
        with patch.object(llm.request, "urlopen", return_value=FakeResponse(blocked)) as send:
            with self.assertRaises(llm.AIUnavailableError):
                llm.generate_reply(**self.input)
        self.assertEqual(send.call_count, 1)

    def test_gemini_server_error_uses_groq(self):
        calls = []

        def side_effect(req, timeout):
            calls.append(req.full_url)
            if len(calls) == 1:
                raise HTTPError(req.full_url, 503, "Unavailable", {}, None)
            return FakeResponse({"choices": [{"message": {"content": "Groq hint"}}]})

        with patch.object(llm.request, "urlopen", side_effect=side_effect):
            self.assertEqual(llm.generate_reply(**self.input), "Groq hint")
        self.assertEqual(len(calls), 2)

    def test_gemini_token_limit_retries_with_larger_budget(self):
        requests = []

        def side_effect(req, timeout):
            requests.append(json.loads(req.data)["generationConfig"]["maxOutputTokens"])
            if len(requests) == 1:
                return FakeResponse({"candidates": [{"finishReason": "MAX_TOKENS",
                                                    "content": {"parts": [{"text": "Unfinished"}]}}]})
            return FakeResponse({"candidates": [{"finishReason": "STOP",
                                                "content": {"parts": [{"text": "Complete reply."}]}}]})

        with patch.object(llm.request, "urlopen", side_effect=side_effect):
            self.assertEqual(llm.generate_reply(**self.input), "Complete reply.")
        self.assertEqual(requests, [llm.MAX_OUTPUT_TOKENS, llm.RETRY_OUTPUT_TOKENS])

    def test_gemini_token_limit_uses_complete_groq_reply(self):
        calls = []

        def side_effect(req, timeout):
            calls.append(req.full_url)
            if "generativelanguage" in req.full_url:
                return FakeResponse({"candidates": [{"finishReason": "MAX_TOKENS",
                                                    "content": {"parts": [{"text": "Unfinished "}]}}]})
            return FakeResponse({"choices": [{"finish_reason": "stop",
                                               "message": {"content": "Complete reply."}}]})

        with patch.object(llm.request, "urlopen", side_effect=side_effect):
            self.assertEqual(llm.generate_reply(**self.input), "Complete reply.")
        self.assertEqual(len(calls), 3)
        self.assertTrue(all("generativelanguage" in url for url in calls[:2]))
        self.assertIn("groq.com", calls[2])

    def test_groq_token_limit_retries_with_larger_budget(self):
        requests = []

        def side_effect(req, timeout):
            requests.append(json.loads(req.data)["max_completion_tokens"])
            if len(requests) == 1:
                return FakeResponse({"choices": [{"finish_reason": "length",
                                                   "message": {"content": "Unfinished"}}]})
            return FakeResponse({"choices": [{"finish_reason": "stop",
                                               "message": {"content": "Complete reply."}}]})

        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with patch.object(llm.request, "urlopen", side_effect=side_effect):
                self.assertEqual(llm.generate_reply(**self.input), "Complete reply.")
        self.assertEqual(requests, [llm.MAX_OUTPUT_TOKENS, llm.RETRY_OUTPUT_TOKENS])

    def test_groq_token_limit_is_not_saved_as_complete_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"CLARIFY_DB_PATH": str(Path(folder) / "chat.db"),
                                       "GEMINI_API_KEY": "", "GROQ_API_KEY": "test-groq"}):
                session = main.new_session(SessionCreate(topic="Algebra", objective="Learn"))
                with patch.object(llm.request, "urlopen", return_value=FakeResponse(
                    {"choices": [{"finish_reason": "length", "message": {"content": "Unfinished "}}]}
                )):
                    with self.assertRaises(HTTPException) as error:
                        main.send_message(session.id, MessageCreate(action="ask", text="Why?"))
                self.assertEqual(error.exception.status_code, 502)
                self.assertEqual(main.read_session(session.id).messages, [])

    def test_gemini_exhausted_retries_do_not_save_partial_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"CLARIFY_DB_PATH": str(Path(folder) / "chat.db"),
                                       "GROQ_API_KEY": ""}):
                session = main.new_session(SessionCreate(topic="Algebra", objective="Learn"))
                with patch.object(llm.request, "urlopen", return_value=FakeResponse(
                    {"candidates": [{"finishReason": "MAX_TOKENS",
                                    "content": {"parts": [{"text": "Incomplete"}]}}]}
                )) as send:
                    with self.assertRaises(HTTPException) as error:
                        main.send_message(session.id, MessageCreate(action="ask", text="Why?"))
                self.assertEqual(error.exception.status_code, 502)
                self.assertEqual(send.call_count, 2)
                self.assertEqual(main.read_session(session.id).messages, [])

    def test_gemini_unknown_finish_reason_does_not_return_partial_text(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with patch.object(llm.request, "urlopen", return_value=FakeResponse(
                {"candidates": [{"finishReason": "OTHER", "content": {"parts": [{"text": "Partial"}]}}]}
            )):
                with self.assertRaises(llm.AIUnavailableError):
                    llm.generate_reply(**self.input)

    def test_both_down_no_fake_answer(self):
        with patch.object(llm.request, "urlopen", side_effect=URLError("network error")):
            with self.assertRaises(llm.AIUnavailableError):
                llm.generate_reply(**self.input)

    def test_missing_gemini_uses_only_groq(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with patch.object(llm.request, "urlopen", return_value=FakeResponse(
                {"choices": [{"message": {"content": "Groq response"}}]}
            )) as send:
                self.assertEqual(llm.generate_reply(**self.input), "Groq response")
            self.assertIn("groq.com", send.call_args.args[0].full_url)

    def test_groq_namespaced_model_is_allowed_but_invalid_names_are_rejected(self):
        self.assertEqual(llm._model("openai/gpt-oss-20b", allow_namespace=True), "openai/gpt-oss-20b")
        for name in ("openai//gpt-oss-20b", "../secret", "openai/gpt-oss-20b?x=1"):
            with self.assertRaises(llm.ProviderConfigurationError):
                llm._model(name, allow_namespace=True)
        with self.assertRaises(llm.ProviderConfigurationError):
            llm._model("openai/gpt-oss-20b")

    def test_missing_keys_fails_without_network(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GROQ_API_KEY": ""}):
            with patch.object(llm.request, "urlopen") as send:
                with self.assertRaises(llm.AIUnavailableError):
                    llm.generate_reply(**self.input)
                send.assert_not_called()

    def test_ai_reply_reaches_route_and_persists_without_real_key(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"CLARIFY_DB_PATH": str(Path(folder) / "chat.db")}):
                session = main.new_session(SessionCreate(topic="Algebra", objective="Learn"))
                with patch.object(llm.request, "urlopen", return_value=FakeResponse(
                    {"candidates": [{"content": {"parts": [{"text": "Think of brackets."}]}}]}
                )):
                    reply = main.send_message(session.id, MessageCreate(action="ask", text="Why?"))
                self.assertEqual(reply.text, "Think of brackets.")
                self.assertEqual([m.text for m in main.read_session(session.id).messages],
                                 ["Why?", "Think of brackets."])

    def test_prior_messages_keep_roles_for_both_providers(self):
        prior = [
            Message(id="1", role="user", text="Earlier", created_at="t"),
            Message(id="2", role="assistant", text="Earlier reply", created_at="t"),
        ]
        messages = llm._conversation("Math", "Learn", "Private notes", prior, "Again?")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertIn("Private notes", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "Again?")


if __name__ == "__main__":
    unittest.main()
