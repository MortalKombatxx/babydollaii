import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from analysis import (
    MissingAPIKeyError,
    analyze_with_groq,
    as_percent,
    extract_json_object,
    resolve_api_key,
)


SAMPLE_RESULT = {
    "secure_rating": 88,
    "avoidant_rating": 12,
    "toxic_risk_rating": 8,
    "deception_score": 10,
    "flagged_keywords": [],
    "justification": "Owns the mistake and proposes repair.",
    "secure_translation": "I take responsibility and we repair it together.",
    "status": "APPROVED",
}


class FakeCompletions:
    def __init__(self, handler):
        self._handler = handler

    def create(self, **kwargs):
        return self._handler(**kwargs)


class FakeClient:
    def __init__(self, handler):
        self.chat = SimpleNamespace(completions=FakeCompletions(handler))


class ExtractJsonTests(unittest.TestCase):
    def test_parses_raw_object(self):
        self.assertEqual(extract_json_object(json.dumps(SAMPLE_RESULT)), SAMPLE_RESULT)

    def test_parses_fenced_and_prefixed_payloads(self):
        raw = "Reasoning...\n```json\n" + json.dumps(SAMPLE_RESULT) + "\n```"
        self.assertEqual(extract_json_object(raw), SAMPLE_RESULT)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            extract_json_object("")


class PercentTests(unittest.TestCase):
    def test_clamps_and_scales(self):
        self.assertEqual(as_percent(0.42), 42)
        self.assertEqual(as_percent(140), 100)
        self.assertEqual(as_percent("n/a"), 0)


class SecretTests(unittest.TestCase):
    def test_prefers_sidebar_key(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}, clear=False):
            self.assertEqual(resolve_api_key(" sidebar-key ", "GROQ_API_KEY"), "sidebar-key")

    def test_uses_environment_when_sidebar_blank(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "env-key"}, clear=False):
            self.assertEqual(resolve_api_key("  ", "GROQ_API_KEY"), "env-key")


class GroqFallbackTests(unittest.TestCase):
    def test_missing_key(self):
        with self.assertRaises(MissingAPIKeyError):
            analyze_with_groq("hello", "")

    def test_retries_json_object_failure_then_succeeds(self):
        calls = []

        def handler(**kwargs):
            calls.append(kwargs)
            if kwargs.get("response_format"):
                raise RuntimeError(
                    "Error code: 400 - {'error': {'code': 'json_validate_failed'}}"
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(SAMPLE_RESULT)))]
            )

        with patch("groq.Groq", return_value=FakeClient(handler)):
            result = analyze_with_groq("hello", "test-key")

        self.assertEqual(result["status"], "APPROVED")
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn("response_format", calls[0])
        self.assertNotIn("response_format", calls[1])

    def test_skips_decommissioned_model(self):
        calls = []

        def handler(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "llama-3.3-70b-versatile":
                raise RuntimeError("model_decommissioned")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(SAMPLE_RESULT)))]
            )

        with patch("groq.Groq", return_value=FakeClient(handler)):
            result = analyze_with_groq("hello", "test-key")

        self.assertEqual(result["status"], "APPROVED")
        self.assertEqual(calls[0], "llama-3.3-70b-versatile")
        self.assertIn("llama-3.1-8b-instant", calls)


if __name__ == "__main__":
    unittest.main()
