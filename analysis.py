"""Behavioral screening inference used by the Streamlit gateway."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

SYSTEM_PROMPT = """
You are a cynical, world-class forensic psychologist screening for covert narcissism, weaponized therapy-speak, and emotional unavailability.
Do not be fooled by wellness buzzwords. If a user uses psychology terms to avoid accountability, punish them heavily.

Respond with a JSON object only. No markdown. Match this schema:
{
  "secure_rating": 5,
  "avoidant_rating": 98,
  "toxic_risk_rating": 92,
  "deception_score": 95,
  "flagged_keywords": ["protecting my peace", "energetic alignment", "unresolved trauma"],
  "justification": "Expose the exact therapy-speak words they used to deflect accountability.",
  "secure_translation": "How a truly secure partner communicates.",
  "status": "REJECTED"
}
""".strip()

RESULT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "secure_rating": {"type": "number"},
        "avoidant_rating": {"type": "number"},
        "toxic_risk_rating": {"type": "number"},
        "deception_score": {"type": "number"},
        "flagged_keywords": {"type": "array", "items": {"type": "string"}},
        "justification": {"type": "string"},
        "secure_translation": {"type": "string"},
        "status": {"type": "string"},
    },
    "required": [
        "secure_rating",
        "avoidant_rating",
        "toxic_risk_rating",
        "deception_score",
        "flagged_keywords",
        "justification",
        "secure_translation",
        "status",
    ],
    "additionalProperties": False,
}

GROQ_JSON_OBJECT_MODELS = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)
GROQ_STRUCTURED_OUTPUT_MODELS = ("openai/gpt-oss-20b",)
OPENAI_MODEL = "gpt-5.6"
PROVIDER_GROQ = "groq"
PROVIDER_OPENAI = "openai"

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


class MissingAPIKeyError(RuntimeError):
    """Raised when no provider key is available from the sidebar, env, or secrets."""


def resolve_api_key(user_key: str | None, secret_name: str) -> str:
    if user_key and user_key.strip():
        return user_key.strip()

    env_value = os.environ.get(secret_name, "").strip()
    if env_value:
        return env_value

    try:
        import streamlit as st

        value = st.secrets[secret_name]
    except Exception:
        return ""
    return str(value).strip() if value is not None else ""


def extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Model returned an empty response.")

    candidates = [text]
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.insert(0, text[start : end + 1])

    seen: set[str] = set()
    errors: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed
        errors.append("JSON root was not an object.")

    raise ValueError("Model did not return valid JSON. " + "; ".join(errors[:2]))


def as_percent(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if 0 <= number <= 1:
        number *= 100
    return max(0, min(100, int(round(number))))


def _messages(user_input: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]


def _is_json_mode_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "json_validate_failed",
            "failed to validate json",
            "structured_generation_failed",
        )
    )


def _is_missing_model(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in ("model_decommissioned", "model_not_found", "model does not exist")
    )


def _complete_chat(
    create: Callable[..., Any],
    *,
    model: str,
    user_input: str,
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _messages(user_input),
        "temperature": 0.0,
    }
    formats = [response_format] if response_format else [None]
    if response_format:
        formats.append(None)

    last_error: Exception | None = None
    for fmt in formats:
        call_kwargs = dict(kwargs)
        if fmt is not None:
            call_kwargs["response_format"] = fmt
        try:
            response = create(**call_kwargs)
        except Exception as exc:
            last_error = exc
            if fmt is not None and _is_json_mode_failure(exc):
                continue
            raise
        content = ""
        try:
            content = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            continue
        try:
            return extract_json_object(content)
        except ValueError as exc:
            last_error = exc
            continue

    raise last_error or ValueError("Model did not return valid JSON.")


def _groq_formats(model: str) -> list[dict[str, Any] | None]:
    if model in GROQ_STRUCTURED_OUTPUT_MODELS:
        return [
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "behavioral_screen",
                    "strict": True,
                    "schema": RESULT_JSON_SCHEMA,
                },
            },
            {"type": "json_object"},
        ]
    return [{"type": "json_object"}]


def analyze_with_groq(user_input: str, api_key: str) -> dict[str, Any]:
    if not api_key:
        raise MissingAPIKeyError(
            "Groq is not configured. Add GROQ_API_KEY in Streamlit secrets, "
            "or paste a Groq key in the sidebar."
        )

    from groq import Groq

    client = Groq(api_key=api_key)
    models = GROQ_JSON_OBJECT_MODELS + GROQ_STRUCTURED_OUTPUT_MODELS
    last_error: Exception | None = None

    for model in models:
        formats = _groq_formats(model)
        for fmt in formats:
            try:
                return _complete_chat(
                    client.chat.completions.create,
                    model=model,
                    user_input=user_input,
                    response_format=fmt,
                )
            except Exception as exc:
                last_error = exc
                if (
                    _is_json_mode_failure(exc)
                    or _is_missing_model(exc)
                    or isinstance(exc, ValueError)
                ):
                    continue
                raise

    raise last_error or RuntimeError("Groq analysis failed.")


def analyze_with_openai(user_input: str, api_key: str) -> dict[str, Any]:
    if not api_key:
        raise MissingAPIKeyError(
            "OpenAI is not configured. Add OPENAI_API_KEY in Streamlit secrets, "
            "or paste an OpenAI key in the sidebar."
        )

    import openai

    client = openai.OpenAI(api_key=api_key)
    try:
        return _complete_chat(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            user_input=user_input,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        if "temperature" in str(exc).lower():
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=_messages(user_input),
                response_format={"type": "json_object"},
            )
            return extract_json_object(response.choices[0].message.content or "")
        raise


def analyze_text(user_input: str, *, provider: str, api_key: str) -> dict[str, Any]:
    if provider == PROVIDER_OPENAI:
        return analyze_with_openai(user_input, api_key)
    return analyze_with_groq(user_input, api_key)
