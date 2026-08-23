"""Pluggable LLM backend for the classification tail.

Two providers, one interface:

**Anthropic** is the documented default and the path a production deployment
would take. Razorpay's own Agent Studio is built on the Claude Agent SDK, so it
is also the aligned choice.

**Ollama** runs a small model locally. This is what the measured results in this
repository actually use, and the reasoning is not only that it is free.

The tail is closed-set classification over fifteen labels, on the ~1% of traffic
carrying an error code Razorpay has not published. It is a robustness path, not a
volume path. A frontier model is not obviously the right tool for it, and
asserting that it is -- without measuring -- would be the opposite of engineering
judgement. So the comparison is run for real: a 3B and a 7B local model against
held-out codes with known answers, reported in `classifier_eval`.

If the measurement had shown the small models failing, the honest conclusion
would have been that the tail needs a frontier model and the cost is justified.
The point is that the question was answered with a number rather than a
preference.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import get_settings

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass(frozen=True)
class LLMResponse:
    data: dict[str, Any]
    model: str
    provider: str


class LLMUnavailable(RuntimeError):
    """Backend not reachable or not configured.

    Raised rather than returned so callers cannot accidentally treat an outage as
    a classification. Every caller degrades to the exception list.
    """


class LLMProvider(Protocol):
    name: str

    def complete_json(self, prompt: str, schema: dict, model: str) -> LLMResponse: ...


class AnthropicProvider:
    """Claude via the official SDK. Requires ANTHROPIC_API_KEY."""

    name = "anthropic"

    def complete_json(self, prompt: str, schema: dict, model: str) -> LLMResponse:
        import anthropic

        key = get_settings().anthropic_api_key
        if not key:
            raise LLMUnavailable("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        return LLMResponse(data=json.loads(text), model=model, provider=self.name)


class OllamaProvider:
    """A local model over Ollama's HTTP API.

    `format` is passed the JSON schema itself, so the runtime constrains
    generation rather than the prompt merely asking politely for JSON. A 3B model
    asked nicely for JSON will eventually return prose; a 3B model constrained by
    a schema cannot.
    """

    name = "ollama"

    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host.rstrip("/")

    def complete_json(self, prompt: str, schema: dict, model: str) -> LLMResponse:
        import httpx

        try:
            response = httpx.post(
                f"{self.host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": schema,
                    "stream": False,
                    # Deterministic decoding. A classifier that returns different
                    # causes for the same input on two runs cannot be audited,
                    # and the audit trail is the point.
                    "options": {"temperature": 0.0, "seed": 42},
                },
                timeout=120.0,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - any transport failure is unavailability
            raise LLMUnavailable(f"ollama at {self.host}: {exc}") from exc

        body = response.json()
        raw = body.get("response", "")
        try:
            return LLMResponse(data=json.loads(raw), model=model, provider=self.name)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"ollama returned non-JSON: {raw[:160]!r}") from exc


def get_provider(name: str | None = None) -> LLMProvider:
    """Resolve a provider.

    Defaults to Anthropic when a key is present, otherwise the local model. The
    fallback is deliberate: the system must be runnable by anyone who clones the
    repository without asking them to pay for an API key first.
    """
    if name == "anthropic":
        return AnthropicProvider()
    if name == "ollama":
        return OllamaProvider()
    return AnthropicProvider() if get_settings().anthropic_api_key else OllamaProvider()
