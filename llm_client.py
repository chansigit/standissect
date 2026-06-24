"""Self-contained, zero-dependency structured-LLM client.

VENDORED SHARED MODULE — canonical source:
    stanmetacols/src/stanmetacols/llm_client.py
Keep copies in sibling repos (e.g. standissect) byte-identical.

Standard library only (json, urllib). No third-party import — not even pydantic.
`call_structured` takes a `parse` callable so each caller chooses its own
validation (pydantic `Model.model_validate`, a dataclass builder, a plain
function, …).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class LLMUnavailable(Exception):
    """The structured LLM call could not be completed (no key, no network, HTTP
    error, empty/garbled reply, or validation failure). Callers with a
    deterministic fallback catch this."""


def extract_json(text):
    """Return the outermost JSON object/array in `text`, tolerating Markdown
    code fences and surrounding prose. Raise LLMUnavailable if none found."""
    if not text or not str(text).strip():
        raise LLMUnavailable("empty model response")
    t = str(text).strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        raise LLMUnavailable("no JSON found in model response")
    start = min(starts)
    close = "}" if t[start] == "{" else "]"
    end = t.rfind(close)
    if end < start:
        raise LLMUnavailable("no JSON found in model response")
    return t[start:end + 1]


def _content_text(message):
    """Assistant text from an OpenAI-style message — tolerate plain-string
    content or a list of {type:text, text:...} parts."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


class OpenAICompatClient:
    """Minimal stdlib client for any OpenAI-compatible /chat/completions
    endpoint (OpenAI, Volcengine ARK, DeepSeek, vLLM, Ollama, …). No SDK."""

    def __init__(self, base_url, api_key, model, *,
                 timeout=60.0, temperature=None, max_tokens=None):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _url(self):
        url = (self.base_url or "").rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        return url

    def complete(self, system, user):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(), data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMUnavailable(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc
        try:
            return _content_text(data["choices"][0]["message"])
        except Exception as exc:
            raise LLMUnavailable(f"unexpected response shape: {exc}") from exc


def call_structured(client, system, user, parse, *, list_key=None):
    """Drive `client.complete(system, user) -> str`, extract + load JSON, and
    return `parse(data)`. Any failure raises LLMUnavailable.

    `parse` is a callable `dict|list -> T` (pass `Model.model_validate` for
    pydantic, or any function). `list_key` wraps a bare top-level array as
    `{list_key: [...]}` before parsing."""
    try:
        text = client.complete(system, user)
    except LLMUnavailable:
        raise
    except Exception as exc:
        raise LLMUnavailable(str(exc)) from exc
    blob = extract_json(text)
    try:
        data = json.loads(blob)
    except Exception as exc:
        raise LLMUnavailable(f"response is not valid JSON: {exc}") from exc
    if isinstance(data, list) and list_key is not None:
        data = {list_key: data}
    try:
        return parse(data)
    except LLMUnavailable:
        raise
    except Exception as exc:
        raise LLMUnavailable(f"response does not match schema: {exc}") from exc
