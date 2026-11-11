"""
core/chat.py — model-agnostic chat backend for the Scraping Studio.

A thin, streaming chat layer with a **model palette**: it auto-detects whatever
is reachable and lets the UI offer a choice.

  - **Claude** (Anthropic API) — first-class when an API key is available
    (``ANTHROPIC_API_KEY`` / ``LLM-API-KEY`` env, or a key pasted into the UI and
    passed through ``api_key``). Lazy-imports the ``anthropic`` SDK.
  - **Ollama** (local) — every locally-pulled *chat* model (embedding-only models
    like ``nomic-embed-text`` are filtered out). Works offline, no key.

Both providers expose the same streaming generator (``stream_chat``) so the UI can
render tokens live with ``st.write_stream``. ``complete_chat`` is the buffered
convenience wrapper. Everything degrades gracefully: a missing SDK / key / daemon
surfaces as a readable error string rather than an exception in the UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator

import requests

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# A couple of current Claude ids worth offering when a key is present.
CLAUDE_MODELS = [
    ("claude-opus-4-8", "Claude Opus 4.8"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
]

DEFAULT_SYSTEM = (
    "You are MIRA's data-acquisition copilot, embedded in a scraping & data-"
    "transformation studio. Help the user find data sources, compile and clean "
    "URLs, plan scrapes, and shape raw data for an ML pipeline. Be concise and "
    "practical; prefer concrete URLs, selectors, and pandas/SQL snippets. When the "
    "user pastes data or candidate sources, reason about schema, target, leakage, "
    "and preprocessing."
)


@dataclass(frozen=True)
class ChatModel:
    id: str            # stable widget id, e.g. "anthropic:claude-opus-4-8"
    provider: str      # "anthropic" | "ollama"
    model: str         # model name passed to the provider
    label: str         # human label for the palette
    icon: str          # emoji
    ready: bool        # usable right now (key present / daemon up)
    note: str = ""     # why it's not ready, if applicable


# --- key handling -------------------------------------------------------------
def env_api_key() -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM-API-KEY")
    if key:
        return key
    try:
        from core.llm import _llm_api_key  # reuse mira's resolver
        return _llm_api_key()
    except Exception:
        return None


def _anthropic_sdk() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


# --- model discovery ----------------------------------------------------------
def list_ollama_models(timeout: float = 2.0) -> list[str]:
    """Locally-pulled chat models (embedding-only models excluded). [] if down."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        return []
    out = []
    for n in names:
        if not n or "embed" in n.lower():
            continue
        out.append(n)
    return sorted(out)


def available_models(*, runtime_key: str | None = None) -> list[ChatModel]:
    """Build the palette. Claude entries first (ready iff key+SDK), then Ollama."""
    models: list[ChatModel] = []
    has_key = bool(runtime_key or env_api_key())
    has_sdk = _anthropic_sdk()
    claude_ready = has_key and has_sdk
    claude_note = ""
    if not has_sdk:
        claude_note = "pip install anthropic"
    elif not has_key:
        claude_note = "add an API key"
    for mid, label in CLAUDE_MODELS:
        models.append(ChatModel(
            id=f"anthropic:{mid}", provider="anthropic", model=mid,
            label=label, icon="🧠", ready=claude_ready, note=claude_note,
        ))

    for name in list_ollama_models():
        models.append(ChatModel(
            id=f"ollama:{name}", provider="ollama", model=name,
            label=f"{name}  (local)", icon="💻", ready=True,
        ))
    return models


def pick_default(models: list[ChatModel]) -> ChatModel | None:
    """Prefer a ready Claude model, else any ready model, else the first listed."""
    for m in models:
        if m.provider == "anthropic" and m.ready:
            return m
    for m in models:
        if m.ready:
            return m
    return models[0] if models else None


# --- streaming chat -----------------------------------------------------------
def stream_chat(
    messages: list[dict[str, str]],
    *,
    provider: str,
    model: str,
    api_key: str | None = None,
    system: str | None = None,
    max_tokens: int = 1500,
    timeout: int = 120,
) -> Iterator[str]:
    """Yield assistant text chunks. Errors are yielded as a readable string."""
    sys_prompt = system or DEFAULT_SYSTEM
    try:
        if provider == "anthropic":
            yield from _stream_anthropic(messages, model, api_key, sys_prompt, max_tokens)
        elif provider == "ollama":
            yield from _stream_ollama(messages, model, sys_prompt, timeout)
        else:
            yield f"_(unknown provider: {provider})_"
    except Exception as exc:  # noqa: BLE001 — surface, don't crash the UI
        yield f"\n\n⚠️ **{provider} error:** {exc}"


def _stream_anthropic(messages, model, api_key, system, max_tokens) -> Iterator[str]:
    key = api_key or env_api_key()
    if not key:
        yield "⚠️ No Anthropic API key. Paste one in the palette or set ANTHROPIC_API_KEY."
        return
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    # Anthropic takes system separately; messages must be user/assistant only.
    conv = [m for m in messages if m.get("role") in ("user", "assistant")]
    with client.messages.stream(
        model=model, max_tokens=max_tokens, system=system, messages=conv,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _stream_ollama(messages, model, system, timeout) -> Iterator[str]:
    import json

    conv = [{"role": "system", "content": system}] + [
        m for m in messages if m.get("role") in ("user", "assistant", "system")
    ]
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": model, "messages": conv, "stream": True,
              "options": {"temperature": 0.3}},
        stream=True, timeout=timeout,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        chunk = (obj.get("message") or {}).get("content", "")
        if chunk:
            yield chunk
        if obj.get("done"):
            break


def complete_chat(messages: list[dict[str, str]], **kw: Any) -> str:
    """Buffered (non-streaming) convenience wrapper."""
    return "".join(stream_chat(messages, **kw))
