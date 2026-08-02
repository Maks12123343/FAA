"""Shared OpenAI-compatible API client."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def _clean_for_json(value):
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, list):
        return [_clean_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean_for_json(item) for key, item in value.items()}
    return value


def _content_to_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("value")
                    or ""
                )
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    if isinstance(value, dict):
        text = value.get("text") or value.get("content") or value.get("value") or ""
        return text.strip() if isinstance(text, str) else ""
    return ""


def _extract_chat_text(data: dict) -> tuple[str, str]:
    if not isinstance(data, dict):
        raise RuntimeError(f"chat response was not a JSON object: {type(data).__name__}")
    if data.get("error"):
        raise RuntimeError(f"chat response error: {data.get('error')}")

    choices = data.get("choices") or []
    if choices:
        choice = choices[0] or {}
        message = choice.get("message") or {}
        delta = choice.get("delta") or {}
        text = (
            _content_to_text(message.get("content"))
            or _content_to_text(delta.get("content"))
            or _content_to_text(choice.get("text"))
        )
        finish = choice.get("finish_reason") or choice.get("native_finish_reason") or data.get("status") or "stop"
        if text:
            return text, "max_tokens" if finish == "length" else finish

        choice_keys = sorted(choice.keys()) if isinstance(choice, dict) else []
        message_keys = sorted(message.keys()) if isinstance(message, dict) else []
        raise RuntimeError(
            "chat response contained no assistant content "
            f"(finish={finish!r}, choice_keys={choice_keys}, message_keys={message_keys})"
        )

    text = _content_to_text(data.get("output_text")) or _content_to_text(data.get("content"))
    if text:
        finish = data.get("status") or "stop"
        return text, "max_tokens" if finish == "incomplete" else finish

    raise RuntimeError(f"chat response contained no choices/content; keys={sorted(data.keys())}")


def _parse_sse_chat_response(raw: str) -> dict:
    content_parts = []
    finish = None
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            item = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in item.get("choices") or []:
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            text = (
                _content_to_text(delta.get("content"))
                or _content_to_text(message.get("content"))
                or _content_to_text(choice.get("text"))
            )
            if text:
                content_parts.append(text)
            finish = choice.get("finish_reason") or choice.get("native_finish_reason") or finish
    text = "".join(content_parts).strip()
    if not text:
        raise RuntimeError("SSE response contained no assistant content")
    return {"choices": [{"message": {"content": text}, "finish_reason": finish or "stop"}]}


def _responses_url(api_url: str) -> str:
    api_url = (api_url or "").rstrip("/")
    if api_url.endswith("/chat/completions"):
        return api_url[: -len("/chat/completions")] + "/responses"
    if api_url.endswith("/v1"):
        return api_url + "/responses"
    return "https://a6api.com/v1/responses"


def _rewrite_settings() -> tuple[str, str, str, str, str]:
    settings = config.load_settings()
    api_key = (settings.get("rewrite_api_key", "") or os.environ.get("REWRITE_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("No rewrite API key configured in Settings.")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError("Rewrite API key must be the real ASCII API key, not a placeholder.") from exc

    api_url = (
        settings.get("rewrite_api_url")
        or os.environ.get("REWRITE_API_URL")
        or "https://a6api.com/v1/chat/completions"
    )
    model = (
        settings.get("rewrite_model")
        or os.environ.get("REWRITE_MODEL")
        or "gpt-5.5"
    )
    reasoning = (
        settings.get("rewrite_reasoning_effort")
        or os.environ.get("REWRITE_REASONING_EFFORT")
        or "high"
    )
    if str(reasoning).strip().lower() in ("", "none", "off", "false", "0"):
        reasoning = ""
    max_tokens_raw = settings.get("rewrite_max_tokens") or os.environ.get("REWRITE_MAX_TOKENS") or "12000"
    return api_url, api_key, model, reasoning, max_tokens_raw


def _messages_to_responses_input(system: str, messages: list) -> str:
    blocks = []
    if system:
        blocks.append("SYSTEM INSTRUCTIONS:\n" + system)
    for msg in messages or []:
        role = str(msg.get("role", "user")).upper()
        content = msg.get("content", "")
        blocks.append(f"{role}:\n{content}")
    return "\n\n---\n\n".join(blocks)


def _extract_responses_text(data: dict) -> tuple[str, str]:
    parts = []
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            text = content.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    for item in data.get("content") or []:
        text = item.get("text") if isinstance(item, dict) else None
        if isinstance(text, str) and text:
            parts.append(text)
    text = "".join(parts).strip()
    if not text and isinstance(data.get("output_text"), str):
        text = data["output_text"].strip()
    if not text:
        raise RuntimeError("Responses API returned no text")
    finish = data.get("status") or "stop"
    return text, "max_tokens" if finish == "incomplete" else finish


def _call_rewrite_responses(
    api_url: str,
    api_key: str,
    model: str,
    system: str,
    messages: list,
    timeout: int,
    max_tokens_raw,
    reasoning_effort: str,
) -> tuple[str, str]:
    import requests

    payload = {
        "model": model,
        "input": _messages_to_responses_input(system, messages),
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    try:
        max_tokens = int(max_tokens_raw)
        if max_tokens > 0:
            payload["max_output_tokens"] = max_tokens
    except (TypeError, ValueError):
        pass
    body = json.dumps(_clean_for_json(payload), ensure_ascii=False).encode("utf-8")
    resp = requests.post(
        _responses_url(api_url),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "FAA/1.0",
            "Accept": "application/json",
        },
        data=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return _extract_responses_text(data)


def _call_openai_compatible(
    provider_name: str,
    api_url: str,
    api_key: str,
    model: str,
    system: str,
    messages: list,
    timeout: int,
    max_retries: int,
    emit=None,
    step_label: str = "api",
    reasoning_effort: str = "",
    max_tokens_raw: str = "12000",
) -> tuple[str, str]:
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    try:
        max_tokens = int(max_tokens_raw)
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
    except (TypeError, ValueError):
        pass
    body = json.dumps(_clean_for_json(payload), ensure_ascii=False).encode("utf-8")

    last_err = None
    for attempt in range(max_retries):
        if emit and attempt > 0:
            emit(step_label, f"{provider_name} API call attempt {attempt + 1}/{max_retries}")
        try:
            resp = requests.post(
                api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "FAA/1.0",
                    "Accept": "application/json",
                },
                data=body,
                timeout=timeout,
            )
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            raw_text = resp.text or ""
            try:
                if "text/event-stream" in content_type.lower() or raw_text.lstrip().startswith("data:"):
                    data = _parse_sse_chat_response(raw_text)
                else:
                    data = resp.json()
            except json.JSONDecodeError as exc:
                body_preview = raw_text[:500]
                raise RuntimeError(
                    f"non-JSON response status={resp.status_code} "
                    f"content_type={content_type!r} "
                    f"body={body_preview!r}"
                ) from exc
            return _extract_chat_text(data)
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            if "SSE response contained no assistant content" in detail:
                try:
                    print(f"[api_client] {provider_name} chat returned empty SSE; trying Responses API...", flush=True)
                    return _call_rewrite_responses(
                        api_url=api_url,
                        api_key=api_key,
                        model=model,
                        system=system,
                        messages=messages,
                        timeout=timeout,
                        max_tokens_raw=max_tokens_raw,
                        reasoning_effort=reasoning_effort,
                    )
                except Exception as fallback_e:
                    detail += f"; responses fallback {type(fallback_e).__name__}: {fallback_e}"
            try:
                if getattr(e, "response", None) is not None:
                    detail += f" status={e.response.status_code}"
                    detail += f" body={(e.response.text or '')[:500]!r}"
            except Exception:
                pass
            last_err = f"{detail} attempt {attempt + 1}"
            print(f"[api_client] {provider_name}: {last_err}", flush=True)
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"[api_client] Retry in {wait}s...", flush=True)
                time.sleep(wait)

    raise RuntimeError(f"{provider_name} failed: {last_err}")


def call_rewrite_api(
    system: str,
    messages: list,
    timeout: int = 180,
    max_retries: int = 3,
    emit=None,
    step_label: str = "api",
) -> tuple[str, str]:
    """Call the rewrite API (currently A6API/OpenAI-compatible)."""
    api_url, api_key, model, reasoning_effort, max_tokens_raw = _rewrite_settings()
    return _call_openai_compatible(
        provider_name="A6API",
        api_url=api_url,
        api_key=api_key,
        model=model,
        system=system,
        messages=messages,
        timeout=timeout,
        max_retries=max_retries,
        emit=emit,
        step_label=step_label,
        reasoning_effort=reasoning_effort,
        max_tokens_raw=max_tokens_raw,
    )
