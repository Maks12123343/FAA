"""Shared Byesu OpenAI-compatible API client."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def _byesu_settings(use_rewrite_model: bool = True) -> tuple[str, str, str]:
    settings = config.load_settings()
    api_key = (os.environ.get("BYESU_API_KEY") or settings.get("byesu_api_key", "")).strip()
    if not api_key:
        raise RuntimeError("No BYESU_API_KEY env var or byesu_api_key configured in Settings.")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError("BYESU_API_KEY must be the real ASCII API key, not a placeholder like 'твій_ключ'.") from exc
    if not api_key.startswith("sk-"):
        raise RuntimeError("BYESU_API_KEY must start with 'sk-'.")

    api_url = (
        os.environ.get("BYESU_API_URL")
        or settings.get("byesu_api_url")
        or "https://byesu.com/v1/chat/completions"
    )
    if use_rewrite_model:
        model = (
            os.environ.get("BYESU_REWRITE_MODEL")
            or settings.get("byesu_rewrite_model")
            or os.environ.get("BYESU_MODEL")
            or settings.get("byesu_model")
            or "gpt-5.5"
        )
    else:
        model = (
            os.environ.get("BYESU_MODEL")
            or settings.get("byesu_model")
            or os.environ.get("BYESU_REWRITE_MODEL")
            or settings.get("byesu_rewrite_model")
            or "gpt-5.5"
        )
    return api_url, api_key, model


def _clean_for_json(value):
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, list):
        return [_clean_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean_for_json(item) for key, item in value.items()}
    return value


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
            text = delta.get("content") or message.get("content") or choice.get("text") or ""
            if isinstance(text, str) and text:
                content_parts.append(text)
            finish = choice.get("finish_reason") or choice.get("native_finish_reason") or finish
    text = "".join(content_parts).strip()
    if not text:
        raise RuntimeError("SSE response contained no assistant content")
    return {"choices": [{"message": {"content": text}, "finish_reason": finish or "stop"}]}


def call_byesu(
    system: str,
    messages: list,
    timeout: int = 180,
    max_retries: int = 3,
    emit=None,
    step_label: str = "api",
    use_rewrite_model: bool = True,
) -> tuple[str, str]:
    """Call Byesu chat completions and return (text, stop_reason)."""
    import requests

    api_url, api_key, model = _byesu_settings(use_rewrite_model=use_rewrite_model)
    settings = config.load_settings()
    max_tokens_raw = os.environ.get("BYESU_MAX_TOKENS") or settings.get("byesu_max_tokens") or "12000"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
    }
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
            emit(step_label, f"Byesu API call attempt {attempt + 1}/{max_retries}")
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
            choice = data["choices"][0]
            text = choice["message"]["content"].strip()
            finish = choice.get("finish_reason") or choice.get("native_finish_reason") or "stop"
            stop_reason = "max_tokens" if finish == "length" else finish
            return text, stop_reason
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            try:
                if getattr(e, "response", None) is not None:
                    detail += f" status={e.response.status_code}"
                    detail += f" body={(e.response.text or '')[:500]!r}"
            except Exception:
                pass
            last_err = f"{detail} attempt {attempt + 1}"
            print(f"[api_client] Byesu: {last_err}", flush=True)
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"[api_client] Retry in {wait}s...", flush=True)
                time.sleep(wait)

    raise RuntimeError(f"Byesu failed: {last_err}")


def call_byesu_rewrite(
    system: str,
    messages: list,
    timeout: int = 180,
    max_retries: int = 3,
    emit=None,
    step_label: str = "api",
) -> tuple[str, str]:
    """Call Byesu using the rewrite model."""
    return call_byesu(
        system,
        messages,
        timeout=timeout,
        max_retries=max_retries,
        emit=emit,
        step_label=step_label,
        use_rewrite_model=True,
    )
