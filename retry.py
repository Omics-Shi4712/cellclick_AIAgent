"""Shared JSON transport retry primitives for versioned CellClick protocols.

This module deliberately contains no protocol-specific prompts, schemas, or
workflow rules.  Version packages select their own policy values when calling
``retry_json``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from .llm import JsonChatClient


DEFAULT_RETRY_BACKOFF_BASE_SECONDS = 0.25
DEFAULT_REPAIR_PROMPT = (
    "{user}\n\nYour previous JSON did not satisfy the required response contract: {error}. "
    "Return only a corrected JSON object that satisfies every required field and constraint."
)


def retry_kind(error):
    """Classify V5 failures without depending on a particular provider SDK."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in (408, 429) or isinstance(status, int) and status >= 500:
        return "transient"
    if isinstance(status, int) and 400 <= status < 500:
        return "fatal"
    name = type(error).__name__
    if name == "SchemaCompatibilityError":
        return "fatal"
    if isinstance(error, (TimeoutError, ConnectionError, OSError)) or any(
            word in name for word in ("Timeout", "Connection", "Network", "ProtocolError")):
        return "transient"
    if isinstance(error, (ValueError, TypeError)):
        return "repair"
    return "fatal"


def retry_delay(error, default):
    """Honor both standard Retry-After formats without exposing response data."""
    headers = getattr(getattr(error, "response", None), "headers", {})
    value = headers.get("retry-after", "")
    try:
        return max(default, float(value))
    except (ValueError, TypeError):
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone
            deadline = parsedate_to_datetime(value)
            return max(default, (deadline - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return default


async def retry_json(
    client: JsonChatClient, *, system: str, user: str, schema: dict[str, Any], model: str,
    temperature: float, retries: int, validator: Callable[[dict[str, Any]], Any] | None = None,
    repair_prompt: str = DEFAULT_REPAIR_PROMPT,
    backoff_base_seconds: float = DEFAULT_RETRY_BACKOFF_BASE_SECONDS,
    strict: bool = False, stage: str | None = None,
    attempt_log: list[dict[str, Any]] | None = None,
    v5_policy: bool = False,
) -> tuple[dict[str, Any], str]:
    """Retry transport, JSON, and caller-supplied validation failures.

    ``repair_prompt`` and ``backoff_base_seconds`` are explicit parameters so
    protocol versions can preserve their historical retry contracts without
    depending on another version package.
    """
    error: Exception | None = None
    request_user = user
    last_raw: str | None = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            request = {"system": system, "user": request_user, "schema": schema,
                       "model": model, "temperature": temperature}
            if strict:
                request["strict"] = True
            if v5_policy:
                from .v5.runtime import CURRENT
                runtime = CURRENT.get()
                value, raw = await asyncio.wait_for(client.complete_json(**request),
                    timeout=runtime.total_timeout if runtime else 610)
            else:
                value, raw = await client.complete_json(**request)
            last_raw = raw
            if attempt_log is not None:
                attempt_log.append({"stage": stage, "attempt": attempt + 1, "raw": raw, "status": "received"})
            if validator is not None:
                validator(value)
            if v5_policy:
                from .v5.runtime import emit
                emit("llm_attempt", stage=stage, attempt=attempt + 1,
                     duration_seconds=time.perf_counter() - started, status="ok",
                     token_usage=getattr(client, "last_usage", {}))
            return value, raw
        except Exception as caught:
            error = caught
            kind = retry_kind(caught) if v5_policy else "repair"
            if v5_policy:
                from .v5.runtime import emit
                emit("llm_attempt", stage=stage, attempt=attempt + 1,
                     duration_seconds=time.perf_counter() - started,
                     status="error", error_type=type(caught).__name__, error_kind=kind,
                     token_usage=getattr(client, "last_usage", {}))
                if kind == "fatal":
                    raise
            if attempt_log is not None:
                attempt_log.append({"stage": stage, "attempt": attempt + 1,
                                    "error": f"{type(caught).__name__}: {caught}", "status": "error"})
            if attempt < retries:
                stage_hint = f" Current stage: {stage}. Do not change stage." if stage else ""
                previous = f" Previous invalid JSON: {last_raw}" if last_raw is not None else ""
                if kind == "repair":
                    request_user = repair_prompt.format(user=user, error=caught) + previous + stage_hint
                delay = backoff_base_seconds * (2 ** attempt)
                if v5_policy:
                    delay = retry_delay(caught, delay)
                await asyncio.sleep(delay)
    assert error is not None
    raise error
