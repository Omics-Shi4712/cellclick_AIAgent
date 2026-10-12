"""Shared JSON transport retry primitives for versioned CellClick protocols.

This module deliberately contains no protocol-specific prompts, schemas, or
workflow rules.  Version packages select their own policy values when calling
``retry_json``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from .llm import JsonChatClient


DEFAULT_RETRY_BACKOFF_BASE_SECONDS = 0.25
DEFAULT_REPAIR_PROMPT = (
    "{user}\n\nYour previous JSON did not satisfy the required response contract: {error}. "
    "Return only a corrected JSON object that satisfies every required field and constraint."
)


async def retry_json(
    client: JsonChatClient, *, system: str, user: str, schema: dict[str, Any], model: str,
    temperature: float, retries: int, validator: Callable[[dict[str, Any]], Any] | None = None,
    repair_prompt: str = DEFAULT_REPAIR_PROMPT,
    backoff_base_seconds: float = DEFAULT_RETRY_BACKOFF_BASE_SECONDS,
    strict: bool = False, stage: str | None = None,
    attempt_log: list[dict[str, Any]] | None = None,
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
        try:
            request = {"system": system, "user": request_user, "schema": schema,
                       "model": model, "temperature": temperature}
            if strict:
                request["strict"] = True
            value, raw = await client.complete_json(**request)
            last_raw = raw
            if attempt_log is not None:
                attempt_log.append({"stage": stage, "attempt": attempt + 1, "raw": raw, "status": "received"})
            if validator is not None:
                validator(value)
            return value, raw
        except Exception as caught:
            error = caught
            if attempt_log is not None:
                attempt_log.append({"stage": stage, "attempt": attempt + 1,
                                    "error": f"{type(caught).__name__}: {caught}", "status": "error"})
            if attempt < retries:
                stage_hint = f" Current stage: {stage}. Do not change stage." if stage else ""
                previous = f" Previous invalid JSON: {last_raw}" if last_raw is not None else ""
                request_user = repair_prompt.format(user=user, error=caught) + previous + stage_hint
                await asyncio.sleep(backoff_base_seconds * (2 ** attempt))
    assert error is not None
    raise error
