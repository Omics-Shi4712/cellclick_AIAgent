"""Frozen transport retry semantics for CellClick v1."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cellclick.llm import JsonChatClient
from cellclick.retry import retry_json as _retry_json


RETRY_POLICY_VERSION = "exponential-json-repair-v1"
RETRY_BACKOFF_BASE_SECONDS = 0.25
REPAIR_PROMPT = (
    "{user}\n\nYour previous JSON did not satisfy the required response contract: {error}. "
    "Return only a corrected JSON object that satisfies every required field and constraint."
)


async def retry_json(
    client: JsonChatClient, *, system: str, user: str, schema: dict[str, Any], model: str,
    temperature: float, retries: int, validator: Callable[[dict[str, Any]], Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """V1-compatible facade for the shared transport retry implementation."""
    return await _retry_json(
        client,
        system=system,
        user=user,
        schema=schema,
        model=model,
        temperature=temperature,
        retries=retries,
        validator=validator,
        repair_prompt=REPAIR_PROMPT,
        backoff_base_seconds=RETRY_BACKOFF_BASE_SECONDS,
    )
