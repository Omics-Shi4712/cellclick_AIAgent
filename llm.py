"""Small OpenAI-compatible JSON client; LangGraph itself stays provider-neutral."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


_JSON_SYSTEM_INSTRUCTION = "Return exactly one valid JSON object and no non-JSON text."


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize Pydantic JSON Schema to the provider strict subset."""
    import copy

    result = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            node.pop("minLength", None)
            node.pop("maxLength", None)
            node.pop("minItems", None)
            node.pop("maxItems", None)
            if "const" in node:
                node["enum"] = [node.pop("const")]
            if node.get("type") == "object" or "properties" in node:
                properties = node.get("properties", {})
                node["additionalProperties"] = False
                node["required"] = list(properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def ensure_json_system_instruction(system: str) -> str:
    """Ensure JSON mode satisfies providers that inspect message text."""
    if "json" in system.casefold():
        return system
    separator = "" if system.endswith(("\n", " ")) else "\n"
    return f"{system}{separator}{_JSON_SYSTEM_INSTRUCTION}"


class JsonChatClient(Protocol):
    async def complete_json(self, *, system: str, user: str, schema: dict[str, Any], model: str,
                            temperature: float, strict: bool = False) -> tuple[dict[str, Any], str]: ...


class OpenAIChatClient:
    """OpenAI Chat Completions adapter configured solely from environment variables."""

    def __init__(self, *, timeout=None, max_retries=None) -> None:
        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required for CellClick")
        base_url = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "").strip()
        if base_url.rstrip("/").endswith("/chat/completions"):
            base_url = base_url.rstrip("/")[: -len("/chat/completions")]
        options = {}
        if timeout is not None:
            options["timeout"] = timeout
        if max_retries is not None:
            options["max_retries"] = max_retries
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url.rstrip("/") or None, **options)

    async def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], model: str, temperature: float,
        strict: bool = False,
    ) -> tuple[dict[str, Any], str]:
        self.last_usage = {}
        system = ensure_json_system_instruction(system)
        request = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        # V4.3 phase calls require provider-enforced strict schema. Older protocol
        # calls retain the compatibility fallback for relays that only support JSON mode.
        try:
            response = await self._client.chat.completions.create(
                **request,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "cellclick_response", "schema": _strict_schema(schema) if strict else schema,
                    "strict": strict,
                }},
            )
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            if strict or status_code != 400:
                raise
            response = await self._client.chat.completions.create(
                **request, response_format={"type": "json_object"},
            )
        usage = getattr(response, "usage", None)
        self.last_usage = {key: getattr(usage, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                           if isinstance(getattr(usage, key, None), int)}
        content = response.choices[0].message.content or ""
        try:
            value = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("CellClick model response is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("CellClick model response must be a JSON object")
        return value, content

    async def aclose(self) -> None:
        """Release the HTTP transport while the request event loop is still alive."""
        await self._client.close()


async def close_json_client(client: JsonChatClient) -> None:
    """Close live clients without requiring offline and test clients to implement cleanup."""
    close = getattr(client, "aclose", None)
    if close is not None:
        await close()


def canonical_request_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_fingerprint(
    *, system: str, user: str, schema: dict[str, Any], model: str, temperature: float,
) -> str:
    """Fingerprint the exact request presented to the JsonChatClient interface."""
    payload = {
        "system": system,
        "user": user,
        "schema": schema,
        "model": model,
        "temperature": temperature,
    }
    return hashlib.sha256(canonical_request_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenJsonResponse:
    parsed: dict[str, Any]
    raw: str


class ReplayMissError(LookupError):
    """Raised when replay has no response for the exact request contract."""


class ReplayJsonClient:
    """Fail-closed offline client backed solely by frozen JSON responses."""

    def __init__(self, responses: Mapping[str, FrozenJsonResponse]) -> None:
        self._responses = dict(responses)
        self.call_fingerprints: list[str] = []

    async def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], model: str, temperature: float,
        strict: bool = False,
    ) -> tuple[dict[str, Any], str]:
        fingerprint = request_fingerprint(
            system=system,
            user=user,
            schema=schema,
            model=model,
            temperature=temperature,
        )
        self.call_fingerprints.append(fingerprint)
        try:
            response = self._responses[fingerprint]
        except KeyError as error:
            raise ReplayMissError(f"No frozen response for request {fingerprint}") from error
        return copy.deepcopy(response.parsed), response.raw


async def retry_json(
    client: JsonChatClient, **kwargs: Any,
) -> tuple[dict[str, Any], str]:
    """Compatibility wrapper for callers that imported the former shared helper."""
    from cellclick.retry import retry_json as shared_retry_json

    return await shared_retry_json(client, **kwargs)
