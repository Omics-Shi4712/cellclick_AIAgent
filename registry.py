"""Public protocol registry for explicitly versioned CellClick workflows."""

from __future__ import annotations

from .base import ProtocolSpec
from .v1 import V1_PROTOCOL
from .v2 import V2_PROTOCOL
from .v3 import V3_PROTOCOL
from .v4 import V4_PROTOCOL


_PROTOCOLS: dict[str, ProtocolSpec] = {
    V1_PROTOCOL.version: V1_PROTOCOL,
    V2_PROTOCOL.version: V2_PROTOCOL,
    V3_PROTOCOL.version: V3_PROTOCOL,
    V4_PROTOCOL.version: V4_PROTOCOL,
}


def available_protocols() -> tuple[str, ...]:
    return tuple(sorted(_PROTOCOLS))


def get_protocol(version: str) -> ProtocolSpec:
    if not isinstance(version, str) or not version.strip():
        raise ValueError("CellClick protocol_version must be a non-empty string")
    normalized = version.strip()
    try:
        return _PROTOCOLS[normalized]
    except KeyError as error:
        supported = ", ".join(available_protocols())
        raise ValueError(
            f"Unsupported CellClick protocol_version {normalized!r}; supported: {supported}"
        ) from error


def require_protocol(version: str) -> None:
    get_protocol(version)


__all__ = ("available_protocols", "get_protocol", "require_protocol")
