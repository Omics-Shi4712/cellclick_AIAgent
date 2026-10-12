"""CellClick shared runtime API.

Versioned annotation implementations are exposed from ``cellclick.v1`` and
future version namespaces; this module only exposes shared client and registry
facilities.
"""

from .llm import OpenAIChatClient


def available_protocols():
    """Return registered protocol versions without loading them at package import time."""
    from .registry import available_protocols as _available_protocols

    return _available_protocols()


def get_protocol(version: str):
    """Load and return one registered protocol on demand."""
    from .registry import get_protocol as _get_protocol

    return _get_protocol(version)


def require_protocol(version: str) -> None:
    """Validate one registered protocol version on demand."""
    from .registry import require_protocol as _require_protocol

    _require_protocol(version)

__all__ = ("OpenAIChatClient", "available_protocols", "get_protocol", "require_protocol")
