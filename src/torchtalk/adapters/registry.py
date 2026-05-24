"""Registry for framework adapters."""

from __future__ import annotations

from .base import DEFAULT_FRAMEWORK, FrameworkAdapter, FrameworkId
from .pytorch import PyTorchAdapter

_ADAPTERS: dict[FrameworkId, FrameworkAdapter] = {}


def register_adapter(
    adapter: FrameworkAdapter,
    *,
    overwrite: bool = False,
) -> FrameworkAdapter:
    """Register a framework adapter."""

    framework_id = adapter.framework_id.lower()
    if framework_id in _ADAPTERS and not overwrite:
        raise ValueError(f"Adapter already registered for framework '{framework_id}'")
    _ADAPTERS[framework_id] = adapter
    return adapter


def unregister_adapter(framework_id: FrameworkId) -> FrameworkAdapter | None:
    """Remove a registered adapter."""

    return _ADAPTERS.pop(framework_id.lower(), None)


def get_adapter(framework_id: FrameworkId = DEFAULT_FRAMEWORK) -> FrameworkAdapter:
    """Get a registered adapter by framework id."""

    normalized = framework_id.lower()
    try:
        return _ADAPTERS[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTERS))
        raise KeyError(
            f"Unknown framework '{framework_id}'. Registered adapters: {known or 'none'}"
        ) from exc


def list_adapters() -> tuple[FrameworkId, ...]:
    """Return the registered framework ids."""

    return tuple(sorted(_ADAPTERS))


register_adapter(PyTorchAdapter())
