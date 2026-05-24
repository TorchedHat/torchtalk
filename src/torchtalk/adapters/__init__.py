"""Framework adapter package."""

from .base import DEFAULT_FRAMEWORK, KNOWN_FRAMEWORKS, FrameworkAdapter, FrameworkId

__all__ = [
    "DEFAULT_FRAMEWORK",
    "KNOWN_FRAMEWORKS",
    "FrameworkAdapter",
    "FrameworkId",
    "PyTorchAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "unregister_adapter",
]


def __getattr__(name: str):
    if name == "PyTorchAdapter":
        from .pytorch import PyTorchAdapter

        return PyTorchAdapter
    if name in {
        "get_adapter",
        "list_adapters",
        "register_adapter",
        "unregister_adapter",
    }:
        from .registry import (
            get_adapter,
            list_adapters,
            register_adapter,
            unregister_adapter,
        )

        return {
            "get_adapter": get_adapter,
            "list_adapters": list_adapters,
            "register_adapter": register_adapter,
            "unregister_adapter": unregister_adapter,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
