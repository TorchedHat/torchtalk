"""Tests for framework adapter registration and bootstrap dispatch."""

from __future__ import annotations

from torchtalk import indexer
from torchtalk.adapters import (
    get_adapter,
    register_adapter,
    unregister_adapter,
)


class FakeAdapter:
    framework_id = "fake"
    display_name = "Fake Framework"

    def __init__(self, calls: list[tuple]):
        self.calls = calls

    def resolve_source(self, cli_flag: str | None = None) -> str | None:
        self.calls.append(("resolve_source", cli_flag))
        return cli_flag or "/tmp/fake-source"

    def validate_source(self, path):
        return True, f"Valid fake source: {path}"

    def bootstrap(
        self,
        source: str | None = None,
        *,
        index_path: str | None = None,
    ) -> None:
        self.calls.append(("bootstrap", source, index_path))


class TestAdapterRegistry:
    def test_default_adapter_is_pytorch(self):
        adapter = get_adapter()
        assert adapter.framework_id == "pytorch"
        assert adapter.display_name == "PyTorch"

    def test_can_register_and_unregister_fake_adapter(self):
        calls: list[tuple] = []
        adapter = FakeAdapter(calls)
        register_adapter(adapter)
        try:
            assert get_adapter("fake") is adapter
        finally:
            unregister_adapter("fake")


class TestAdapterBootstrap:
    def test_init_via_adapter_uses_registered_adapter(self):
        prior_framework = indexer._state.framework
        prior_source_root = indexer._state.source_root
        prior_pytorch_source = indexer._state.pytorch_source

        calls: list[tuple] = []
        adapter = FakeAdapter(calls)
        register_adapter(adapter)
        try:
            resolved = indexer.init_via_adapter(
                framework="fake",
                source="/tmp/fake-root",
            )
            assert resolved == "/tmp/fake-root"
            assert calls == [("bootstrap", "/tmp/fake-root", None)]
            assert indexer._state.framework == "fake"
            assert indexer._state.source_root == "/tmp/fake-root"
        finally:
            unregister_adapter("fake")
            indexer._state.framework = prior_framework
            indexer._state.source_root = prior_source_root
            indexer._state.pytorch_source = prior_pytorch_source
