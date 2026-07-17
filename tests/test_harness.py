"""Tests for the harness convention manifest."""

import dataclasses

import pytest

from torchtalk import harness as harness_mod
from torchtalk.harness import (
    PYTORCH_HARNESS,
    PYTORCH_MANIFEST,
    ConventionManifest,
    Harness,
)


class TestPyTorchManifest:
    def test_harness_satisfies_protocol(self):
        assert isinstance(PYTORCH_HARNESS, Harness)
        assert PYTORCH_HARNESS.manifest is PYTORCH_MANIFEST


class TestHarnessRegistry:
    @pytest.fixture(autouse=True)
    def restore_active(self):
        try:
            yield
        finally:
            harness_mod.set_active_harness("pytorch")
            harness_mod._REGISTRY.pop("fakerepo", None)

    def test_pytorch_registered_as_default(self):
        assert harness_mod.get_harness() is PYTORCH_HARNESS
        assert harness_mod.active_manifest() is PYTORCH_MANIFEST

    def test_register_and_activate(self):
        m = ConventionManifest(package="fakerepo", cpp_search_dirs=("csrc",))

        @dataclasses.dataclass(frozen=True)
        class FakeHarness:
            manifest: ConventionManifest = m

        harness_mod.register_harness("fakerepo", FakeHarness())
        harness_mod.set_active_harness("fakerepo")
        assert harness_mod.active_manifest().package == "fakerepo"
        assert harness_mod.get_harness("pytorch") is PYTORCH_HARNESS

    def test_unknown_harness_raises(self):
        with pytest.raises(KeyError, match="Unknown harness"):
            harness_mod.get_harness("nonexistent")
        with pytest.raises(KeyError, match="Unknown harness"):
            harness_mod.set_active_harness("nonexistent")


class TestRealRepoManifests:
    def test_vllm_and_torchvision_registered(self):
        assert harness_mod.get_harness("vllm").manifest.package == "vllm"
        assert harness_mod.get_harness("torchvision").manifest.package == "torchvision"
