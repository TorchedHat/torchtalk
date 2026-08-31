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


class TestCachePathsHarnessQualified:
    def test_explicit_package_qualifies_filenames(self):
        from torchtalk.config import cache_paths

        paths = cache_paths("/src", package="vllm")
        assert all("vllm" in p.name for p in paths.values())

    def test_default_follows_active_harness(self):
        from torchtalk.config import cache_paths

        try:
            harness_mod.set_active_harness("vllm")
            vllm_paths = cache_paths("/src")
        finally:
            harness_mod.set_active_harness("pytorch")
        pytorch_paths = cache_paths("/src")
        assert all("vllm" in p.name for p in vllm_paths.values())
        assert all("pytorch" in p.name for p in pytorch_paths.values())
        for key, path in pytorch_paths.items():
            assert path != vllm_paths[key]


class TestManifestOpFields:
    def test_pytorch_defaults(self):
        m = PYTORCH_MANIFEST
        assert m.op_namespaces == {"torch": "aten"}
        assert "torch/_decomp/decompositions.py" in m.decomp_alias_paths
        assert m.dispatch_stub_root == "aten/src/ATen/native"
        assert "TORCH_BOX" in m.cpp_call_wrappers

    def test_empty_by_default(self):
        m = ConventionManifest(package="x", cpp_search_dirs=("csrc",))
        assert m.op_namespaces == {}
        assert m.decomp_alias_paths == ()
        assert m.dispatch_stub_root == ""
        assert m.cpp_call_wrappers == ()
