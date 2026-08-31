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


class TestTomlManifests:
    """PR-3: manifests load from TOML with extends/depends_on/expected_minimums."""

    def test_builtins_registered_from_toml(self):
        assert harness_mod.get_harness("pytorch").manifest is PYTORCH_MANIFEST
        assert PYTORCH_MANIFEST.package == "pytorch"
        assert harness_mod.get_harness("vllm").manifest.package == "vllm"
        assert harness_mod.get_harness("torchvision").manifest.package == "torchvision"
        assert "torch-extension" in harness_mod.builtin_manifest_names()
        assert "torch-extension" not in harness_mod.list_harnesses()

    def test_pytorch_toml_matches_patterns_constants(self):
        from torchtalk.analysis import patterns as P

        m = PYTORCH_MANIFEST
        assert m.cpp_search_dirs == tuple(P.CPP_SEARCH_DIRS)
        assert m.python_search_dirs == tuple(P.PYTHON_SEARCH_DIRS)
        assert m.test_search_dirs == tuple(P.TEST_SEARCH_DIRS)
        assert m.test_content_patterns == tuple(P.TEST_CONTENT_PATTERNS)
        assert m.test_utility_modules == tuple(P.TEST_UTILITY_MODULES)
        assert m.exclude_patterns == tuple(P.EXCLUDE_PATTERNS)
        assert m.registration_macros == tuple(P.CPP_BINDING_PATTERNS)
        # Drift guards: the TOML must mirror the detector constants it replaces.
        from torchtalk.analysis.binding_detector import _IMPL_WRAPPERS
        from torchtalk.analysis.decomp_aliases import _DECOMP_FILES

        assert m.cpp_call_wrappers == tuple(_IMPL_WRAPPERS)
        assert m.decomp_alias_paths == tuple(_DECOMP_FILES)

    def test_pytorch_expected_minimums(self):
        assert PYTORCH_MANIFEST.expected_minimums["native_functions"] == 2400
        assert PYTORCH_MANIFEST.depends_on == ()

    def test_bridge_section_inherited_from_torch_extension(self):
        base = harness_mod.load_builtin_manifest("torch-extension")
        assert "at" in base.cpp_namespaces and "c10" in base.cpp_namespaces
        assert "torch.nn" in base.base_class_namespaces
        assert harness_mod.VLLM_MANIFEST.cpp_namespaces == base.cpp_namespaces
        assert (
            harness_mod.VLLM_MANIFEST.base_class_namespaces
            == base.base_class_namespaces
        )
        assert harness_mod.PYTORCH_MANIFEST.cpp_namespaces == ()

    def test_bridge_section_loads_from_toml(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text(
            '[package]\nname = "x"\nextends = "torch-extension"\n'
            '[bridge]\ncpp_namespaces = ["at"]\nbase_class_namespaces = ["torch.nn"]\n'
        )
        m = harness_mod.load_manifest(p)
        assert m.cpp_namespaces == ("at",)
        assert m.base_class_namespaces == ("torch.nn",)

    def test_vllm_extends_torch_extension(self):
        m = harness_mod.VLLM_MANIFEST
        base = harness_mod.load_builtin_manifest("torch-extension")
        assert m.depends_on == ("pytorch",)
        assert m.cpp_search_dirs == base.cpp_search_dirs == ("csrc",)
        assert m.cpp_call_wrappers == base.cpp_call_wrappers
        assert m.op_namespaces == {"torch": "aten"}
        # child replaces, not appends
        expected = ("/tests/", "/benchmarks/", "/examples/", "__pycache__")
        assert m.exclude_patterns == expected
        assert m.cpp_macro_aliases["TORCH_LIBRARY_EXPAND"] == "TORCH_LIBRARY"
        assert m.registration_calls[0].call == "direct_register_custom_op"
        assert m.registration_calls[1].key_arg == 0
        assert m.string_dispatchers == {"collective_rpc": 0}
        assert len(m.string_registries) == 10

    def test_extends_replaces_not_appends(self, tmp_path):
        child = tmp_path / "child.toml"
        child.write_text(
            '[package]\nname = "x"\nextends = "torch-extension"\n'
            '[paths]\nexclude_patterns = ["/only/"]\n'
            "[cpp]\ncall_wrappers = []\n"
            "[expected_minimums]\nbindings = 5\n"
        )
        m = harness_mod.load_manifest(child)
        assert m.exclude_patterns == ("/only/",)
        assert m.cpp_call_wrappers == ()
        assert m.depends_on == ("pytorch",)
        assert m.expected_minimums == {"bindings": 5}

    def test_extends_relative_path_and_cycle(self, tmp_path):
        (tmp_path / "base.toml").write_text(
            '[package]\nname = "b"\n[paths]\ncpp_search_dirs = ["src"]\n'
        )
        (tmp_path / "leaf.toml").write_text(
            '[package]\nname = "leaf"\nextends = "base.toml"\n'
        )
        leaf = harness_mod.load_manifest(tmp_path / "leaf.toml")
        assert leaf.cpp_search_dirs == ("src",)
        (tmp_path / "a.toml").write_text('[package]\nname = "a"\nextends = "b.toml"\n')
        (tmp_path / "b.toml").write_text('[package]\nname = "b"\nextends = "a.toml"\n')
        with pytest.raises(harness_mod.ManifestError, match="circular"):
            harness_mod.load_manifest(tmp_path / "a.toml")

    def test_unknown_key_and_missing_required(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text('[package]\nname = "x"\n[paths]\nbogus = 1\n')
        with pytest.raises(harness_mod.ManifestError, match="unknown key"):
            harness_mod.load_manifest(bad)
        bad.write_text('[package]\nname = "x"\n')
        with pytest.raises(harness_mod.ManifestError, match="cpp_search_dirs"):
            harness_mod.load_manifest(bad)
        with pytest.raises(harness_mod.ManifestError, match="not found"):
            harness_mod.load_manifest(tmp_path / "nope.toml")

    def test_value_type_validation(self, tmp_path):
        cases = {
            '[paths]\ncpp_search_dirs = "csrc"': "list of strings",
            "[expected_minimums]\nbindings = true": "must be integers",
            '[[cpp.registration_calls]]\ncall = "x"': "registration_calls",
            "[python.op_namespaces]\nx = true": "table of strings",
        }
        for body, msg in cases.items():
            p = tmp_path / "v.toml"
            head = '[package]\nname = "v"\n'
            if not body.startswith("[paths]"):
                head += '[paths]\ncpp_search_dirs = ["csrc"]\n'
            p.write_text(f"{head}{body}\n")
            with pytest.raises(harness_mod.ManifestError, match=msg):
                harness_mod.load_manifest(p)

    def test_toml_syntax_error_is_manifest_error(self, tmp_path):
        p = tmp_path / "broken.toml"
        p.write_text("[package\nname = ")
        with pytest.raises(harness_mod.ManifestError, match=r"broken\.toml"):
            harness_mod.load_manifest(p)

    def test_unknown_package_key_and_bad_name(self, tmp_path):
        p = tmp_path / "k.toml"
        p.write_text('[package]\nname = "k"\nversion = "1"\n')
        with pytest.raises(harness_mod.ManifestError, match=r"\[package\] version"):
            harness_mod.load_manifest(p)
        p.write_text('[package]\nname = "bad name"\n')
        with pytest.raises(harness_mod.ManifestError, match="name"):
            harness_mod.load_manifest(p)

    def test_unknown_harness_error_lists_builtins(self):
        with pytest.raises(KeyError, match="built-in manifests"):
            harness_mod.get_harness("definitely-not-a-harness")

    def test_repo_local_manifest_activates(self, tmp_path):
        prev = harness_mod.active_harness_name()
        (tmp_path / ".torchtalk.toml").write_text(
            '[package]\nname = "myext"\nextends = "torch-extension"\n'
        )
        try:
            assert harness_mod.find_repo_manifest(tmp_path) is not None
            assert harness_mod.activate_repo_manifest(tmp_path) == "myext"
            assert harness_mod.active_harness_name() == "myext"
            assert harness_mod.active_manifest().cpp_search_dirs == ("csrc",)
        finally:
            harness_mod.set_active_harness(prev)
            harness_mod._REGISTRY.pop("myext", None)
        assert harness_mod.activate_repo_manifest(tmp_path / "empty") is None
