"""Tests for the strict integration-anchor manifest contract."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from torchtalk import harness as harness_mod
from torchtalk.analysis.binding_detector import BindingType
from torchtalk.integration_manifest import (
    IntegrationManifestError,
    load_integration_manifest,
    supported_anchor_checks,
)

REPO_ROOT = Path(__file__).parent.parent
INTEGRATION_DIR = Path(__file__).parent / "integration"


def _valid_manifest(**overrides):
    data = {
        "repo": "example/project",
        "ref": "v1.2.3",
        "env_var": "PYTORCH_SOURCE",
        "sparse_paths": ["csrc"],
        "anchors": [
            {
                "file": "csrc/bindings.cpp",
                "check": "pybind_name",
                "value": "example",
            }
        ],
    }
    data.update(overrides)
    return data


def _write_manifest(tmp_path, data, name="pytorch.yml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _load(tmp_path, data, name="pytorch.yml"):
    return load_integration_manifest(_write_manifest(tmp_path, data, name))


class TestIntegrationManifestPositive:
    def test_loads_file_anchor_and_derives_name(self, tmp_path):
        manifest = _load(tmp_path, _valid_manifest())
        assert manifest["_name"] == "pytorch"
        assert manifest["anchors"][0]["check"] == "pybind_name"

    @pytest.mark.parametrize(
        "anchor",
        [
            {"dir": "csrc", "check": "has_cuda_kernel"},
            {
                "dir": "csrc",
                "check": "has_at_dispatch",
                "glob": "*.cc",
                "content_filter": "AT_DISPATCH",
            },
            {
                "dir": "csrc",
                "check": "has_binding_types",
                "value": ["pybind_function", "torch_library_impl"],
            },
        ],
    )
    def test_loads_supported_directory_anchors(self, tmp_path, anchor):
        manifest = _load(tmp_path, _valid_manifest(anchors=[anchor]))
        assert manifest["anchors"] == [anchor]

    def test_accepts_every_runtime_binding_type(self, tmp_path):
        values = [item.value for item in BindingType]
        manifest = _load(
            tmp_path,
            _valid_manifest(
                anchors=[{"dir": "csrc", "check": "has_binding_types", "value": values}]
            ),
        )
        assert manifest["anchors"][0]["value"] == values

    def test_does_not_change_active_harness(self, tmp_path):
        previous = harness_mod.active_harness_name()
        try:
            harness_mod.set_active_harness("vllm")
            _load(tmp_path, _valid_manifest())
            assert harness_mod.active_harness_name() == "vllm"
        finally:
            harness_mod.set_active_harness(previous)

    def test_all_checked_in_target_manifests_validate(self):
        paths = sorted(
            path
            for path in INTEGRATION_DIR.glob("*.yml")
            if not path.name.startswith("_")
        )
        assert paths
        for path in paths:
            load_integration_manifest(path)

    def test_template_is_schema_valid_when_copied_to_registered_target(self, tmp_path):
        template = INTEGRATION_DIR / "_template.yml"
        copied = tmp_path / "pytorch.yml"
        shutil.copyfile(template, copied)
        load_integration_manifest(copied)


class TestIntegrationManifestRootAndTopLevel:
    @pytest.mark.parametrize("contents", ["", "[]\n", "plain text\n"])
    def test_rejects_non_mapping_or_empty_roots(self, tmp_path, contents):
        path = tmp_path / "pytorch.yml"
        path.write_text(contents)
        with pytest.raises(IntegrationManifestError, match="YAML root"):
            load_integration_manifest(path)

    def test_rejects_yaml_syntax_error(self, tmp_path):
        path = tmp_path / "pytorch.yml"
        path.write_text("repo: [broken\n")
        with pytest.raises(IntegrationManifestError, match="invalid YAML"):
            load_integration_manifest(path)

    def test_rejects_duplicate_key(self, tmp_path):
        path = tmp_path / "pytorch.yml"
        path.write_text("repo: one/two\nrepo: three/four\n")
        with pytest.raises(IntegrationManifestError, match="duplicate key 'repo'"):
            load_integration_manifest(path)

    @pytest.mark.parametrize(
        "field", ["repo", "ref", "env_var", "sparse_paths", "anchors"]
    )
    def test_rejects_missing_top_level_field(self, tmp_path, field):
        data = _valid_manifest()
        data.pop(field)
        with pytest.raises(IntegrationManifestError, match="missing required"):
            _load(tmp_path, data)

    @pytest.mark.parametrize("field", ["repos", "harness", "_name"])
    def test_rejects_unknown_top_level_field(self, tmp_path, field):
        data = _valid_manifest(**{field: "bad"})
        with pytest.raises(IntegrationManifestError, match="unknown field"):
            _load(tmp_path, data)

    @pytest.mark.parametrize(
        "repo", ["org", "https://github.com/org/repo", "org/repo/extra", ""]
    )
    def test_rejects_bad_repo(self, tmp_path, repo):
        with pytest.raises(IntegrationManifestError, match="repo"):
            _load(tmp_path, _valid_manifest(repo=repo))

    @pytest.mark.parametrize("ref", ["", " HEAD", "HEAD", "v1\n2"])
    def test_rejects_bad_ref(self, tmp_path, ref):
        with pytest.raises(IntegrationManifestError, match="ref"):
            _load(tmp_path, _valid_manifest(ref=ref))

    @pytest.mark.parametrize(
        "env_var", ["pytorch_source", "1SOURCE", "SOURCE-NAME", ""]
    )
    def test_rejects_bad_env_var(self, tmp_path, env_var):
        with pytest.raises(IntegrationManifestError, match="env_var"):
            _load(tmp_path, _valid_manifest(env_var=env_var))

    @pytest.mark.parametrize(
        "sparse_paths",
        [[], "csrc", ["/csrc"], ["../csrc"], ["csrc", "csrc"]],
    )
    def test_rejects_bad_sparse_paths(self, tmp_path, sparse_paths):
        with pytest.raises(IntegrationManifestError, match="sparse_paths"):
            _load(tmp_path, _valid_manifest(sparse_paths=sparse_paths))


class TestIntegrationManifestAnchors:
    @pytest.mark.parametrize(
        "anchor",
        [
            {},
            {"check": "unknown"},
            {"check": "pybind_name", "file": "csrc/a.cpp"},
            {"check": "pybind_name", "dir": "csrc", "value": "a"},
            {"check": "has_cuda_kernel", "file": "csrc/a.cpp"},
            {"check": "has_binding_types", "dir": "csrc", "value": "pybind_function"},
            {"check": "has_binding_types", "dir": "csrc", "value": []},
            {"check": "has_binding_types", "dir": "csrc", "value": ["pybind11"]},
            {
                "check": "has_binding_types",
                "dir": "csrc",
                "value": ["pybind_function", "pybind_function"],
            },
            {"check": "has_cuda_kernel", "dir": "csrc", "optional": True},
            {"check": "has_cuda_kernel", "dir": "csrc", "harness": "pytorch"},
            {"check": "has_cuda_kernel", "dir": "/csrc"},
            {"check": "has_cuda_kernel", "dir": "csrc", "glob": ""},
        ],
    )
    def test_rejects_invalid_anchor(self, tmp_path, anchor):
        with pytest.raises(IntegrationManifestError):
            _load(tmp_path, _valid_manifest(anchors=[anchor]))

    def test_rejects_duplicate_anchor_key(self, tmp_path):
        path = tmp_path / "pytorch.yml"
        path.write_text(
            "repo: example/project\nref: v1.2.3\nenv_var: PYTORCH_SOURCE\n"
            "sparse_paths: [csrc]\nanchors:\n"
            "  - check: pybind_name\n    check: has_cuda_kernel\n"
            "    file: csrc/a.cpp\n    value: a\n"
        )
        with pytest.raises(IntegrationManifestError, match="duplicate key 'check'"):
            load_integration_manifest(path)

    def test_supported_checks_match_schema(self):
        assert supported_anchor_checks() == {
            "pybind_name",
            "torch_library_cpp_name",
            "has_cuda_kernel",
            "has_at_dispatch",
            "has_binding_types",
        }


class TestIntegrationManifestHarnessIdentity:
    def test_rejects_unknown_filename_harness(self, tmp_path):
        with pytest.raises(IntegrationManifestError, match="not registered"):
            _load(tmp_path, _valid_manifest(), name="not-a-harness.yml")

    def test_rejects_abstract_filename_harness(self, tmp_path):
        with pytest.raises(IntegrationManifestError, match="abstract"):
            _load(tmp_path, _valid_manifest(), name="torch-extension.yml")

    def test_rejects_template_filename(self, tmp_path):
        with pytest.raises(IntegrationManifestError, match="reserved"):
            _load(tmp_path, _valid_manifest(), name="_template.yml")


def test_all_integration_consumers_use_shared_loader():
    runner = (REPO_ROOT / "tests" / "test_binding_detector_pytorch.py").read_text()
    smoke = (REPO_ROOT / "scripts" / "harness_smoke.py").read_text()
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "integration-tests.yml"
    ).read_text()
    assert "load_integration_manifest" in runner
    assert "load_integration_manifest" in smoke
    assert "load_integration_manifest" in workflow
