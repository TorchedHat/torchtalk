"""
Test BindingDetector against real repositories using manifest-driven anchors.

Each YAML file in tests/integration/ defines a target repo with anchors
(file -> expected symbol). Tests are parametrized over these anchors so
adding a new target is a YAML file, not new test code.

Requires the manifest's env_var (e.g. PYTORCH_SOURCE) to point at a checkout.
Skips when the env var is unset.

Usage:
    PYTORCH_SOURCE=/path/to/pytorch pytest tests/test_binding_detector_pytorch.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from torchtalk.analysis.binding_detector import BindingDetector, BindingType

MANIFESTS_DIR = Path(__file__).parent / "integration"


def _load_manifests() -> list[dict]:
    """Load all YAML manifests from tests/integration/."""
    manifests = []
    if MANIFESTS_DIR.exists():
        for path in sorted(MANIFESTS_DIR.glob("*.yml")):
            if path.name.startswith("_"):
                continue  # _template.yml and other non-target files
            with open(path) as f:
                manifest = yaml.safe_load(f)
                manifest["_name"] = path.stem
                manifests.append(manifest)
    return manifests


def _source_path(manifest: dict) -> Path | None:
    """Resolve source path from the manifest's env_var."""
    env_var = manifest.get("env_var", "")
    if val := os.environ.get(env_var):
        p = Path(val)
        if p.exists():
            return p
    return None


MANIFESTS = _load_manifests()


def _anchor_ids() -> list[str]:
    """Build readable test IDs from manifests."""
    ids = []
    for m in MANIFESTS:
        for anchor in m.get("anchors", []):
            check = anchor["check"]
            target = anchor.get("file") or anchor.get("dir", "")
            ids.append(f"{m['_name']}/{check}/{Path(target).name}")
    return ids


def _anchor_params() -> list[tuple[dict, dict, Path | None]]:
    """Build (manifest, anchor, source_path) tuples for parametrize."""
    params = []
    for m in MANIFESTS:
        source = _source_path(m)
        for anchor in m.get("anchors", []):
            params.append((m, anchor, source))
    return params


PARAMS = _anchor_params()

pytestmark = pytest.mark.skipif(
    not PARAMS or all(p[2] is None for p in PARAMS),
    reason="No integration manifests with available source checkouts",
)


@pytest.fixture
def detector():
    """Create a BindingDetector instance."""
    return BindingDetector()


class TestIntegrationAnchors:
    """Parametrized integration tests driven by YAML manifests."""

    @pytest.mark.parametrize(
        "manifest,anchor,source",
        PARAMS,
        ids=_anchor_ids() if PARAMS else [],
    )
    def test_anchor(self, detector, manifest, anchor, source):
        """Verify a single anchor from the integration manifest."""
        if source is None:
            pytest.skip(f"{manifest.get('env_var')} not set")

        check = anchor["check"]

        if check == "pybind_name":
            self._check_pybind_name(detector, source, anchor)
        elif check == "torch_library_cpp_name":
            self._check_torch_library_cpp_name(detector, source, anchor)
        elif check == "has_cuda_kernel":
            self._check_has_cuda_kernel(detector, source, anchor)
        elif check == "has_at_dispatch":
            self._check_has_at_dispatch(detector, source, anchor)
        elif check == "has_binding_types":
            self._check_has_binding_types(detector, source, anchor)
        else:
            pytest.fail(f"Unknown check type: {check}")

    def _check_pybind_name(self, detector, source, anchor):
        """Assert a specific python_name exists in bindings for a file."""
        path = source / anchor["file"]
        if not path.exists():
            pytest.skip(f"File not found: {path}")

        content = path.read_text(errors="replace")
        graph = detector.detect_bindings(str(path), content)

        names = {b.python_name for b in graph.bindings if b.python_name}
        expected = anchor["value"]
        assert expected in names, (
            f"Expected {expected} in {anchor['file']}, got: {sorted(names)[:10]}"
        )

    def _check_torch_library_cpp_name(self, detector, source, anchor):
        """Assert a specific cpp_name exists in TORCH_LIBRARY bindings."""
        path = source / anchor["file"]
        if not path.exists():
            pytest.skip(f"File not found: {path}")

        content = path.read_text(errors="replace")
        graph = detector.detect_bindings(str(path), content)

        cpp_names = {b.cpp_name for b in graph.bindings if b.cpp_name}
        expected = anchor["value"]
        assert expected in cpp_names, (
            f"Expected {expected} in {anchor['file']}, got: {sorted(cpp_names)}"
        )

    def _check_has_cuda_kernel(self, detector, source, anchor):
        """Assert at least one CUDA kernel with a non-empty name is found."""
        scan_dir = source / anchor["dir"]
        if not scan_dir.exists():
            pytest.skip(f"Directory not found: {scan_dir}")

        glob_pattern = anchor.get("glob", "*.cu")
        content_filter = anchor.get("content_filter", "__global__")

        found_kernel = None
        for cu_file in sorted(scan_dir.glob(glob_pattern))[:20]:
            content = cu_file.read_text(errors="replace")
            if content_filter not in content:
                continue
            graph = detector.detect_bindings(str(cu_file), content)
            if graph.cuda_kernels:
                found_kernel = graph.cuda_kernels[0]
                break

        assert found_kernel is not None, f"No CUDA kernel found in {anchor['dir']}"
        assert found_kernel.name, (
            f"Kernel in {anchor['dir']} must have a non-empty name"
        )

    def _check_has_at_dispatch(self, detector, source, anchor):
        """Assert at least one AT_DISPATCH binding with a cpp_name is found."""
        scan_dir = source / anchor["dir"]
        if not scan_dir.exists():
            pytest.skip(f"Directory not found: {scan_dir}")

        glob_pattern = anchor.get("glob", "*.cpp")
        content_filter = anchor.get("content_filter", "AT_DISPATCH")

        found_binding = None
        for cpp_file in sorted(scan_dir.glob(glob_pattern))[:30]:
            content = cpp_file.read_text(errors="replace")
            if content_filter not in content:
                continue
            graph = detector.detect_bindings(str(cpp_file), content)
            at_dispatch = [
                b
                for b in graph.bindings
                if b.binding_type == BindingType.AT_DISPATCH.value
            ]
            if at_dispatch:
                found_binding = at_dispatch[0]
                break

        assert found_binding is not None, (
            f"No AT_DISPATCH binding found in {anchor['dir']}"
        )
        assert found_binding.cpp_name, (
            f"AT_DISPATCH binding in {anchor['dir']} must have a non-empty cpp_name"
        )

    def _check_has_binding_types(self, detector, source, anchor):
        """Assert specific binding types are present in a directory scan."""
        scan_dir = source / anchor["dir"]
        if not scan_dir.exists():
            pytest.skip(f"Directory not found: {scan_dir}")

        graph = detector.detect_bindings_in_directory(str(scan_dir))
        types = {b.binding_type for b in graph.bindings}

        for expected_type in anchor["value"]:
            assert expected_type in types, (
                f"Expected {expected_type} in {anchor['dir']}, got: {sorted(types)}"
            )
