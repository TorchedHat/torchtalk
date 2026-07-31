"""
Test BindingDetector against the actual PyTorch repository.

Requires PYTORCH_SOURCE or PYTORCH_PATH environment variable to be set.

Usage:
    PYTORCH_SOURCE=/path/to/pytorch pytest tests/test_binding_detector_pytorch.py -v
"""

import pytest

from torchtalk.analysis.binding_detector import BindingDetector, BindingType

from .conftest import get_pytorch_path

PYTORCH_PATH = get_pytorch_path()

pytestmark = pytest.mark.skipif(
    PYTORCH_PATH is None,
    reason="PYTORCH_SOURCE or PYTORCH_PATH environment variable not set",
)


@pytest.fixture
def detector():
    """Create a BindingDetector instance."""
    return BindingDetector()


class TestPybind11Detection:
    def test_detects_bindings_in_module_cpp(self, detector):
        test_file = PYTORCH_PATH / "torch/csrc/Module.cpp"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        content = test_file.read_text(errors="replace")
        graph = detector.detect_bindings(str(test_file), content)

        names = {b.python_name for b in graph.bindings if b.python_name}
        assert "_WeakTensorRef" in names, (
            f"Expected _WeakTensorRef in Module.cpp, got: {sorted(names)[:10]}"
        )


class TestTorchLibraryDetection:
    def test_detects_torch_library_in_rnn(self, detector):
        test_file = PYTORCH_PATH / "aten/src/ATen/native/RNN.cpp"
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        content = test_file.read_text(errors="replace")
        graph = detector.detect_bindings(str(test_file), content)

        cpp_names = {b.cpp_name for b in graph.bindings if b.cpp_name}
        assert "quantized_lstm" in cpp_names, (
            f"Expected quantized_lstm in RNN.cpp bindings, got: {sorted(cpp_names)}"
        )


class TestCudaKernelDetection:
    def test_detects_cuda_kernels(self, detector):
        cuda_dir = PYTORCH_PATH / "aten/src/ATen/native/cuda"
        if not cuda_dir.exists():
            pytest.skip(f"CUDA directory not found: {cuda_dir}")

        found_kernel = None
        for cu_file in list(cuda_dir.glob("*.cu"))[:20]:
            content = cu_file.read_text(errors="replace")
            if "__global__" not in content:
                continue

            graph = detector.detect_bindings(str(cu_file), content)
            if graph.cuda_kernels:
                found_kernel = graph.cuda_kernels[0]
                break

        assert found_kernel is not None, "Should find a CUDA kernel in native/cuda/"
        assert found_kernel.name, "Kernel must have a non-empty name"


class TestAtDispatchDetection:
    def test_detects_at_dispatch_macros(self, detector):
        native_dir = PYTORCH_PATH / "aten/src/ATen/native"
        if not native_dir.exists():
            pytest.skip(f"Native directory not found: {native_dir}")

        found_binding = None
        for cpp_file in list(native_dir.glob("*.cpp"))[:30]:
            content = cpp_file.read_text(errors="replace")
            if "AT_DISPATCH" not in content:
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

        assert found_binding is not None, "Should find AT_DISPATCH macros"
        assert found_binding.cpp_name, (
            "AT_DISPATCH binding must have a non-empty cpp_name"
        )


class TestDirectoryScan:
    def test_scans_autograd_directory(self, detector):
        scan_dir = PYTORCH_PATH / "torch/csrc/autograd"
        if not scan_dir.exists():
            pytest.skip(f"Directory not found: {scan_dir}")

        graph = detector.detect_bindings_in_directory(str(scan_dir))

        types = {b.binding_type for b in graph.bindings}
        assert BindingType.PYBIND_METHOD.value in types, (
            f"Expected pybind_method in autograd bindings, got types: {sorted(types)}"
        )

    def test_categorizes_bindings_by_type(self, detector):
        scan_dir = PYTORCH_PATH / "torch/csrc/autograd"
        if not scan_dir.exists():
            pytest.skip(f"Directory not found: {scan_dir}")

        graph = detector.detect_bindings_in_directory(str(scan_dir))

        types = {b.binding_type for b in graph.bindings}
        assert BindingType.PYBIND_METHOD.value in types
        assert BindingType.TORCH_LIBRARY_IMPL.value in types, (
            f"Expected both pybind and torch_library types, got: {sorted(types)}"
        )
