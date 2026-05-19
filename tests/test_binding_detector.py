"""Unit tests for binding_detector internals (no PyTorch source required)."""

from __future__ import annotations

from torchtalk.analysis.binding_detector import (
    _DEVICE_PATTERN,
    _KERNEL_PATTERN,
    BindingDetector,
    BindingType,
    _clean_impl_target,
)


class TestCleanImplTarget:
    def test_bare_name(self):
        assert _clean_impl_target("foo") == "foo"

    def test_strips_leading_ampersand(self):
        assert _clean_impl_target("&foo") == "foo"

    def test_strips_namespace(self):
        assert _clean_impl_target("at::native::foo") == "foo"
        assert _clean_impl_target("&at::native::foo") == "foo"

    def test_strips_torch_fn(self):
        assert _clean_impl_target("TORCH_FN(foo)") == "foo"
        assert _clean_impl_target("TORCH_FN(at::native::foo)") == "foo"

    def test_strips_torch_fn_boxed(self):
        assert _clean_impl_target("TORCH_FN_BOXED(foo)") == "foo"
        assert _clean_impl_target("TORCH_FN_BOXED(at::native::foo)") == "foo"

    def test_makefallthrough_falls_back_to_op_name(self):
        # `m.impl("abs", CppFunction::makeFallthrough())` has no real impl;
        # use op_name so by_cpp_name["abs"] still resolves.
        assert (
            _clean_impl_target("CppFunction::makeFallthrough(", op_name="abs") == "abs"
        )

    def test_makenamednotsupported_falls_back(self):
        assert (
            _clean_impl_target("CppFunction::makeNamedNotSupported(", op_name="foo")
            == "foo"
        )

    def test_makefromboxedfunction_extracts_template_arg(self):
        assert (
            _clean_impl_target(
                "CppFunction::makeFromBoxedFunction<&unsupportedDynamicOp>("
            )
            == "unsupportedDynamicOp"
        )
        assert (
            _clean_impl_target("CppFunction::makeFromBoxedFunction<at::native::foo>(")
            == "foo"
        )

    def test_static_cast_falls_back_to_op_name(self):
        # static_cast captures break the regex; use op_name fallback.
        assert _clean_impl_target("static_cast<int64_t (*", op_name="size") == "size"

    def test_lambda_falls_back_to_op_name(self):
        assert _clean_impl_target("[](Tensor", op_name="layer_norm") == "layer_norm"

    def test_empty_returns_op_name(self):
        assert _clean_impl_target("", op_name="foo") == "foo"


class TestImplRegex:
    """Verify cpp_name no longer leaks `TORCH_FN(` wrappers."""

    def _detect(self, src: str) -> list[tuple[str, str]]:
        detector = BindingDetector()
        graph = detector.detect_bindings("test.cpp", src)
        return [
            (b.python_name, b.cpp_name)
            for b in graph.bindings
            if b.binding_type == BindingType.TORCH_LIBRARY_IMPL.value
        ]

    def test_torch_fn_wrapper_extracts_inner_name(self):
        src = """
        TORCH_LIBRARY_IMPL(aten, CPU, m) {
            m.impl("resize_", TORCH_FN(at::native::resize_));
            m.impl("add", TORCH_FN(add_kernel));
        }
        """
        bindings = self._detect(src)
        cpp_names = {cpp for _, cpp in bindings}
        assert "resize_" in cpp_names
        assert "add_kernel" in cpp_names
        assert not any("TORCH_FN" in cpp for cpp in cpp_names)

    def test_ampersand_and_namespace_stripped(self):
        src = """
        TORCH_LIBRARY_IMPL(aten, CPU, m) {
            m.impl("foo", &at::native::foo);
            m.impl("bar", at::native::bar);
        }
        """
        bindings = self._detect(src)
        cpp_names = {cpp for _, cpp in bindings}
        assert "foo" in cpp_names
        assert "bar" in cpp_names

    def test_makefallthrough_keys_under_op_name(self):
        # The fallthrough has no real C++ impl, but we still want the binding
        # keyed under `abs` so a walk through `at::native::abs` finds it.
        src = """
        TORCH_LIBRARY_IMPL(aten, Named, m) {
            m.impl("abs", CppFunction::makeFallthrough());
            m.impl("abs.out", CppFunction::makeFallthrough());
        }
        """
        bindings = self._detect(src)
        cpp_names = {cpp for _, cpp in bindings}
        assert "abs" in cpp_names
        # Overload `abs.out` should also key under bare `abs`
        assert all(cpp == "abs" for _, cpp in bindings)

    def test_makefromboxedfunction_keys_under_template_arg(self):
        src = """
        TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
            m.impl("nonzero",
                torch::CppFunction::makeFromBoxedFunction<&unsupportedDynamicOp>());
        }
        """
        bindings = self._detect(src)
        cpp_names = {cpp for _, cpp in bindings}
        assert "unsupportedDynamicOp" in cpp_names


class TestKernelPattern:
    def _name(self, code: str) -> str | None:
        m = _KERNEL_PATTERN.search(code)
        return m.group(1) if m else None

    def test_simple_global_kernel(self):
        assert self._name("__global__ void simpleKernel(int* a) {") == "simpleKernel"

    def test_template_prefix(self):
        code = "template <typename T>\n__global__ void templated(T* a) {"
        assert self._name(code) == "templated"

    def test_launch_bounds_attribute(self):
        code = "__launch_bounds__(256, 4) __global__ void boundedKernel(float* a) {"
        assert self._name(code) == "boundedKernel"

    def test_c10_launch_bounds_macro(self):
        code = "C10_LAUNCH_BOUNDS_1(256) __global__ void clampedKernel(int* a) {"
        assert self._name(code) == "clampedKernel"

    def test_template_and_launch_bounds_combo(self):
        code = "template <int N> __launch_bounds__(N) __global__ void combo(int* a) {"
        assert self._name(code) == "combo"

    def test_static_modifier(self):
        assert self._name("static __global__ void staticKernel(int* a) {") == (
            "staticKernel"
        )

    def test_skips_non_kernel_function(self):
        assert self._name("void notKernel(int* a) {") is None


class TestDevicePattern:
    def _name(self, code: str) -> str | None:
        m = _DEVICE_PATTERN.search(code)
        return m.group(1) if m else None

    def test_simple_device_function(self):
        assert self._name("__device__ T fetch(const T* p) {") == "fetch"

    def test_inline_modifier(self):
        code = "__device__ inline int64_t start_index(int64_t a) {"
        assert self._name(code) == "start_index"

    def test_forceinline_const(self):
        code = "__device__ __forceinline__ scalar_t op(scalar_t a) const {"
        assert self._name(code) == "op"

    def test_static_host_device_combo(self):
        code = (
            "static __host__ __device__ __forceinline__ "
            "int isfinite_ensure_cuda_math(float val) {"
        )
        assert self._name(code) == "isfinite_ensure_cuda_math"

    def test_pointer_return(self):
        assert self._name("__device__ T* byte_offset(T* ptr, int64_t offset) {") == (
            "byte_offset"
        )

    def test_skips_host_only_function(self):
        assert self._name("void notDevice(int* a) {") is None


class TestCudaDeviceFuncBinding:
    def test_emits_device_func_binding_in_cu_file(self):
        detector = BindingDetector()
        src = "__device__ inline int helper(int x) { return x; }\n"
        graph = detector.detect_bindings("test.cu", src)
        device_bindings = [
            b
            for b in graph.bindings
            if b.binding_type == BindingType.CUDA_DEVICE_FUNC.value
        ]
        assert len(device_bindings) == 1
        assert device_bindings[0].cpp_name == "helper"
        assert device_bindings[0].dispatch_key == "CUDA"

    def test_skips_device_funcs_in_cpp_files(self):
        detector = BindingDetector()
        src = "__device__ inline int helper(int x) { return x; }\n"
        graph = detector.detect_bindings("test.cpp", src)
        device_bindings = [
            b
            for b in graph.bindings
            if b.binding_type == BindingType.CUDA_DEVICE_FUNC.value
        ]
        assert device_bindings == []
