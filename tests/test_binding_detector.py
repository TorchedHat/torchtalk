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

    def test_launch_bounds_after_global(self):
        code = "__global__ void __launch_bounds__(256) after_kernel(float* x) {"
        assert self._name(code) == "after_kernel"

    def test_bare_macro_attribute_after_global(self):
        code = (
            "__global__ __quickreduce_launch_bounds_two_shot__ static void\n"
            "allreduce_prototype_twoshot(T const* A, T* B, uint32_t N) {"
        )
        assert self._name(code) == "allreduce_prototype_twoshot"

    def test_non_void_template_return_type(self):
        code = (
            "template <typename scalar_t, int width>\n"
            "__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>"
            "\n"
            "fused_add_rms_norm_kernel(scalar_t* __restrict__ input, float epsilon) {"
        )
        assert self._name(code) == "fused_add_rms_norm_kernel"

    def test_call_like_macro_and_launch_bounds_after_global(self):
        code = (
            "template <typename T>\n"
            "__global__ void __launch_bounds__(1024, 1) MY_ATTR(x)\n"
            "stacked_kernel(T* a) {"
        )
        assert self._name(code) == "stacked_kernel"

    def test_nested_parens_in_launch_bounds(self):
        code = (
            "template <class Type, bool UE8M0_SF = false>\n"
            "__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))\n"
            "cvt_fp16_to_fp4(int32_t numRows, Type const* in) {"
        )
        assert self._name(code) == "cvt_fp16_to_fp4"

    def test_line_comment_between_attribute_and_name(self):
        code = (
            "__global__ void __launch_bounds__(32 * warps_per_block, blocks_per_sm)\n"
            "// a is column major, b is row major\n"
            "hadamard_transform_kernel(b16* a, b16* out, int total_num_chunks) {"
        )
        assert self._name(code) == "hadamard_transform_kernel"

    def test_launch_bounds_is_never_a_kernel_name(self):
        code = "__global__ void __launch_bounds__(512) real_name(int* a) {"
        names = [m.group(1) for m in _KERNEL_PATTERN.finditer(code)]
        assert names == ["real_name"]


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


class TestManifestMacroConfig:
    """Primitive 1: macro-alias expansion + token substitution (vLLM shape)."""

    def _detector(self):
        return BindingDetector(
            macro_aliases={"TORCH_LIBRARY_EXPAND": "TORCH_LIBRARY"},
            token_map={"TORCH_EXTENSION_NAME": "_C"},
        )

    def test_macro_alias_expands_to_torch_library(self):
        code = (
            "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {\n"
            '  ops.def("paged_attention(Tensor q) -> Tensor");\n'
            '  ops.impl("paged_attention", &paged_attention);\n'
            "}\n"
        )
        graph = self._detector().detect_bindings("/x.cpp", code)
        by_type = {b.binding_type: b for b in graph.bindings}
        op = by_type[BindingType.TORCH_OP.value]
        assert op.python_name == "_C.paged_attention"
        assert op.namespace == "_C"
        assert op.line_number == 2
        impl = by_type[BindingType.TORCH_LIBRARY_IMPL.value]
        assert impl.cpp_name == "paged_attention"

    def test_token_map_resolves_pybind_module_name(self):
        code = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd", &fwd); }'
        graph = self._detector().detect_bindings("/y.cpp", code)
        assert graph.bindings[0].namespace == "_C"
        assert "TORCH_EXTENSION_NAME" not in {b.namespace for b in graph.bindings}

    def test_unconfigured_detector_is_unchanged(self):
        code = (
            'TORCH_LIBRARY(aten, m) { m.def("relu(Tensor self) -> Tensor"); }\n'
            "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {\n"
            '  ops.def("paged_attention(Tensor q) -> Tensor");\n'
            "}\n"
        )
        graph = BindingDetector().detect_bindings("/z.cpp", code)
        names = {b.python_name for b in graph.bindings}
        assert "aten.relu" in names
        assert not any("paged_attention" in n for n in names)

    def test_word_boundary_no_partial_token_hits(self):
        code = (
            "TORCH_LIBRARY_EXPANDED_THING(foo, m) {}\n"
            "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {\n"
            '  ops.def("op_a(Tensor t) -> Tensor");\n'
            "}\n"
        )
        graph = self._detector().detect_bindings("/w.cpp", code)
        assert {b.namespace for b in graph.bindings if b.namespace} == {"_C"}


class TestModuleVarTorchOps:
    def test_library_block_uses_declared_module_variable(self):
        code = (
            "TORCH_LIBRARY(myns, lib) {\n"
            '  lib.def("custom_op(Tensor t) -> Tensor");\n'
            "}\n"
        )
        graph = BindingDetector().detect_bindings("/v.cpp", code)
        assert any(b.python_name == "myns.custom_op" for b in graph.bindings)


class TestBindingToDict:
    def test_emits_line_number_key(self):
        from torchtalk.analysis.binding_detector import Binding

        d = Binding(
            python_name="x",
            cpp_name="x",
            binding_type="torch_op",
            file_path="/f.cpp",
            line_number=7,
        ).to_dict()
        assert d["line_number"] == 7
        assert "line" not in d


class TestStandalonePybindBounds:
    def test_def_chain_bounded_by_semicolon(self):
        code = (
            'py::class_<Foo>(m, "Foo")\n'
            '    .def("real_method", &Foo::real_method);\n'
            "\n"
            'm.def("unrelated_fn", &unrelated_fn);\n'
            'other.attr("x").def("phantom_method", &Bar::phantom);\n'
        )
        graph = BindingDetector().detect_bindings("/x.cpp", code)
        names = {b.python_name for b in graph.bindings}
        assert "Foo.real_method" in names
        assert "Foo.phantom_method" not in names

    def test_module_region_not_double_extracted(self):
        code = (
            "PYBIND11_MODULE(mymod, m) {\n"
            '  py::class_<Foo>(m, "Foo")\n'
            '      .def("method_a", &Foo::method_a);\n'
            "}\n"
        )
        graph = BindingDetector().detect_bindings("/y.cpp", code)
        rows = [
            (b.python_name, b.binding_type)
            for b in graph.bindings
            if b.python_name == "Foo.method_a"
        ]
        assert len(rows) == 1


class TestEnclosingFunctionKeywords:
    def test_kernel_launch_inside_if_attributes_to_function(self):
        code = (
            "__global__ void my_kernel(float* x) {}\n"
            "void launch_my_kernel(float* x) {\n"
            "  if (x != nullptr) {\n"
            "    my_kernel<<<1, 1>>>(x);\n"
            "  }\n"
            "}\n"
        )
        graph = BindingDetector().detect_bindings("/k.cu", code)
        kernel = next(k for k in graph.cuda_kernels if k.name == "my_kernel")
        assert kernel.called_by == ["launch_my_kernel"]

    def test_at_dispatch_inside_for_not_named_for(self):
        code = (
            "void gelu_impl(Tensor& t) {\n"
            "  for (int i = 0; i < 2; i++) {\n"
            '    AT_DISPATCH_FLOATING_TYPES(t.scalar_type(), "gelu", [&] {});\n'
            "  }\n"
            "}\n"
        )
        graph = BindingDetector().detect_bindings("/d.cpp", code)
        disp = next(b for b in graph.bindings if b.binding_type == "at_dispatch")
        assert disp.cpp_name == "gelu_impl"


class TestSearchDirBounds:
    """Directory scans honor manifest cpp_search_dirs."""

    _CODE = 'TORCH_LIBRARY(aten, m) { m.def("relu(Tensor self) -> Tensor"); }\n'

    def _repo(self, tmp_path_factory):
        # mktemp instead of tmp_path: tmp_path embeds the test name, whose
        # "test_" segment trips the scanner's exclusion patterns.
        repo = tmp_path_factory.mktemp("scandir")
        (repo / "csrc").mkdir()
        (repo / "vendored").mkdir()
        (repo / "csrc" / "inside.cpp").write_text(self._CODE)
        (repo / "vendored" / "outside.cpp").write_text(self._CODE)
        return repo

    def test_scan_limited_to_search_dirs(self, tmp_path_factory):
        repo = self._repo(tmp_path_factory)
        detector = BindingDetector(search_dirs=("csrc",))
        graph = detector.detect_bindings_in_directory(str(repo))
        files = {b.file_path for b in graph.bindings}
        assert files
        assert all("vendored" not in f for f in files)

    def test_unbounded_detector_scans_whole_tree(self, tmp_path_factory):
        repo = self._repo(tmp_path_factory)
        graph = BindingDetector().detect_bindings_in_directory(str(repo))
        files = {b.file_path for b in graph.bindings}
        assert any("vendored" in f for f in files)
        assert any("csrc" in f for f in files)

    def test_missing_search_dirs_scan_nothing(self, tmp_path_factory):
        repo = self._repo(tmp_path_factory)
        detector = BindingDetector(search_dirs=("nonexistent",))
        graph = detector.detect_bindings_in_directory(str(repo))
        assert graph.bindings == []


class TestManifestExcludeAndPrefilter:
    """Manifest exclude_patterns and registration_macros drive directory scans."""

    _CODE = 'TORCH_LIBRARY(aten, m) { m.def("relu(Tensor self) -> Tensor"); }\n'

    def _repo(self, tmp_path_factory):
        repo = tmp_path_factory.mktemp("scandir")
        (repo / "csrc" / "generated").mkdir(parents=True)
        (repo / "csrc" / "tests").mkdir()
        (repo / "csrc" / "generated" / "gen.cpp").write_text(self._CODE)
        (repo / "csrc" / "tests" / "helper.cpp").write_text(self._CODE)
        return repo

    def test_manifest_excludes_replace_defaults(self, tmp_path_factory):
        repo = self._repo(tmp_path_factory)
        detector = BindingDetector(search_dirs=("csrc",), exclude_patterns=("/tests/",))
        graph = detector.detect_bindings_in_directory(str(repo))
        files = {b.file_path for b in graph.bindings}
        assert any("generated" in f for f in files)
        assert not any("/tests/" in f for f in files)

    def test_default_excludes_apply_when_manifest_unset(self, tmp_path_factory):
        repo = self._repo(tmp_path_factory)
        detector = BindingDetector(search_dirs=("csrc",))
        graph = detector.detect_bindings_in_directory(str(repo))
        assert graph.bindings == []

    def test_registration_macros_prefilter(self, tmp_path_factory):
        repo = tmp_path_factory.mktemp("scandir")
        (repo / "csrc").mkdir()
        (repo / "csrc" / "ops.cpp").write_text(self._CODE)
        detector = BindingDetector(
            search_dirs=("csrc",), registration_macros=("CUSTOM_REGISTER",)
        )
        assert detector.detect_bindings_in_directory(str(repo)).bindings == []
        fallback = BindingDetector(search_dirs=("csrc",))
        assert fallback.detect_bindings_in_directory(str(repo)).bindings


class TestCallWrappers:
    def test_default_wrappers_include_box_and_selective(self):
        assert _clean_impl_target("TORCH_BOX(&foo)") == "foo"
        assert _clean_impl_target("TORCH_SELECTIVE_FN(ns::bar)") == "bar"

    def test_custom_wrappers_override_defaults(self):
        assert _clean_impl_target("MY_WRAP(foo)", wrappers=("MY_WRAP",)) == "foo"
        assert _clean_impl_target("TORCH_FN(foo)", wrappers=("MY_WRAP",)) != "foo"

    def test_detector_passes_wrappers(self):
        src = """
        TORCH_LIBRARY_IMPL(aten, CPU, m) {
            m.impl("relu", MY_WRAP(relu_kernel));
        }
        """
        graph = BindingDetector(call_wrappers=("MY_WRAP",)).detect_bindings(
            "ops.cpp", src
        )
        assert "relu_kernel" in {b.cpp_name for b in graph.bindings}
