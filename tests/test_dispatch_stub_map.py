"""Unit tests for kernel-impl → ATen op map construction."""

from __future__ import annotations

from torchtalk.analysis.dispatch_stub_map import (
    _stub_to_op,
    extract_kernel_impl_to_op,
)


def _write(tmp_path, rel: str, body: str) -> None:
    p = tmp_path / "aten/src/ATen/native" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


class TestStubToOp:
    def test_strips_stub_suffix(self):
        assert _stub_to_op("hardsigmoid_stub", {"hardsigmoid"}) == "hardsigmoid"

    def test_strips_kernel_suffix(self):
        assert _stub_to_op("hardsigmoid_kernel", {"hardsigmoid"}) == "hardsigmoid"

    def test_strips_kernel_impl_suffix(self):
        assert _stub_to_op("foo_kernel_impl", {"foo"}) == "foo"

    def test_walks_underscore_segments(self):
        # `softmax_lastdim_kernel` → strip `_kernel` → still no match → walk
        # back to `softmax`.
        assert _stub_to_op("softmax_lastdim_kernel", {"softmax"}) == "softmax"

    def test_returns_none_when_no_match(self):
        assert _stub_to_op("totally_unknown_name", {"abs", "neg"}) is None

    def test_prefers_longer_match(self):
        # If both `softmax_backward` and `softmax` exist, pick the longer one.
        assert (
            _stub_to_op("softmax_backward_kernel", {"softmax", "softmax_backward"})
            == "softmax_backward"
        )


class TestExtractKernelImplToOp:
    def test_basic_register_dispatch(self, tmp_path):
        _write(
            tmp_path,
            "cpu/Activation.cpp",
            "REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel)\n",
        )
        out = extract_kernel_impl_to_op(tmp_path, {"hardsigmoid": {}})
        assert out == {"hardsigmoid_kernel": "hardsigmoid"}

    def test_register_avx_variants(self, tmp_path):
        _write(
            tmp_path,
            "cpu/SoftMaxKernel.cpp",
            "ALSO_REGISTER_AVX512_DISPATCH(softmax_lastdim_kernel, "
            "&softmax_lastdim_kernel_impl)\n"
            "REGISTER_AVX512(foo_stub, &foo_kernel)\n",
        )
        out = extract_kernel_impl_to_op(tmp_path, {"softmax": {}, "foo": {}})
        assert out["softmax_lastdim_kernel_impl"] == "softmax"
        assert out["foo_kernel"] == "foo"

    def test_register_cuda_dispatch(self, tmp_path):
        _write(
            tmp_path,
            "cuda/Activation.cu",
            "REGISTER_CUDA_DISPATCH(threshold_stub, &threshold_kernel_cuda)\n",
        )
        out = extract_kernel_impl_to_op(tmp_path, {"threshold": {}})
        assert out == {"threshold_kernel_cuda": "threshold"}

    def test_skips_when_stub_not_resolvable(self, tmp_path):
        _write(
            tmp_path,
            "cpu/Foo.cpp",
            "REGISTER_DISPATCH(some_unknown_stub, &some_kernel)\n",
        )
        # native_functions doesn't contain the op; entry skipped.
        out = extract_kernel_impl_to_op(tmp_path, {})
        assert out == {}

    def test_returns_empty_when_no_native_functions(self, tmp_path):
        assert extract_kernel_impl_to_op(tmp_path, None) == {}
        assert extract_kernel_impl_to_op(tmp_path, {}) == {}

    def test_handles_missing_native_dir(self, tmp_path):
        # No aten/src/ATen/native/ exists at all.
        assert extract_kernel_impl_to_op(tmp_path, {"foo": {}}) == {}


class TestCallSiteStubBridge:
    """Stubs whose name never strips to a YAML op resolve via their call site."""

    def test_flash_attention_shape(self, tmp_path):
        _write(
            tmp_path,
            "cpu/FlashAttentionKernel.cpp",
            "ALSO_REGISTER_AVX512_DISPATCH(flash_attention_kernel, "
            "&flash_attention_kernel_impl)\n",
        )
        _write(
            tmp_path,
            "transformers/attention.cpp",
            "std::tuple<Tensor, Tensor> "
            "_scaled_dot_product_flash_attention_for_cpu(const Tensor& q) {\n"
            "  if (q.defined()) {\n"
            "    flash_attention_kernel(kCPU, output, logsumexp);\n"
            "  }\n"
            "  return out;\n"
            "}\n",
        )
        out = extract_kernel_impl_to_op(
            tmp_path, {"_scaled_dot_product_flash_attention_for_cpu": {}}
        )
        assert out == {
            "flash_attention_kernel_impl": (
                "_scaled_dot_product_flash_attention_for_cpu"
            )
        }

    def test_unresolvable_call_site_stays_unmapped(self, tmp_path):
        _write(tmp_path, "cpu/K.cpp", "REGISTER_DISPATCH(weird_stub, &weird_impl)\n")
        _write(
            tmp_path,
            "Helpers.cpp",
            "void some_helper(Tensor& t) {\n  weird_stub(kCPU, t);\n}\n",
        )
        out = extract_kernel_impl_to_op(tmp_path, {"unrelated_op": {}})
        assert out == {}

    def test_direct_resolution_not_overridden(self, tmp_path):
        _write(
            tmp_path,
            "cpu/Act.cpp",
            "REGISTER_DISPATCH(gelu_stub, &gelu_kernel_cpu)\n",
        )
        out = extract_kernel_impl_to_op(tmp_path, {"gelu": {}})
        assert out == {"gelu_kernel_cpu": "gelu"}


class TestCallSiteImplResolution:
    def test_enclosing_dispatch_impl_resolves_via_yaml_table(self, tmp_path):
        _write(
            tmp_path,
            "cpu/FlashAttentionKernel.cpp",
            "REGISTER_DISPATCH(flash_attention_kernel, &flash_attention_kernel_impl)\n",
        )
        _write(
            tmp_path,
            "transformers/attention.cpp",
            "std::tuple<Tensor, Tensor> _sdpa_flash_cpu(const Tensor& q) {\n"
            "  flash_attention_kernel(kCPU, out);\n"
            "  return out;\n"
            "}\n",
        )
        nf = {
            "_sdpa_flash_for_cpu": {"dispatch": {"CPU": "_sdpa_flash_cpu"}},
        }
        out = extract_kernel_impl_to_op(tmp_path, nf)
        assert out["flash_attention_kernel_impl"] == "_sdpa_flash_for_cpu"


class TestKernelTuHelperMapping:
    def test_helpers_in_kernel_tu_map_to_its_ops(self, tmp_path):
        _write(
            tmp_path,
            "cpu/FlashAttentionKernel.cpp",
            "void cpu_flash_attention(Tensor& out) {\n}\n"
            "void cpu_flash_attention_backward(Tensor& g) {\n}\n"
            "void flash_attention_kernel_impl(Tensor& out) {\n"
            "  cpu_flash_attention(out);\n"
            "}\n"
            "void flash_attention_backward_kernel_impl(Tensor& g) {\n"
            "  cpu_flash_attention_backward(g);\n"
            "}\n"
            "REGISTER_DISPATCH(fwd_stub, &flash_attention_kernel_impl)\n"
            "REGISTER_DISPATCH(bwd_stub, &flash_attention_backward_kernel_impl)\n",
        )
        _write(
            tmp_path,
            "transformers/attention.cpp",
            "Tensor sdpa_for_cpu(const Tensor& q) {\n  fwd_stub(kCPU, q);\n"
            "  return q;\n}\n"
            "Tensor sdpa_for_cpu_backward(const Tensor& g) {\n"
            "  bwd_stub(kCPU, g);\n  return g;\n}\n",
        )
        nf = {"sdpa_for_cpu": {}, "sdpa_for_cpu_backward": {}}
        out = extract_kernel_impl_to_op(tmp_path, nf)
        assert out["cpu_flash_attention"] == "sdpa_for_cpu"
        assert out["cpu_flash_attention_backward"] == "sdpa_for_cpu_backward"

    def test_many_op_files_do_not_blanket_map_helpers(self, tmp_path):
        body = "".join(
            f"REGISTER_DISPATCH(op{i}_stub, &op{i}_kernel)\n" for i in range(6)
        )
        _write(tmp_path, "cpu/Big.cpp", body + "void shared_helper(int x) {\n}\n")
        nf = {f"op{i}": {} for i in range(6)}
        out = extract_kernel_impl_to_op(tmp_path, nf)
        assert "shared_helper" not in out
