# ExecuTorch harness

TorchTalk's `executorch` harness indexes the parts of an ExecuTorch checkout
that the existing analyzers already understand:

- Python modules under the tracked `src/executorch` package links
- pybind functions and classes
- `TORCH_LIBRARY`, `TORCH_LIBRARY_FRAGMENT`, and `TORCH_LIBRARY_IMPL`
  registrations

`EXECUTORCH_LIBRARY(namespace, "op", implementation)` has a different
argument shape and is not detected by this harness. Supporting that macro
requires a separate detector change and is intentionally outside this
onboarding patch.

## Build an index

Keep the TorchTalk and ExecuTorch repositories in separate directories, then
install TorchTalk and build the index:

```bash
cd /path/to/torchtalk
python -m pip install -e ".[dev]"

torchtalk index build \
  --harness executorch \
  --source /path/to/executorch \
  --no-wait
```

The binding and Python indexes do not require an ExecuTorch build. The optional
C++ call graph additionally requires libclang and a compatible
`compile_commands.json`.

## Pinned validation

Integration tests use `pytorch/executorch@v1.5.1`. Run the source anchors
against an existing checkout with:

```bash
export EXECUTORCH_SOURCE=/path/to/executorch
python -m pytest tests/test_binding_detector_pytorch.py -v -k executorch
```

The two anchors verify:

| Query | Expected source |
|---|---|
| pybind function `get_sgd_optimizer` | `extension/training/pybindings/_training_lib.cpp:137` |
| `llama.update_cache.out` implementation `update_cache_out_no_context` | `extension/llm/custom_ops/op_sdpa_aot.cpp:542` |

Run the measured-count gate with:

```bash
python scripts/harness_smoke.py --harness executorch --clone
```

At the pinned ref, the onboarding patch measured:

```text
bindings: 45
python_modules: 1929
```

The manifest floors are approximately 90% of those measurements:

```text
bindings: >= 40
python_modules: >= 1735
```

The floors catch broad indexing regressions while the two source anchors verify
the specific supported registration and binding shapes.

## Proof queries

After indexing, these queries should return the same source locations as the
anchors:

```text
search("get_sgd_optimizer", mode="bindings")
search("update_cache", mode="bindings")
```

The second query covers the existing `TORCH_LIBRARY_IMPL` registration. It
does not resolve the portable `EXECUTORCH_LIBRARY` implementation or select a
runtime kernel. Dispatcher-style resolution and Python-aware test-impact
analysis are separate features, not capabilities claimed by this harness.
