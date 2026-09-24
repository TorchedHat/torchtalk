# Detector gap: recognize `EXECUTORCH_LIBRARY` registrations

## Summary

ExecuTorch registers portable operators with a three-argument macro:

```cpp
EXECUTORCH_LIBRARY(namespace, "operator.overload", implementation);
```

TorchTalk does not currently emit a binding for this shape. The manifest's
`registration_macros` field only controls which files are scanned, while
`macro_aliases` can only rename macros that already have the
`TORCH_LIBRARY(namespace, module)` block shape. Neither field can describe
which arguments contain the operator name and implementation.

This should be fixed in the detector rather than worked around in the
ExecuTorch manifest.

## Reproducer

- Repository: `pytorch/executorch`
- Pinned ref: `v1.5.1`

Representative registration:
[`extension/llm/custom_ops/op_update_cache.cpp:255`](https://github.com/pytorch/executorch/blob/v1.5.1/extension/llm/custom_ops/op_update_cache.cpp#L255-L259)

```cpp
EXECUTORCH_LIBRARY(
    llama,
    "update_cache.out",
    torch::executor::native::update_cache_out);
```

### Actual result

No binding is emitted for the registration in `op_update_cache.cpp`.
Searching for `update_cache` returns the existing `TORCH_LIBRARY_IMPL`
registrations from `op_sdpa_aot.cpp`, but not the portable registration above.

### Expected result

The detector should emit a binding with:

| Field | Expected value |
|---|---|
| operator name | `llama.update_cache.out` |
| C++ implementation | `update_cache_out` |
| source | `extension/llm/custom_ops/op_update_cache.cpp:255` |
| namespace | `llama` |
| binding type | a distinct direct-registration binding type |

## Count delta

At `pytorch/executorch@v1.5.1`, `extension/llm/custom_ops` contains nine
`EXECUTORCH_LIBRARY` invocations. With the ExecuTorch harness restricted to
the existing supported source roots:

| Result | Binding count |
|---|---:|
| Current detector | 45 |
| With these registrations detected | 54 |
| Delta | **+9** |

The nine source registrations can be verified with:

```bash
git grep -n -E 'EXECUTORCH_LIBRARY[[:space:]]*\(' v1.5.1 -- \
  extension/llm/custom_ops
```

## Proposed scope

A small standalone detector change:

- add one pattern for `EXECUTORCH_LIBRARY(namespace, "op", implementation)`
  in `src/torchtalk/analysis/binding_detector.py`;
- add a distinct `BindingType` for the registration;
- normalize a qualified implementation such as
  `torch::executor::native::update_cache_out` to `update_cache_out`;
- add a focused unit test using the registration above; and
- verify that the ExecuTorch binding count increases by nine at the pinned
  ref.

## Out of scope

- Changes to `src/torchtalk/manifests/executorch.toml`
- Dispatcher-style resolution between ATen and portable registrations
- Runtime execution or backend selection
- Python test-impact analysis

## Acceptance criteria

- The representative registration is returned with the expected operator,
  implementation, file, and line.
- The pinned ExecuTorch binding count changes from 45 to 54.
- A focused detector unit test covers the three-argument macro shape.
- Existing TorchTalk tests continue to pass.
- No manifest workaround is introduced.
