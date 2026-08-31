# Adding a framework

TorchTalk indexes any repo that follows PyTorch's binding and module
conventions — PyTorch itself, and extensions built on it (vLLM, torchvision,
…). Onboarding a framework is **data, not code**: you write a TOML manifest, an
integration-anchor file, and minimums for the smoke test. Each is one small PR
with a verify command. Open a tracking issue with the "Add a framework" template
and tick through it.

No detector changes should be needed. If the counts come out wrong, that is a
bug report against `analysis/` (or a missing manifest field), not something to
patch around in your manifest.

## 0. Vocabulary

| term | meaning |
|---|---|
| **harness** | a registered `ConventionManifest` — the set of conventions for one framework |
| **manifest** | the TOML file that defines a harness (`src/torchtalk/manifests/<name>.toml`) |
| **source** | the checkout being indexed (`--source`, or auto-detected) |
| **snapshot** | the on-disk index for a source at one fingerprint; stores the manifest it was built with |
| **integration anchors** | `tests/integration/<name>.yml` — concrete facts the detectors must find at a pinned ref |
| **expected minimums** | per-harness counts the smoke test must reach |

## 1. Manifest (`src/torchtalk/manifests/<name>.toml`)

Start from the abstract base and override only what differs:

```toml
[package]
name = "myfw"
extends = "torch-extension"   # inherits cpp/python/bridge conventions from PyTorch
depends_on = ["pytorch"]      # bridges are built toward these harnesses

[paths]
cpp_search_dirs = ["csrc"]    # REQUIRED — where C++/CUDA sources live
python_search_dirs = ["myfw"] # where Python modules are indexed from
python_package_roots = ["myfw"]

[cpp]
# call_wrappers = ["MYFW_BOX"]        # macros that wrap fn pointers in registrations

# [python.op_namespaces]
# myfw = "myfw"                       # TORCH_LIBRARY(myfw, m) → "myfw::"

[bridge]
# cpp_namespaces / base_class_namespaces — inherited from torch-extension

[expected_minimums]
bindings = 100
python_modules = 200
```

Rules:

- `extends` — keys in the child **replace** the base value (tuples are not
  appended). List everything you need.
- `depends_on` — names of other harnesses. Each import/op/base-class reference
  from your package into those packages becomes an `ExternalRef` edge
  (see `docs/bridge-design.md`). Almost always `["pytorch"]`.
- `[paths] cpp_search_dirs` is the only required key. Everything else has a
  base-class default.
- Field reference: `ConventionManifest` in `src/torchtalk/harness.py` — the
  `_SECTION_FIELDS` table maps TOML sections to dataclass fields.

Verify:

```sh
torchtalk index build --source /path/to/myfw --harness myfw
#   Bindings:        …   Python modules:  …   External refs:  …
```

A repo can also ship its own `.torchtalk.toml` at its root; `torchtalk index
build/update` auto-activates it. Use that for private forks; upstream the
manifest into `manifests/` for anything public.

Manifests are loaded with `load_builtin_manifest("myfw")` and registered under
`[package] name`, so `--harness myfw` works everywhere `--harness pytorch` does.

## 2. Integration anchors (`tests/integration/<name>.yml`)

Copy `tests/integration/_template.yml` (files starting with `_` are skipped by
the loader). Each anchor is a fact about the pinned `ref` that a detector must
reproduce — a specific pybind name, a `TORCH_LIBRARY` C++ impl, "this dir has
CUDA kernels". Grep each one by hand first; pick things that survive patch
releases.

Keep `sparse_paths` minimal — CI sparse-clones only those directories.

Then add `<name>` to `matrix.target` in
`.github/workflows/integration-tests.yml`.

Verify:

```sh
export MYFW_SOURCE=/path/to/myfw       # the env_var you set in the yml
python -m pytest tests/test_binding_detector_pytorch.py -v -k myfw
```

## 3. Smoke minimums

`scripts/harness_smoke.py` indexes a real checkout and fails if any count is
below `[expected_minimums]` in the manifest. Set each minimum to roughly
90% of the count you measured in step 1 — tight enough to catch a detector
regression, loose enough to survive upstream churn. Then add `<name>` to
`.github/workflows/harness-smoke.yml`.

Verify:

```sh
python scripts/harness_smoke.py --harness myfw --clone
```

`--harness` only lists harnesses that have `expected_minimums`, so if yours is
missing from `--help`, step 1 isn't done.

## 4. When the counts look wrong

Common shapes, in the order to try them:

1. **Bindings missing** — a registration macro or wrapper the detector doesn't
   know. Try `[cpp] call_wrappers` / `[python.op_namespaces]` first.
2. **Mangled names** (`TORCH_BOX(&foo` instead of `foo`) — same fix as above.
3. **Python modules missing / doubled qualname** (`myfw.myfw.x`) — check
   `python_package_roots` matches the directory that holds `__init__.py`.
4. **0 CUDA kernels** — check `cpp_search_dirs` includes the `.cu` directory.
5. **0 C++ call-graph edges** — expected without `compile_commands.json`;
   not a manifest problem.

If a manifest field can't express it, file an issue against `analysis/` with
the file + line that's mis-detected and the count delta. Don't work around it
in the manifest.

## 5. Docs are part of the deliverable

If this document was wrong or missing something while you did the steps above,
fix it in the same PR series. The next framework should be easier than yours.

## Reference: the existing harnesses

| harness | kind | notes |
|---|---|---|
| `pytorch` | root | defines the conventions; `depends_on = []` |
| `torch-extension` | abstract base | not indexable; `extends` target for everything else |
| `vllm` | extension | first real instance (PR #10); integration anchors + minimums are the open B1/B2 tasks |
| `torchvision` | extension | manifest only — the intended "do it from the doc" test of this page |
