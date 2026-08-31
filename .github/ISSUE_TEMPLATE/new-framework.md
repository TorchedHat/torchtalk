---
name: Add a framework
about: Track onboarding a new PyTorch-based framework (manifest, integration anchors, smoke minimums)
title: "Framework: <name>"
labels: framework
---

<!-- Read docs/adding-a-framework.md first. Each box below is one small PR. -->

**Framework:** <name>  ·  **Repo:** <org/repo>  ·  **Pinned ref:** <tag>
**Why:** <one line — what does indexing this repo let people do?>

### Checklist (one PR each, in order)

- [ ] **Manifest** — `src/torchtalk/manifests/<name>.toml` (`extends = "torch-extension"`, `depends_on = ["pytorch"]`). Verify: `torchtalk index build --source <path> --harness <name>` prints non-zero bindings / python modules.
- [ ] **Integration anchors** — `tests/integration/<name>.yml` from `_template.yml`; add `<name>` to `matrix.target` in `.github/workflows/integration-tests.yml`. Verify: `<ENV_VAR>=<path> python -m pytest tests/test_binding_detector_pytorch.py -k <name>`.
- [ ] **Smoke minimums** — `[expected_minimums]` in the manifest; add `<name>` to `.github/workflows/harness-smoke.yml`. Verify: `python scripts/harness_smoke.py --harness <name> --clone`.
- [ ] **Detector gaps** — anything the counts above surface. File one issue per gap; try a manifest field before touching `analysis/`.
- [ ] **Docs** — anything `docs/adding-a-framework.md` got wrong or didn't cover. Fix the doc, not just the manifest.

### Baseline counts at the pinned ref

| bindings | cuda_kernels | python_modules | nn_modules | external_refs |
|---|---|---|---|---|
| | | | | |

### Known gaps / questions

-
