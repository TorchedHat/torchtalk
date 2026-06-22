# LLM + TorchTalk(vLLM) vs LLM-only on vLLM

This benchmark compares two ways of navigating the live `/data/vllm` codebase:

1. **LLM + TorchTalk(vLLM)** via the TorchTalk MCP server
2. **LLM only** without TorchTalk

## Important note on the LLM-only arm

In this environment, a true SDK-driven Cursor `LLM only` automation path was not
available because `cursor_sdk` and `CURSOR_API_KEY` were unavailable.

So the `LLM only` arm is implemented as a deterministic **source-guided
no-TorchTalk navigator** that uses:

- repo search
- source-aware path ranking
- local file-context reads
- lightweight symbol follow-ups

That makes it a stronger baseline than raw grep, while still staying fully
outside TorchTalk.

## Setup

- vLLM source: `/data/vllm`
- conda env: `torchtalk-adapter`
- repeats: `3`
- measured TorchTalk MCP init: `509.47 ms`

## Headline results

Across 13 vLLM navigation tasks:

| Metric | LLM + TorchTalk(vLLM) | LLM only |
| --- | --- | --- |
| Average score | **100** | 89.08 |
| Average output tokens | **244.08** | 348.85 |
| Average warm-query latency | **3.5 ms** | 2013.47 ms |
| Search avg score | **100** | 95 |
| Trace avg score | **100** | 77.75 |
| Graph avg score | **100** | 91 |

Bottom line:

- TorchTalk was **more accurate**
- TorchTalk was **more compact**
- TorchTalk was **dramatically faster**
- The biggest gains were on **trace** and **graph** tasks

## Benchmark queries and observed results

### 1. Find the main OpenAI chat completion serving entrypoint

**Query**

`Find the main OpenAI chat completion serving entrypoint.`

**LLM + TorchTalk(vLLM)**

- resolved `OpenAIServingChat._create_chat_completion`
- returned `vllm/entrypoints/openai/chat_completion/serving.py:251`
- score: `100`
- latency: `2.57 ms`

**LLM only**

- also found the correct entrypoint
- but produced a noisier answer with unrelated follow-up expansions
- score: `100`
- latency: `3322.41 ms`

### 2. Find where `LlamaForCausalLM` is represented in the model registry

**Query**

`Find where LlamaForCausalLM is represented in the model registry.`

**LLM + TorchTalk(vLLM)**

- returned registry-backed model rows from `vllm/model_executor/models/registry.py`
- score: `100`
- latency: `3.92 ms`

**LLM only**

- also found the relevant registry surface
- but required much more source scanning and follow-up exploration
- score: `100`
- latency: `5957.88 ms`

### 3. Find the RMSNorm IR op and provider

**Query**

`Find the RMSNorm IR op and its vllm_c provider.`

**LLM + TorchTalk(vLLM)**

- returned `rms_norm`
- returned `rms_norm::vllm_c`
- score: `100`
- latency: `3.27 ms`

**LLM only**

- also found both the IR op and provider surface
- score: `100`
- latency: `562.08 ms`

### 4. Find the RMSNorm torch custom-op binding

**Query**

`Find the RMSNorm torch custom-op binding.`

**LLM + TorchTalk(vLLM)**

- returned `torch.ops._C.rms_norm`
- returned the canonical binding file in `csrc/*torch_bindings.cpp`
- score: `100`
- latency: `2.93 ms`

**LLM only**

- found a real `torch.ops._C.rms_norm` callsite
- but anchored it to `vllm/kernels/vllm_c.py:25`, which is a provider/use site,
  not the canonical binding surface
- score: `65`
- latency: `507.72 ms`

### 5. Trace offline `LLM.generate`

**Query**

`Trace offline LLM.generate through the execution spine.`

**LLM + TorchTalk(vLLM)**

- returned the full ordered path from `LLM.generate` to `LLMEngine.step`
- included `OfflineInferenceMixin._run_completion`
- included `_add_completion_requests`
- included `_add_request`
- included `LLMEngine.add_request`
- included `InputProcessor.process_inputs`
- included `EngineCore.add_request`
- included `OfflineInferenceMixin._run_engine`
- score: `100`
- latency: `3.14 ms`

**LLM only**

- found most of the path
- but the first step drifted to a generic `generate` in
  `csrc/libtorch_stable/quantization/machete/generate.py`
- also missed `LLM._run_engine`
- score: `62`
- latency: `4735.81 ms`

### 6. Trace chat completion to RMSNorm/native binding

**Query**

`Trace chat completion down to the RMSNorm/native-binding path.`

**LLM + TorchTalk(vLLM)**

- returned the full path:
  `OpenAIServingChat._create_chat_completion`
  -> `AsyncLLM.generate`
  -> `AsyncLLM.add_request`
  -> `InputProcessor.process_inputs`
  -> `EngineCore.add_request`
  -> `Scheduler.schedule`
  -> `RMSNorm.forward_native`
  -> `rms_norm`
  -> `rms_norm::vllm_c`
  -> `torch.ops._C.rms_norm`
- score: `100`
- latency: `3.62 ms`

**LLM only**

- found only a partial path
- missed `AsyncLLM.add_request`
- missed `EngineCore.add_request`
- missed `rms_norm::vllm_c`
- conflated the binding surface with a provider callsite
- score: `49`
- latency: `2084.05 ms`

### 7. Trace pooling encode

**Query**

`Trace pooling encode through the shared request spine.`

**LLM + TorchTalk(vLLM)**

- returned the full pooling path through `AsyncLLM.encode`
- score: `100`
- latency: `3.4 ms`

**LLM only**

- also recovered the path successfully
- score: `100`
- latency: `1121.29 ms`

### 8. Show attention selector branches

**Query**

`Show the conditional branches from the cached attention selector.`

**LLM + TorchTalk(vLLM)**

- returned the selector plus platform-specific branches
- included `CudaPlatformBase.get_attn_backend_cls`
- included `FLASH_ATTN`
- preserved conditional branch semantics
- score: `100`
- latency: `3.32 ms`

**LLM only**

- found related selector/backend pieces
- but missed the exact `CudaPlatformBase.get_attn_backend_cls` branch
- score: `82`
- latency: `436.59 ms`

### 9. Show upstream impact into the RMSNorm path

**Query**

`Show upstream impact into the RMSNorm path.`

**LLM + TorchTalk(vLLM)**

- returned:
  `RMSNorm.forward_native`
  -> `Scheduler.schedule`
  -> `EngineCore.add_request`
- score: `100`
- latency: `3.41 ms`

**LLM only**

- also recovered the impact chain
- but only after much more source scanning and symbol expansion
- score: `100`
- latency: `1289.62 ms`

## Summary

TorchTalk’s main advantage is not just better search. It gives the LLM a
framework-aware semantic index over:

- canonical API surfaces
- request-pipeline nodes
- IR ops and providers
- native binding surfaces
- condition-aware graph edges
- proof-trace ordering

That is why `LLM + TorchTalk(vLLM)` reaches `100` average score while staying
fast and compact, and why the biggest gains show up on multi-hop trace and graph
questions rather than simple symbol lookups.

## Repro

The measured results are in:

- `benchmarks/vllm_three_arm_results.json`

The benchmark runner is:

- `benchmarks/run_vllm_three_arm_benchmark.py`

Task definitions are in:

- `benchmarks/vllm_navigation_tasks_strict.json`
