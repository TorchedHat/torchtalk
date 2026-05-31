"""Static vLLM indexing helpers for TorchTalk."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

_API_METHOD_SPECS: dict[str, tuple[str, ...]] = {
    "vllm/entrypoints/llm.py": (
        "LLM.generate",
        "LLM.chat",
        "LLM._run_completion",
        "LLM._add_request",
    ),
    "vllm/entrypoints/openai/chat_completion/serving.py": (
        "OpenAIServingChat.render_chat_request",
        "OpenAIServingChat.create_chat_completion",
        "OpenAIServingChat._create_chat_completion",
    ),
    "vllm/entrypoints/pooling/base/serving.py": (
        "PoolingServingBase._prepare_generators",
    ),
    "vllm/entrypoints/serve/render/serving.py": (
        "OpenAIServingRender.render_chat_request",
        "OpenAIServingRender.render_chat",
        "OpenAIServingRender.render_completion_request",
        "OpenAIServingRender.render_completion",
    ),
}

_PIPELINE_METHOD_SPECS: dict[str, tuple[str, ...]] = {
    "vllm/v1/engine/llm_engine.py": (
        "LLMEngine.get_supported_tasks",
        "LLMEngine.add_request",
        "LLMEngine.step",
    ),
    "vllm/v1/engine/async_llm.py": (
        "AsyncLLM.add_request",
        "AsyncLLM.generate",
        "AsyncLLM.encode",
    ),
    "vllm/v1/engine/input_processor.py": (
        "InputProcessor.process_inputs",
    ),
    "vllm/v1/engine/core.py": (
        "EngineCore.add_request",
    ),
    "vllm/v1/core/sched/scheduler.py": (
        "Scheduler.schedule",
    ),
    "vllm/v1/attention/selector.py": (
        "get_attn_backend",
        "_cached_get_attn_backend",
        "get_mamba_attn_backend",
        "_cached_get_mamba_attn_backend",
    ),
}

_LAYER_METHOD_SPECS: dict[str, tuple[str, ...]] = {
    "vllm/model_executor/layers/layernorm.py": (
        "RMSNorm.forward_native",
    ),
}

_PLATFORM_METHOD_SPECS: dict[str, tuple[str, ...]] = {
    "vllm/platforms/interface.py": (
        "Platform.get_attn_backend_cls",
        "Platform.get_default_ir_op_priority",
    ),
    "vllm/platforms/cuda.py": (
        "CudaPlatformBase.get_attn_backend_cls",
        "CudaPlatformBase.get_default_ir_op_priority",
    ),
    "vllm/platforms/rocm.py": (
        "RocmPlatform.get_attn_backend_cls",
        "RocmPlatform.get_default_ir_op_priority",
    ),
    "vllm/platforms/xpu.py": (
        "XPUPlatform.get_attn_backend_cls",
        "XPUPlatform.get_default_ir_op_priority",
    ),
    "vllm/platforms/cpu.py": (
        "CpuPlatform.get_attn_backend_cls",
    ),
}

_MODEL_REGISTRY_VARS = {
    "_TEXT_GENERATION_MODELS": "text_generation",
    "_EMBEDDING_MODELS": "embedding",
    "_MULTIMODAL_MODELS": "multimodal",
    "_SPECULATIVE_DECODING_MODELS": "speculative",
}

_ATTENTION_ENUM_CLASSES = {
    "AttentionBackendEnum": "attention",
    "MambaAttentionBackendEnum": "mamba_attention",
}

_CPP_DEF_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\.def\s*\(")
_CPP_IMPL_PATTERN = re.compile(
    r'([A-Za-z_][A-Za-z0-9_]*)\.impl\(\s*"([^"]+)"(?P<body>.*?)\);',
    re.S,
)
_PROVIDER_PATTERN = re.compile(
    r'@ir\.ops\.(\w+)\.register_impl\(\s*"([^"]+)"(?P<body>.*?)\)\s*\ndef\s+'
    r"([A-Za-z_][A-Za-z0-9_]*)",
    re.S,
)


def build_vllm_index(source: str) -> dict[str, Any]:
    """Build a static-first index for authoritative vLLM mapping layers."""

    root = Path(source).resolve()
    records_by_family: dict[str, list[dict[str, Any]]] = {
        "api_entrypoints": _extract_selected_python_records(
            root, _API_METHOD_SPECS, "api_entrypoints"
        ),
        "request_pipeline_nodes": _extract_selected_python_records(
            root, _PIPELINE_METHOD_SPECS, "request_pipeline_nodes"
        ),
        "layer_nodes": _extract_selected_python_records(
            root, _LAYER_METHOD_SPECS, "layer_nodes"
        ),
        "platform_defaults": _extract_selected_python_records(
            root, _PLATFORM_METHOD_SPECS, "platform_defaults"
        ),
        "model_architectures": _extract_model_registry_records(root),
        "attention_backends": _extract_attention_backend_records(root),
        "ir_ops": _extract_ir_ops(root),
        "ir_providers": _extract_ir_providers(root),
        "custom_ops": _extract_decorated_class_records(
            root,
            "vllm/model_executor",
            "CustomOp",
            "custom_ops",
        ),
        "pluggable_layers": _extract_decorated_class_records(
            root,
            "vllm/model_executor",
            "PluggableLayer",
            "pluggable_layers",
        ),
        "torch_custom_ops": _extract_native_bindings(root),
    }

    lookup_indexes = _build_lookup_indexes(records_by_family)
    graph_payload = _build_graph(records_by_family, lookup_indexes)

    return {
        "metadata": {
            "framework": "vllm",
            "source_path": str(root),
        },
        **records_by_family,
        "lookup_indexes": lookup_indexes,
        "proof_traces": graph_payload["proof_traces"],
        "graph_nodes": graph_payload["graph_nodes"],
        "graph_edges": graph_payload["graph_edges"],
        "dynamic_notes": graph_payload["dynamic_notes"],
    }


def _extract_selected_python_records(
    root: Path,
    specs: dict[str, tuple[str, ...]],
    family: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative_path, qualified_names in specs.items():
        path = root / relative_path
        records.extend(_extract_python_members(path, family, set(qualified_names)))
    return records


def _extract_python_members(
    path: Path,
    family: str,
    qualified_names: set[str],
) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))

    records: list[dict[str, Any]] = []
    top_level_funcs = {name for name in qualified_names if "." not in name}
    class_method_targets: dict[str, set[str]] = {}
    for qualname in qualified_names:
        if "." not in qualname:
            continue
        owner, method_name = qualname.split(".", 1)
        class_method_targets.setdefault(owner, set()).add(method_name)

    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in top_level_funcs
        ):
            records.append(
                _record(
                    family=family,
                    name=node.name,
                    file_path=str(path),
                    line_number=node.lineno,
                    kind="python_function",
                    async_function=isinstance(node, ast.AsyncFunctionDef),
                )
            )
        elif isinstance(node, ast.ClassDef):
            methods = class_method_targets.get(node.name)
            if not methods:
                continue
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name in methods
                ):
                    records.append(
                        _record(
                            family=family,
                            name=f"{node.name}.{item.name}",
                            file_path=str(path),
                            line_number=item.lineno,
                            kind="python_method",
                            async_function=isinstance(item, ast.AsyncFunctionDef),
                            owner=node.name,
                        )
                    )
    return records


def _extract_model_registry_records(root: Path) -> list[dict[str, Any]]:
    path = root / "vllm/model_executor/models/registry.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))
    records: list[dict[str, Any]] = []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            category = _MODEL_REGISTRY_VARS.get(target.id)
            if category is None or not isinstance(node.value, ast.Dict):
                continue
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                impl_module = None
                impl_class = None
                if isinstance(value, ast.Tuple) and len(value.elts) >= 2:
                    first, second = value.elts[0], value.elts[1]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        impl_module = first.value
                    if isinstance(second, ast.Constant) and isinstance(
                        second.value, str
                    ):
                        impl_class = second.value
                records.append(
                    _record(
                        family="model_architectures",
                        name=key.value,
                        file_path=str(path),
                        line_number=getattr(key, "lineno", node.lineno),
                        kind="model_architecture",
                        category=category,
                        impl_module=impl_module,
                        impl_class=impl_class,
                    )
                )
    return records


def _extract_attention_backend_records(root: Path) -> list[dict[str, Any]]:
    path = root / "vllm/v1/attention/backends/registry.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))
    records: list[dict[str, Any]] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        backend_group = _ATTENTION_ENUM_CLASSES.get(node.name)
        if backend_group is None:
            continue
        for item in node.body:
            if not isinstance(item, ast.Assign) or len(item.targets) != 1:
                continue
            target = item.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = _literal_value(item.value)
            records.append(
                _record(
                    family="attention_backends",
                    name=target.id,
                    file_path=str(path),
                    line_number=item.lineno,
                    kind="attention_backend",
                    backend_group=backend_group,
                    class_path=value,
                )
            )
    return records


def _extract_ir_ops(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ops_root = root / "vllm/ir/ops"
    for path in sorted(ops_root.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text, filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            has_register_op, op_name, allow_inplace = _ir_decorator_metadata(
                node.decorator_list
            )
            if not has_register_op:
                continue
            records.append(
                _record(
                    family="ir_ops",
                    name=op_name or node.name,
                    file_path=str(path),
                    line_number=node.lineno,
                    kind="ir_op",
                    function_name=node.name,
                    allow_inplace=allow_inplace,
                )
            )
    return records


def _extract_ir_providers(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    kernels_root = root / "vllm/kernels"
    for path in sorted(kernels_root.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _PROVIDER_PATTERN.finditer(text):
            op_name, provider, body, function_name = match.groups()
            line_number = text[: match.start()].count("\n") + 1
            records.append(
                _record(
                    family="ir_providers",
                    name=f"{op_name}::{provider}",
                    file_path=str(path),
                    line_number=line_number,
                    kind="ir_provider",
                    op_name=op_name,
                    provider=provider,
                    function_name=function_name,
                    provider_args=" ".join(body.split()),
                )
            )
    return records


def _extract_decorated_class_records(
    root: Path,
    relative_root: str,
    decorator_owner: str,
    family: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    search_root = root / relative_root
    for path in sorted(search_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            registration_name = _decorated_registration_name(
                node.decorator_list,
                decorator_owner,
            )
            if registration_name is None:
                continue
            records.append(
                _record(
                    family=family,
                    name=registration_name,
                    file_path=str(path),
                    line_number=node.lineno,
                    kind=family[:-1],
                    class_name=node.name,
                    decorator_owner=decorator_owner,
                )
            )
    return records


def _extract_native_bindings(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "csrc").rglob("*torch_bindings.cpp")):
        text = path.read_text(encoding="utf-8", errors="replace")
        impl_map = _extract_cpp_impl_map(text)
        for (
            variable_name,
            schema,
            op_name,
            line_number,
        ) in _extract_cpp_definitions(text):
            torch_namespace = _torch_namespace_for_binding_var(variable_name)
            impl_info = impl_map.get(op_name, [])
            records.append(
                _record(
                    family="torch_custom_ops",
                    name=op_name,
                    file_path=str(path),
                    line_number=line_number,
                    kind="torch_custom_op",
                    schema=schema,
                    binding_var=variable_name,
                    torch_namespace=torch_namespace,
                    torch_symbol=f"torch.ops.{torch_namespace}.{op_name}",
                    impls=impl_info,
                    id_suffix=Path(path).parent.name,
                )
            )
    return records


def _build_lookup_indexes(
    records_by_family: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, list[str]] = {}
    by_family: dict[str, list[str]] = {}

    for family, records in records_by_family.items():
        family_ids: list[str] = []
        for record in records:
            record_id = record["id"]
            by_id[record_id] = record
            family_ids.append(record_id)
            for key in _search_keys(record):
                by_name.setdefault(key, []).append(record_id)
        by_family[family] = family_ids

    return {
        "by_id": by_id,
        "by_name": by_name,
        "by_family": by_family,
    }


def _build_graph(
    records_by_family: dict[str, list[dict[str, Any]]],
    lookup_indexes: dict[str, Any],
) -> dict[str, Any]:
    by_id = lookup_indexes["by_id"]
    graph_nodes = dict(by_id)
    graph_edges: list[dict[str, Any]] = []
    dynamic_notes = [
        (
            "attention backend selection is conditional on platform, "
            "configuration, and backend validation"
        ),
        (
            "ir provider dispatch depends on platform defaults and provider "
            "priority configuration"
        ),
        "custom op enablement depends on compilation config and platform support",
    ]

    def find_id(family: str, name: str, *, id_suffix: str | None = None) -> str:
        for record in records_by_family.get(family, []):
            if record["name"] != name:
                continue
            if id_suffix and not record["id"].endswith(f":{id_suffix}"):
                continue
            return record["id"]
        raise KeyError(f"Missing graph node for {family}:{name}")

    def add_edge(
        source_id: str,
        target_id: str,
        *,
        conditions: dict[str, Any] | None = None,
        confidence: str = "high",
        notes: str = "",
    ) -> None:
        graph_edges.append(
            {
                "source": source_id,
                "target": target_id,
                "conditions": conditions or {},
                "confidence": confidence,
                "notes": notes,
                "evidence": _edge_evidence(by_id[source_id], by_id[target_id]),
            }
        )

    # Synthetic nodes needed for lower-level proof traces.
    torch_op_node = _synthetic_node(
        family="runtime_nodes",
        name="torch.ops._C.rms_norm",
        file_path=by_id[
            find_id("torch_custom_ops", "rms_norm", id_suffix="libtorch_stable")
        ]["file_path"],
        line_number=by_id[
            find_id("torch_custom_ops", "rms_norm", id_suffix="libtorch_stable")
        ]["line_number"],
        kind="runtime_anchor",
        description="Torch custom op dispatch anchor for rms_norm",
    )
    graph_nodes[torch_op_node["id"]] = torch_op_node
    by_id[torch_op_node["id"]] = torch_op_node

    scheduler_id = find_id("request_pipeline_nodes", "Scheduler.schedule")
    engine_add_request_id = find_id("request_pipeline_nodes", "EngineCore.add_request")
    input_process_id = find_id(
        "request_pipeline_nodes",
        "InputProcessor.process_inputs",
    )
    async_add_request_id = find_id("request_pipeline_nodes", "AsyncLLM.add_request")
    async_generate_id = find_id("request_pipeline_nodes", "AsyncLLM.generate")
    async_encode_id = find_id("request_pipeline_nodes", "AsyncLLM.encode")
    llm_generate_id = find_id("api_entrypoints", "LLM.generate")
    llm_run_completion_id = find_id("api_entrypoints", "LLM._run_completion")
    llm_add_request_id = find_id("api_entrypoints", "LLM._add_request")
    llm_engine_add_request_id = find_id(
        "request_pipeline_nodes",
        "LLMEngine.add_request",
    )
    llm_engine_step_id = find_id("request_pipeline_nodes", "LLMEngine.step")
    chat_create_id = find_id(
        "api_entrypoints",
        "OpenAIServingChat._create_chat_completion",
    )
    pooling_prepare_id = find_id(
        "api_entrypoints",
        "PoolingServingBase._prepare_generators",
    )
    rms_layer_id = find_id("layer_nodes", "RMSNorm.forward_native")
    rms_ir_id = find_id("ir_ops", "rms_norm")
    rms_provider_id = find_id("ir_providers", "rms_norm::vllm_c")
    rms_binding_id = find_id(
        "torch_custom_ops",
        "rms_norm",
        id_suffix="libtorch_stable",
    )
    selector_id = find_id("request_pipeline_nodes", "get_attn_backend")
    selector_cached_id = find_id("request_pipeline_nodes", "_cached_get_attn_backend")
    platform_selector_ids = [
        find_id("platform_defaults", "Platform.get_attn_backend_cls"),
        find_id("platform_defaults", "CudaPlatformBase.get_attn_backend_cls"),
        find_id("platform_defaults", "RocmPlatform.get_attn_backend_cls"),
        find_id("platform_defaults", "XPUPlatform.get_attn_backend_cls"),
        find_id("platform_defaults", "CpuPlatform.get_attn_backend_cls"),
    ]
    fused_moe_id = find_id("pluggable_layers", "fused_moe")
    unquantized_moe_id = find_id("custom_ops", "unquantized_fused_moe")
    moe_sum_binding_id = find_id("torch_custom_ops", "moe_sum", id_suffix="moe")

    add_edge(chat_create_id, async_generate_id)
    add_edge(async_generate_id, async_add_request_id)
    add_edge(async_add_request_id, input_process_id)
    add_edge(input_process_id, engine_add_request_id)
    add_edge(engine_add_request_id, scheduler_id)
    add_edge(
        scheduler_id,
        rms_layer_id,
        notes="Model forward anchor through the scheduled execution path",
    )
    add_edge(rms_layer_id, rms_ir_id)
    add_edge(rms_ir_id, rms_provider_id, conditions={"provider": "vllm_c"})
    add_edge(rms_provider_id, torch_op_node["id"], conditions={"torch_namespace": "_C"})
    add_edge(torch_op_node["id"], rms_binding_id)

    add_edge(llm_generate_id, llm_run_completion_id)
    add_edge(llm_run_completion_id, llm_add_request_id)
    add_edge(llm_add_request_id, llm_engine_add_request_id)
    add_edge(llm_engine_add_request_id, input_process_id)
    add_edge(
        llm_generate_id,
        llm_engine_step_id,
        notes="Offline generation repeatedly steps the engine",
    )

    add_edge(pooling_prepare_id, async_encode_id, conditions={"runner_type": "pooling"})
    add_edge(
        async_encode_id,
        async_add_request_id,
        conditions={"runner_type": "pooling"},
    )

    add_edge(selector_id, selector_cached_id)
    for platform_selector_id in platform_selector_ids:
        add_edge(
            selector_cached_id,
            platform_selector_id,
            conditions={
                "platform_dispatch": by_id[platform_selector_id]["name"].split(".")[0]
            },
            confidence="conditional",
        )
    for backend in records_by_family["attention_backends"]:
        if backend["name"] == "CUSTOM":
            continue
        add_edge(
            selector_cached_id,
            backend["id"],
            conditions={"selected_backend": backend["name"]},
            confidence="conditional",
            notes=(
                "Backend selection depends on platform defaults and backend "
                "validation"
            ),
        )

    add_edge(
        fused_moe_id,
        unquantized_moe_id,
        conditions={"moe_family": "unquantized"},
        confidence="conditional",
    )
    add_edge(
        unquantized_moe_id,
        moe_sum_binding_id,
        conditions={"backend_family": "moe"},
        confidence="conditional",
    )

    proof_traces = {
        "chat_completion_to_rms_norm": _proof_trace(
            "chat_completion_to_rms_norm",
            "Chat completion to rms_norm",
            [
                chat_create_id,
                async_generate_id,
                async_add_request_id,
                input_process_id,
                engine_add_request_id,
                scheduler_id,
                rms_layer_id,
                rms_ir_id,
                rms_provider_id,
                torch_op_node["id"],
                rms_binding_id,
            ],
        ),
        "offline_generate": _proof_trace(
            "offline_generate",
            "Offline LLM.generate()",
            [
                llm_generate_id,
                llm_run_completion_id,
                llm_add_request_id,
                llm_engine_add_request_id,
                input_process_id,
                engine_add_request_id,
                llm_engine_step_id,
            ],
        ),
        "pooling_encode": _proof_trace(
            "pooling_encode",
            "Pooling encode flow",
            [
                pooling_prepare_id,
                async_encode_id,
                async_add_request_id,
                input_process_id,
                engine_add_request_id,
            ],
        ),
        "attention_backend_selection": _proof_trace(
            "attention_backend_selection",
            "Attention backend selection",
            [
                selector_id,
                selector_cached_id,
                platform_selector_ids[1],
            ],
        ),
        "fused_moe_family": _proof_trace(
            "fused_moe_family",
            "One fused_moe family path",
            [
                fused_moe_id,
                unquantized_moe_id,
                moe_sum_binding_id,
            ],
        ),
    }

    return {
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "proof_traces": proof_traces,
        "dynamic_notes": dynamic_notes,
    }


def _edge_evidence(source: dict[str, Any], target: dict[str, Any]) -> list[str]:
    return [
        f"{source['name']} → {source['file_path']}:{source['line_number']}",
        f"{target['name']} → {target['file_path']}:{target['line_number']}",
    ]


def _proof_trace(trace_id: str, title: str, node_ids: list[str]) -> dict[str, Any]:
    return {
        "id": trace_id,
        "title": title,
        "steps": node_ids,
    }


def _record(
    *,
    family: str,
    name: str,
    file_path: str,
    line_number: int,
    kind: str,
    id_suffix: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    suffix = f":{id_suffix}" if id_suffix else ""
    return {
        "id": f"{family}:{name}{suffix}",
        "family": family,
        "name": name,
        "file_path": file_path,
        "line_number": line_number,
        "kind": kind,
        "details": details,
    }


def _synthetic_node(
    *,
    family: str,
    name: str,
    file_path: str,
    line_number: int,
    kind: str,
    **details: Any,
) -> dict[str, Any]:
    return _record(
        family=family,
        name=name,
        file_path=file_path,
        line_number=line_number,
        kind=kind,
        **details,
    )


def _literal_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if node.value is None:
            return None
    return None


def _ir_decorator_metadata(
    decorators: list[ast.expr],
) -> tuple[bool, str | None, bool]:
    for decorator in decorators:
        if isinstance(decorator, ast.Name) and decorator.id == "register_op":
            return True, None, False
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            if decorator.func.id != "register_op":
                continue
            op_name = None
            allow_inplace = False
            for keyword in decorator.keywords:
                if (
                    keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    op_name = keyword.value.value
                if keyword.arg == "allow_inplace" and isinstance(
                    keyword.value, ast.Constant
                ):
                    allow_inplace = bool(keyword.value.value)
            return True, op_name, allow_inplace
    return False, None, False


def _decorated_registration_name(
    decorators: list[ast.expr],
    decorator_owner: str,
) -> str | None:
    for decorator in decorators:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != "register":
            continue
        owner = func.value
        if not isinstance(owner, ast.Name) or owner.id != decorator_owner:
            continue
        if not decorator.args:
            continue
        first_arg = decorator.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value
    return None


def _extract_cpp_definitions(text: str) -> list[tuple[str, str, str, int]]:
    definitions: list[tuple[str, str, str, int]] = []
    for match in _CPP_DEF_PATTERN.finditer(text):
        variable_name = match.group(1)
        end = text.find(");", match.start())
        if end == -1:
            continue
        block = text[match.start() : end]
        string_parts = re.findall(r'"([^"]*)"', block)
        if not string_parts:
            continue
        schema = "".join(string_parts)
        op_name_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\(", schema)
        if not op_name_match:
            continue
        op_name = op_name_match.group(1)
        line_number = text[: match.start()].count("\n") + 1
        definitions.append((variable_name, schema, op_name, line_number))
    return definitions


def _extract_cpp_impl_map(text: str) -> dict[str, list[dict[str, Any]]]:
    impls: dict[str, list[dict[str, Any]]] = {}
    for match in _CPP_IMPL_PATTERN.finditer(text):
        variable_name, op_name, body = match.groups()
        line_number = text[: match.start()].count("\n") + 1
        normalized = " ".join(body.split())
        impl_name_match = re.search(r"&([A-Za-z_][A-Za-z0-9_]*)", normalized)
        backend_match = re.search(r"torch::k([A-Za-z0-9_]+)", normalized)
        impls.setdefault(op_name, []).append(
            {
                "binding_var": variable_name,
                "line_number": line_number,
                "impl_name": impl_name_match.group(1) if impl_name_match else "",
                "backend": backend_match.group(1) if backend_match else "",
            }
        )
    return impls


def _torch_namespace_for_binding_var(variable_name: str) -> str:
    if variable_name == "cache_ops":
        return "_C_cache_ops"
    if variable_name == "cuda_utils":
        return "_C_cuda_utils"
    if variable_name == "custom_ar":
        return "_C_custom_ar"
    return "_C"


def _search_keys(record: dict[str, Any]) -> set[str]:
    keys = {record["name"].lower()}
    keys.add(record["name"].split(".")[-1].lower())
    keys.add(record["name"].split("::")[0].lower())
    details = record.get("details", {})
    for value in details.values():
        if isinstance(value, str) and value:
            keys.add(value.lower())
    return {key for key in keys if key}

