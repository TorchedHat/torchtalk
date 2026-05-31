"""vLLM tool helpers built on the framework-aware TorchTalk state."""

from __future__ import annotations

from collections import deque

from ..formatting import create_formatter, relative_path
from ..indexer import _ensure_capability, _source_base, _state

_SEARCH_FAMILIES = {
    "bindings": ("torch_custom_ops",),
    "apis": (
        "api_entrypoints",
        "request_pipeline_nodes",
        "layer_nodes",
        "platform_defaults",
    ),
    "models": ("model_architectures",),
    "backends": ("attention_backends", "platform_defaults"),
    "ops": ("ir_ops", "ir_providers", "custom_ops", "pluggable_layers"),
}


def _rel_path(path: str) -> str:
    return relative_path(path, _source_base())


def _records_by_id() -> dict[str, dict]:
    return _state.indexes.get("by_id", {})


def _records_by_name() -> dict[str, list[str]]:
    return _state.indexes.get("by_name", {})


def _graph_nodes() -> dict[str, dict]:
    return _state.graph_nodes


def _graph_edges() -> list[dict]:
    return _state.graph_edges


def _graph_maps() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    outbound: dict[str, list[dict]] = {}
    inbound: dict[str, list[dict]] = {}
    for edge in _graph_edges():
        outbound.setdefault(edge["source"], []).append(edge)
        inbound.setdefault(edge["target"], []).append(edge)
    return outbound, inbound


def _matches_query(record: dict, query: str) -> bool:
    needle = query.lower()
    if needle in record.get("name", "").lower():
        return True
    details = record.get("details", {})
    for value in details.values():
        if isinstance(value, str) and needle in value.lower():
            return True
    return False


def _candidate_records(
    query: str,
    families: tuple[str, ...] | None = None,
) -> list[dict]:
    by_id = _records_by_id()
    exact_ids = _records_by_name().get(query.lower(), [])
    candidates = [by_id[record_id] for record_id in exact_ids if record_id in by_id]

    if families:
        candidates = [
            record for record in candidates if record.get("family") in families
        ]

    if candidates:
        return candidates

    for record in by_id.values():
        if families and record.get("family") not in families:
            continue
        if _matches_query(record, query):
            candidates.append(record)
    candidates.sort(key=lambda record: (record["family"], len(record["name"])))
    return candidates


def _format_conditions(edge: dict) -> str:
    conditions = edge.get("conditions") or {}
    if not conditions:
        return ""
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(conditions.items()))
    return f" [{rendered}]"


async def search(query: str, mode: str = "bindings", limit: int = 10) -> str:
    _ensure_capability("search_bindings")

    families = _SEARCH_FAMILIES.get(mode)
    candidates = _candidate_records(query, families=families)
    if not candidates:
        scope = f" in mode '{mode}'" if mode != "bindings" else ""
        return f"No vLLM entities found matching `{query}`{scope}."

    md = create_formatter()
    title = f"vLLM Search: `{query}`"
    if mode != "bindings":
        title += f" ({mode})"
    md.h2(title)
    md.text(f"Found {len(candidates)} result(s)\n")

    for record in candidates[:limit]:
        path = _rel_path(record["file_path"])
        md.item(
            f"**{record['name']}** [{record['family']}] → "
            f"`{path}:{record['line_number']}`"
        )
        details = record.get("details", {})
        detail_bits = []
        for key in (
            "category",
            "impl_module",
            "impl_class",
            "backend_group",
            "provider",
            "torch_symbol",
        ):
            value = details.get(key)
            if value:
                detail_bits.append(f"{key}={value}")
        if detail_bits:
            md.item(", ".join(detail_bits), 1)

    if len(candidates) > limit:
        md.text(f"\n*Showing {limit} of {len(candidates)} results*")
    return md.build()


async def trace(name: str, focus: str = "full") -> str:
    _ensure_capability("trace_apis")

    md = create_formatter()
    md.h2(f"vLLM Trace: `{name}`")

    matches = _matching_trace_segments(name)
    if matches:
        trace_payload, start_index = matches[0]
        md.h3(trace_payload["title"])
        steps = trace_payload["steps"][start_index:]
        for idx, node_id in enumerate(steps, start=1):
            node = _graph_nodes()[node_id]
            path = _rel_path(node["file_path"])
            md.item(
                f"{idx}. `{node['name']}` [{node['family']}] → "
                f"`{path}:{node['line_number']}`"
            )
            if idx < len(steps):
                edge = _edge_between(steps[idx - 1], steps[idx])
                if edge:
                    md.item(
                        f"conditions={_format_conditions(edge).strip() or 'none'} "
                        f"confidence={edge.get('confidence', 'high')}",
                        1,
                    )
        return md.build()

    candidates = _candidate_records(name)
    if not candidates:
        return f"vLLM entity `{name}` not found."

    record = candidates[0]
    path = _rel_path(record["file_path"])
    md.item(
        f"`{record['name']}` [{record['family']}] → "
        f"`{path}:{record['line_number']}`"
    )
    details = record.get("details", {})
    if details:
        for key, value in sorted(details.items()):
            if value not in (None, "", [], {}):
                md.item(f"{key}: `{value}`", 1)

    outbound, inbound = _graph_maps()
    if record["id"] in outbound:
        md.h3("Downstream")
        for edge in outbound[record["id"]][:10]:
            target = _graph_nodes()[edge["target"]]
            md.item(f"`{target['name']}`{_format_conditions(edge)}", 1)
    if focus == "full" and record["id"] in inbound:
        md.h3("Upstream")
        for edge in inbound[record["id"]][:10]:
            source = _graph_nodes()[edge["source"]]
            md.item(f"`{source['name']}`{_format_conditions(edge)}", 1)
    return md.build()


async def graph(function_name: str, mode: str = "callers", depth: int = 2) -> str:
    _ensure_capability("graph_flow")

    candidates = _candidate_records(function_name)
    if not candidates:
        return f"No vLLM graph node found for `{function_name}`."
    node = candidates[0]
    outbound, inbound = _graph_maps()

    if mode == "calls":
        edges = outbound.get(node["id"], [])
        return _render_edge_list(f"vLLM Calls: `{node['name']}`", edges, "target")
    if mode == "impact":
        return _render_impact(node["id"], depth, inbound)
    edges = inbound.get(node["id"], [])
    return _render_edge_list(f"vLLM Callers: `{node['name']}`", edges, "source")


def _matching_trace_segments(name: str) -> list[tuple[dict, int]]:
    needle = name.lower()
    ranked_matches: list[tuple[tuple[int, int, str], tuple[dict, int]]] = []
    for trace_payload in _state.proof_traces.values():
        for index, node_id in enumerate(trace_payload.get("steps", [])):
            node = _graph_nodes().get(node_id)
            if node and _matches_query(node, name):
                exact = 0 if node["name"].lower() == needle else 1
                ranked_matches.append(
                    (
                        (exact, index, trace_payload["id"]),
                        (trace_payload, index),
                    )
                )
                break
    ranked_matches.sort(key=lambda item: item[0])
    return [item[1] for item in ranked_matches]


def _edge_between(source_id: str, target_id: str) -> dict | None:
    for edge in _graph_edges():
        if edge["source"] == source_id and edge["target"] == target_id:
            return edge
    return None


def _render_edge_list(title: str, edges: list[dict], key: str) -> str:
    if not edges:
        return f"No vLLM graph edges found for {title.split(':', 1)[-1].strip()}."

    md = create_formatter()
    md.h2(title)
    for edge in edges[:15]:
        node = _graph_nodes()[edge[key]]
        path = _rel_path(node["file_path"])
        md.item(
            f"`{node['name']}` [{node['family']}] → `{path}:{node['line_number']}`"
            f"{_format_conditions(edge)}"
        )
        md.item(f"confidence={edge.get('confidence', 'high')}", 1)
    if len(edges) > 15:
        md.item(f"*... and {len(edges) - 15} more*", 1)
    return md.build()


def _render_impact(node_id: str, depth: int, inbound: dict[str, list[dict]]) -> str:
    md = create_formatter()
    origin = _graph_nodes()[node_id]
    md.h2(f"vLLM Impact: `{origin['name']}`")

    visited = {node_id}
    queue = deque([(node_id, 0)])
    callers_by_depth: dict[int, list[dict]] = {}

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        for edge in inbound.get(current_id, []):
            source_id = edge["source"]
            if source_id in visited:
                continue
            visited.add(source_id)
            callers_by_depth.setdefault(current_depth + 1, []).append(edge)
            queue.append((source_id, current_depth + 1))

    if not callers_by_depth:
        return f"No vLLM callers found for `{origin['name']}`."

    for level in sorted(callers_by_depth):
        md.h3(f"Depth {level}")
        for edge in callers_by_depth[level][:15]:
            source = _graph_nodes()[edge["source"]]
            path = _rel_path(source["file_path"])
            md.item(
                f"`{source['name']}` [{source['family']}] → "
                f"`{path}:{source['line_number']}`"
                f"{_format_conditions(edge)}"
            )
        if len(callers_by_depth[level]) > 15:
            md.item(f"*... and {len(callers_by_depth[level]) - 15} more*", 1)
        md.blank()

    return md.build()
