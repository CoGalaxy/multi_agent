from __future__ import annotations

from collections import defaultdict

from verifiable_multi_agent.topology.graph import AgentType, CollaborationGraph


def validate_graph(graph: CollaborationGraph) -> list[str]:
    errors: list[str] = []
    constraints = graph.constraints
    if graph.blocked:
        return errors

    node_ids = graph.node_ids()
    node_types = set(graph.node_types())

    if len(graph.nodes) > constraints.max_nodes:
        errors.append(f"too_many_nodes:{len(graph.nodes)}>{constraints.max_nodes}")
    if len(graph.edges) > constraints.max_edges:
        errors.append(f"too_many_edges:{len(graph.edges)}>{constraints.max_edges}")
    if constraints.require_verifier and AgentType.VERIFIER.value not in node_types:
        errors.append("missing_verifier")
    if constraints.require_synthesizer and AgentType.SYNTHESIZER.value not in node_types:
        errors.append("missing_synthesizer")
    if not graph.entry_node or graph.entry_node not in node_ids:
        errors.append("invalid_entry_node")
    if not graph.exit_node or graph.exit_node not in node_ids:
        errors.append("invalid_exit_node")
    for edge in graph.edges:
        if edge.source not in node_ids:
            errors.append(f"invalid_edge_source:{edge.source}")
        if edge.target not in node_ids:
            errors.append(f"invalid_edge_target:{edge.target}")
    if _has_cycle(graph):
        errors.append("cycle_not_allowed")
    return errors


def _has_cycle(graph: CollaborationGraph) -> bool:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for target in adjacency[node_id]:
            if visit(target):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(node.id) for node in graph.nodes)
