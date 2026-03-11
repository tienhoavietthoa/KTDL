from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass
class GraphStats:
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    is_connected: bool


def compute_basic_stats(G: nx.Graph) -> GraphStats:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    avg_degree = sum(dict(G.degree()).values()) / n if n > 0 else 0.0
    is_connected = nx.is_connected(G) if not G.is_directed() else nx.is_weakly_connected(G)
    return GraphStats(n, m, float(density), float(avg_degree), bool(is_connected))


def load_and_preprocess(graphml_path: Path, processed_dir: Path) -> tuple[nx.Graph, GraphStats, GraphStats, Path]:
    G = nx.read_graphml(graphml_path)

    # GraphML often reads node ids as strings -> convert to int if possible
    mapping = {}
    for n in G.nodes():
        try:
            mapping[n] = int(n)
        except Exception:
            mapping[n] = n
    G = nx.relabel_nodes(G, mapping)

    before = compute_basic_stats(G)

    if G.is_directed():
        G = G.to_undirected()

    G.remove_edges_from(list(nx.selfloop_edges(G)))

    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    after = compute_basic_stats(G)

    cleaned_path = processed_dir / "karate_clean.graphml"
    nx.write_graphml(G, cleaned_path)

    return G, before, after, cleaned_path