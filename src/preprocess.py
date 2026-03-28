from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import networkx as nx
import pandas as pd


@dataclass
class GraphStats:
    """Thống kê đồ thị"""
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    is_connected: bool


def preprocess_graph(G: nx.Graph, processed_dir: Path = None) -> nx.Graph:
    """
    Làm sạch đồ thị
    
    Công việc:
    1. Xoá self-loops (cạnh A-A)
    2. Remove isolated nodes
    3. Make undirected (nếu directed)
    4. Keep largest connected component
    5. Convert node IDs to integer
    
    Args:
        G: NetworkX Graph
        processed_dir: thư mục lưu (optional)
    
    Returns:
        Cleaned graph
    """
    print("   - Removing self-loops...")
    G.remove_edges_from(nx.selfloop_edges(G))
    
    print("   - Removing isolated nodes...")
    isolated = list(nx.isolates(G))
    if isolated:
        G.remove_nodes_from(isolated)
    
    print("   - Converting to undirected...")
    if G.is_directed():
        G = G.to_undirected()
    
    print("   - Keeping largest connected component...")
    components = list(nx.connected_components(G))
    if len(components) > 1:
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()
    
    # Ensure integer node IDs
    try:
        G = nx.convert_node_labels_to_integers(G)
    except:
        pass
    
    return G


def compute_graph_stats(G: nx.Graph) -> GraphStats:
    """
    Tính thống kê cơ bản của đồ thị
    
    Args:
        G: NetworkX Graph
    
    Returns:
        GraphStats object
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    avg_degree = (2 * n_edges / n_nodes) if n_nodes > 0 else 0
    is_connected = nx.is_connected(G)
    
    return GraphStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        density=float(density),
        avg_degree=float(avg_degree),
        is_connected=is_connected
    )


def export_graph_stats(G: nx.Graph, out_path: Path) -> None:
    """
    Export graph statistics to CSV
    """
    stats = compute_graph_stats(G)
    
    df = pd.DataFrame([{
        'n_nodes': stats.n_nodes,
        'n_edges': stats.n_edges,
        'density': stats.density,
        'avg_degree': stats.avg_degree,
        'is_connected': stats.is_connected
    }])
    
    df.to_csv(out_path, index=False)
    print(f"✓ Graph stats saved: {out_path}")