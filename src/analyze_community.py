from __future__ import annotations

from typing import Dict
from collections import Counter

import networkx as nx
import pandas as pd


def community_profile_table(G: nx.Graph, membership: Dict[int, int], top_k: int = 5) -> pd.DataFrame:
    """
    Phân tích profile của mỗi cộng đồng
    
    Returns:
        DataFrame với columns:
        - community_id
        - size: số nodes
        - top_pagerank_nodes: top K influential nodes
        - top_betweenness_nodes: top K bridge nodes
    """
    deg = dict(G.degree())
    pr = nx.pagerank(G)
    btw = nx.betweenness_centrality(G, normalized=True)

    df = pd.DataFrame(
        {
            "node_id": [int(n) for n in G.nodes()],
            "community_id": [int(membership[int(n)]) for n in G.nodes()],
            "degree": [float(deg[n]) for n in G.nodes()],
            "pagerank": [float(pr[n]) for n in G.nodes()],
            "betweenness": [float(btw[n]) for n in G.nodes()],
        }
    )

    out_rows = []
    for cid, g in df.groupby("community_id"):
        g1 = g.sort_values("pagerank", ascending=False).head(top_k)
        g2 = g.sort_values("betweenness", ascending=False).head(top_k)
        out_rows.append(
            {
                "community_id": int(cid),
                "size": int(len(g)),
                "top_pagerank_nodes": ", ".join(map(str, g1["node_id"].tolist())),
                "top_betweenness_nodes": ", ".join(map(str, g2["node_id"].tolist())),
            }
        )

    return pd.DataFrame(out_rows).sort_values("size", ascending=False)


def detect_bridge_nodes(
    G: nx.Graph,
    membership: Dict[int, int],
    top_k: int = 5
) -> pd.DataFrame:
    """
    ✨ BRIDGE NODES: Tìm nodes nối nhiều communities nhất
    
    Bridge node = node có nhiều neighbors từ communities khác
    
    Ứng dụng:
    - Identify influential users giữa groups
    - Tìm bottlenecks trong network
    """
    
    bridge_scores = {}
    
    for node in G.nodes():
        node_comm = membership[node]
        neighbors = list(G.neighbors(node))
        
        # Count neighbors từ communities khác
        other_comm_neighbors = [n for n in neighbors if membership[n] != node_comm]
        bridge_scores[node] = len(other_comm_neighbors)
    
    # Sort by bridge score
    sorted_bridges = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    rows = []
    for node, score in sorted_bridges:
        neighbors = list(G.neighbors(node))
        neighbor_comms = Counter(
            membership[n] for n in neighbors if membership[n] != membership[node]
        )
        
        rows.append({
            'node_id': int(node),
            'community_id': int(membership[node]),
            'bridge_score': int(score),
            'degree': int(G.degree(node)),
            'connected_to_comms': len(neighbor_comms),
            'neighbor_communities': str(dict(neighbor_comms))
        })
    
    return pd.DataFrame(rows)


def analyze_inter_community_edges(G: nx.Graph, membership: Dict[int, int]) -> dict:
    """
    ✨ INTER-COMMUNITY STATS: Phân tích cạnh giữa các cộng đồng
    
    Returns:
        {
            'intra_edges': cạnh nội bộ,
            'inter_edges': cạnh giữa communities,
            'intra_ratio': phần trăm nội bộ,
            'inter_by_pair': cạnh giữa mỗi pair communities
        }
    """
    from collections import defaultdict
    
    intra_edges = 0
    inter_edges = 0
    inter_by_pair = defaultdict(int)
    
    for u, v in G.edges():
        u_comm = membership[u]
        v_comm = membership[v]
        
        if u_comm == v_comm:
            intra_edges += 1
        else:
            inter_edges += 1
            pair = tuple(sorted([u_comm, v_comm]))
            inter_by_pair[pair] += 1
    
    total_edges = G.number_of_edges()
    
    return {
        'intra_edges': int(intra_edges),
        'inter_edges': int(inter_edges),
        'total_edges': int(total_edges),
        'intra_ratio': float(intra_edges / total_edges) if total_edges > 0 else 0.0,
        'inter_ratio': float(inter_edges / total_edges) if total_edges > 0 else 0.0,
        'inter_by_pair': dict(inter_by_pair)
    }