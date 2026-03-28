from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd


def node_centrality_table(G: nx.Graph, membership: Dict[int, int]) -> pd.DataFrame:
    """
    Tính 6 centrality measures cho mỗi node
    
    Columns:
    - node_id
    - community_id
    - degree: số bạn
    - pagerank: ảnh hưởng
    - betweenness: nằm giữa các đường
    - closeness: gần các node khác
    - clustering: hình thành tam giác
    """
    deg = dict(G.degree())
    pr = nx.pagerank(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    clo = nx.closeness_centrality(G)
    clustering = nx.clustering(G)

    df = pd.DataFrame(
        {
            "node_id": [int(n) for n in G.nodes()],
            "community_id": [int(membership[int(n)]) for n in G.nodes()],
            "degree": [float(deg[n]) for n in G.nodes()],
            "pagerank": [float(pr[n]) for n in G.nodes()],
            "betweenness": [float(btw[n]) for n in G.nodes()],
            "closeness": [float(clo[n]) for n in G.nodes()],
            "clustering_coeff": [float(clustering[n]) for n in G.nodes()],
        }
    )
    return df