from __future__ import annotations

from typing import Dict
import networkx as nx
import pandas as pd


def node_centrality_table(G: nx.Graph, membership: Dict[int, int]) -> pd.DataFrame:
    deg = dict(G.degree())
    pr = nx.pagerank(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    clo = nx.closeness_centrality(G)

    df = pd.DataFrame(
        {
            "node_id": [int(n) for n in G.nodes()],
            "community_id": [int(membership[int(n)]) for n in G.nodes()],
            "degree": [float(deg[n]) for n in G.nodes()],
            "pagerank": [float(pr[n]) for n in G.nodes()],
            "betweenness": [float(btw[n]) for n in G.nodes()],
            "closeness": [float(clo[n]) for n in G.nodes()],
        }
    )
    return df