from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd


def community_profile_table(G: nx.Graph, membership: Dict[int, int], top_k: int = 5) -> pd.DataFrame:
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