from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def build_node_features(G: nx.Graph) -> pd.DataFrame:
    nodes = sorted(G.nodes())

    deg = dict(G.degree())
    clustering = nx.clustering(G)
    pr = nx.pagerank(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    clo = nx.closeness_centrality(G)

    rows = []
    for n in nodes:
        rows.append(
            {
                "node_id": int(n),
                "degree": float(deg[n]),
                "clustering": float(clustering[n]),
                "pagerank": float(pr[n]),
                "betweenness": float(btw[n]),
                "closeness": float(clo[n]),
            }
        )
    return pd.DataFrame(rows).set_index("node_id")


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")

    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X, metric="euclidean")

    max_diam = 0.0
    for c in unique:
        idx = np.where(labels == c)[0]
        if idx.size <= 1:
            continue
        diam = D[np.ix_(idx, idx)].max()
        max_diam = max(max_diam, float(diam))
    if max_diam == 0.0:
        return float("nan")

    min_inter = float("inf")
    for i, c1 in enumerate(unique):
        idx1 = np.where(labels == c1)[0]
        for c2 in unique[i + 1 :]:
            idx2 = np.where(labels == c2)[0]
            dist = D[np.ix_(idx1, idx2)].min()
            min_inter = min(min_inter, float(dist))

    return float(min_inter / max_diam) if np.isfinite(min_inter) else float("nan")


def compute_clustering_metrics(features_df: pd.DataFrame, membership: Dict[int, int]) -> dict:
    node_ids = features_df.index.to_numpy()
    labels = np.array([membership[int(n)] for n in node_ids], dtype=int)

    X = features_df.to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    if len(np.unique(labels)) < 2:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "dunn": float("nan"),
            "n_clusters": int(len(np.unique(labels))),
        }

    sil = silhouette_score(Xs, labels)
    db = davies_bouldin_score(Xs, labels)
    dunn = dunn_index(Xs, labels)

    return {
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "dunn": float(dunn),
        "n_clusters": int(len(np.unique(labels))),
    }