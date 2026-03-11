from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class LinkPredResult:
    dataset: str
    n_nodes: int
    n_edges: int
    n_train_pos: int
    n_test_pos: int
    n_train_neg: int
    n_test_neg: int
    auc: float
    f1: float
    precision: float
    recall: float
    model_name: str


def _pairs_to_features(G: nx.Graph, pairs: List[Tuple[int, int]]) -> pd.DataFrame:
    """
    Compute classic heuristic link prediction features for each (u,v).
    Features:
      - common_neighbors
      - jaccard
      - adamic_adar
      - preferential_attachment
      - resource_allocation
    """
    # Common neighbors count
    cn = []
    for u, v in pairs:
        cn.append(len(list(nx.common_neighbors(G, u, v))))

    # Generators for other heuristics
    jac = {tuple(sorted((u, v))): p for u, v, p in nx.jaccard_coefficient(G, pairs)}
    aa = {tuple(sorted((u, v))): p for u, v, p in nx.adamic_adar_index(G, pairs)}
    pa = {tuple(sorted((u, v))): p for u, v, p in nx.preferential_attachment(G, pairs)}
    ra = {tuple(sorted((u, v))): p for u, v, p in nx.resource_allocation_index(G, pairs)}

    rows = []
    for i, (u, v) in enumerate(pairs):
        key = tuple(sorted((u, v)))
        rows.append(
            {
                "u": int(u),
                "v": int(v),
                "common_neighbors": float(cn[i]),
                "jaccard": float(jac.get(key, 0.0)),
                "adamic_adar": float(aa.get(key, 0.0)),
                "preferential_attachment": float(pa.get(key, 0.0)),
                "resource_allocation": float(ra.get(key, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _sample_negative_edges(G: nx.Graph, n_samples: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    nodes = list(G.nodes())
    node_set = set(nodes)
    edges = set((min(u, v), max(u, v)) for u, v in G.edges())
    neg = set()

    # rejection sampling
    while len(neg) < n_samples:
        u = int(rng.choice(nodes))
        v = int(rng.choice(nodes))
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if a not in node_set or b not in node_set:
            continue
        if (a, b) in edges:
            continue
        neg.add((a, b))
    return list(neg)


def train_link_predictor(
    G_full: nx.Graph,
    dataset_name: str,
    test_frac: float = 0.2,
    neg_ratio: int = 1,
    seed: int = 42,
) -> tuple[LinkPredResult, LogisticRegression, pd.DataFrame]:
    """
    Link prediction by:
    - Split positive edges into train/test
    - Sample negative edges with ratio neg_ratio
    - Compute heuristic features on train-graph (without test pos edges)
    - Train Logistic Regression
    Returns (result, model, test_scored_df)
    """
    rng = np.random.default_rng(seed)

    edges = list(G_full.edges())
    if len(edges) < 50:
        raise ValueError("Graph too small for link prediction demo (need more edges).")

    # Split positive edges
    train_pos, test_pos = train_test_split(edges, test_size=test_frac, random_state=seed, shuffle=True)

    # Build train graph WITHOUT test positive edges
    G_train = G_full.copy()
    G_train.remove_edges_from(test_pos)

    # Negative sampling
    n_train_neg = int(len(train_pos) * neg_ratio)
    n_test_neg = int(len(test_pos) * neg_ratio)

    train_neg = _sample_negative_edges(G_full, n_train_neg, rng)
    test_neg = _sample_negative_edges(G_full, n_test_neg, rng)

    # Create feature matrices
    X_train_pos = _pairs_to_features(G_train, [(int(u), int(v)) for u, v in train_pos])
    X_train_neg = _pairs_to_features(G_train, [(int(u), int(v)) for u, v in train_neg])
    X_test_pos = _pairs_to_features(G_train, [(int(u), int(v)) for u, v in test_pos])
    X_test_neg = _pairs_to_features(G_train, [(int(u), int(v)) for u, v in test_neg])

    X_train = pd.concat([X_train_pos, X_train_neg], ignore_index=True)
    y_train = np.array([1] * len(X_train_pos) + [0] * len(X_train_neg), dtype=int)

    X_test = pd.concat([X_test_pos, X_test_neg], ignore_index=True)
    y_test = np.array([1] * len(X_test_pos) + [0] * len(X_test_neg), dtype=int)

    feature_cols = [
        "common_neighbors",
        "jaccard",
        "adamic_adar",
        "preferential_attachment",
        "resource_allocation",
    ]

    model = LogisticRegression(max_iter=2000, n_jobs=None, solver="lbfgs")
    model.fit(X_train[feature_cols], y_train)

    y_score = model.predict_proba(X_test[feature_cols])[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_score))
    f1 = float(f1_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))

    res = LinkPredResult(
        dataset=dataset_name,
        n_nodes=G_full.number_of_nodes(),
        n_edges=G_full.number_of_edges(),
        n_train_pos=len(train_pos),
        n_test_pos=len(test_pos),
        n_train_neg=len(train_neg),
        n_test_neg=len(test_neg),
        auc=auc,
        f1=f1,
        precision=prec,
        recall=rec,
        model_name="LogisticRegression(heuristic features)",
    )

    scored = X_test.copy()
    scored["y_true"] = y_test
    scored["y_score"] = y_score
    scored["y_pred"] = y_pred

    return res, model, scored


def recommend_friends(
    G: nx.Graph,
    node_id: int,
    model: LogisticRegression,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Recommend new links for a node using the trained model.
    Scores candidates that are not already neighbors and not itself.
    Uses heuristic features computed on the CURRENT graph G.
    """
    if node_id not in G:
        raise ValueError(f"node {node_id} not in graph")

    neighbors = set(G.neighbors(node_id))
    candidates = [n for n in G.nodes() if n != node_id and n not in neighbors]
    pairs = [(int(node_id), int(v)) for v in candidates]

    feats = _pairs_to_features(G, pairs)
    feature_cols = [
        "common_neighbors",
        "jaccard",
        "adamic_adar",
        "preferential_attachment",
        "resource_allocation",
    ]
    scores = model.predict_proba(feats[feature_cols])[:, 1]
    feats["score"] = scores

    out = feats.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return out[["u", "v", "score", "common_neighbors", "jaccard", "adamic_adar"]]