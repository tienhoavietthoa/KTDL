from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import Timer
from src.data_collect import collect_karate_club
from src.preprocess import load_and_preprocess
from src.algorithms.louvain import run_louvain
from src.algorithms.girvan_newman import run_girvan_newman_best_k
from src.algorithms.label_propagation import run_label_propagation
from src.metrics import build_node_features, compute_clustering_metrics
from src.visualize import plot_network_partition, plot_degree_distribution
from src.analyze_community import community_profile_table
from src.insights import node_centrality_table


@dataclass
class RunResult:
    algorithm: str
    modularity_Q: float
    n_clusters: int
    silhouette: float
    dunn: float
    davies_bouldin: float
    runtime_sec: float

    membership_csv: Path
    network_png: Path
    profiles_csv: Path
    centrality_csv: Path

    stability_runs: int = 1
    modularity_mean: float | None = None
    modularity_std: float | None = None


def run_once(
    algorithm: str,
    raw_dir: Path,
    processed_dir: Path,
    out_figures: Path,
    out_tables: Path,
    seed: int = 42,
    gn_k_min: int = 2,
    gn_k_max: int = 8,
    lp_runs: int = 1,
) -> RunResult:
    _, _, _, graphml_path = collect_karate_club(raw_dir, processed_dir)
    G, _, _, _ = load_and_preprocess(graphml_path, processed_dir)

    plot_degree_distribution(G, out_figures / "degree_distribution.png")

    stability_runs = 1
    q_mean = None
    q_std = None

    with Timer(algorithm) as t:
        if algorithm == "Louvain":
            membership, Q = run_louvain(G, random_state=seed)
            algo_name, prefix = "Louvain", "louvain"

        elif algorithm == "Girvan-Newman":
            membership, Q, best_k, records = run_girvan_newman_best_k(G, k_min=gn_k_min, k_max=gn_k_max)
            pd.DataFrame(records).to_csv(out_tables / "gn_modularity_by_k.csv", index=False, encoding="utf-8")
            algo_name, prefix = f"Girvan-Newman(k={best_k})", "girvan_newman"

        elif algorithm == "Label Propagation":
            stability_runs = max(1, int(lp_runs))
            memberships, Qs = [], []
            for i in range(stability_runs):
                m_i, q_i, _ = run_label_propagation(G, seed=seed + i)
                memberships.append(m_i)
                Qs.append(q_i)

            best_idx = int(np.argmax(Qs))
            membership = memberships[best_idx]
            Q = float(Qs[best_idx])

            q_mean = float(np.mean(Qs))
            q_std = float(np.std(Qs, ddof=1)) if stability_runs > 1 else 0.0

            algo_name = f"Label Propagation(best of {stability_runs}, n={len(set(membership.values()))})"
            prefix = "label_propagation"
        else:
            raise ValueError("Unknown algorithm")

    runtime = float(t.seconds)

    membership_csv = out_tables / f"{prefix}_membership.csv"
    pd.DataFrame([{"node_id": k, "community_id": v} for k, v in sorted(membership.items())]).to_csv(
        membership_csv, index=False, encoding="utf-8"
    )

    network_png = out_figures / f"{prefix}_network.png"
    plot_network_partition(G, membership, network_png, f"{algo_name} communities")

    profiles_csv = out_tables / f"{prefix}_community_profiles.csv"
    community_profile_table(G, membership, top_k=5).to_csv(profiles_csv, index=False, encoding="utf-8")

    centrality_csv = out_tables / f"{prefix}_node_centrality.csv"
    node_centrality_table(G, membership).to_csv(centrality_csv, index=False, encoding="utf-8")

    features_df = build_node_features(G)
    cm = compute_clustering_metrics(features_df, membership)

    return RunResult(
        algorithm=algo_name,
        modularity_Q=float(Q),
        n_clusters=int(cm["n_clusters"]),
        silhouette=float(cm["silhouette"]),
        dunn=float(cm["dunn"]),
        davies_bouldin=float(cm["davies_bouldin"]),
        runtime_sec=runtime,
        membership_csv=membership_csv,
        network_png=network_png,
        profiles_csv=profiles_csv,
        centrality_csv=centrality_csv,
        stability_runs=stability_runs,
        modularity_mean=q_mean,
        modularity_std=q_std,
    )