from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.algorithms.girvan_newman import run_girvan_newman_best_k
from src.algorithms.label_propagation import run_label_propagation
from src.algorithms.louvain import run_louvain
from src.analyze_community import (
    community_profile_table,
    detect_bridge_nodes,
    analyze_inter_community_edges
)
from src.data_collect import collect_karate_club
from src.insights import node_centrality_table
from src.metrics import (
    compute_clustering_metrics,
    compute_ground_truth_metrics,
    build_node_features,
    graph_stats,
)
from src.preprocess import preprocess_graph
from src.visualize import (
    plot_degree_distribution,
    plot_network_partition,
    plot_gn_modularity_curve,
    plot_metrics_comparison,
    plot_community_sizes,
    plot_adjacency_heatmap,
    plot_node_feature_correlation,
)


@dataclass
class RunResult:
    """
    Kết quả chạy 1 thuật toán
    """
    algorithm: str
    modularity_Q: float
    n_clusters: int
    silhouette: float
    dunn: float
    davies_bouldin: float
    runtime_sec: float
    
    # Ground truth metrics
    nmi: Optional[float] = None
    ari: Optional[float] = None
    purity: Optional[float] = None
    
    # Output paths
    membership_csv: Optional[str] = None
    network_png: Optional[str] = None
    profiles_csv: Optional[str] = None
    centrality_csv: Optional[str] = None
    
    # ✨ NEW: Advanced analysis
    heatmap_png: Optional[str] = None
    correlation_png: Optional[str] = None
    bridge_nodes_csv: Optional[str] = None
    
    # Stability (for LP)
    stability_runs: int = 1
    modularity_mean: Optional[float] = None
    modularity_std: Optional[float] = None


def run_once(
    algorithm: str = "Louvain",
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    out_figures: Path = Path("outputs/figures"),
    out_tables: Path = Path("outputs/tables"),
    seed: int = 42,
    gn_k_min: int = 2,
    gn_k_max: int = 8,
    lp_runs: int = 10,
) -> RunResult:
    """
    Chạy 1 thuật toán phát hiện cộng đồng + tính metrics
    
    Args:
        algorithm: "Louvain" | "Girvan-Newman" | "Label Propagation"
        raw_dir: thư mục data/raw
        processed_dir: thư mục data/processed
        out_figures: thư mục outputs/figures
        out_tables: thư mục outputs/tables
        seed: random seed
        gn_k_min, gn_k_max: Girvan-Newman parameters
        lp_runs: số lần chạy Label Propagation
    
    Returns:
        RunResult object với tất cả metrics
    """
    start_time = time.perf_counter()
    
    # 1. Load & preprocess
    print(f"\n{'='*60}")
    print(f"  {algorithm}")
    print(f"{'='*60}")
    
    print("📥 Loading Karate Club...")
    G, edges_csv, nodes_csv, graphml_path = collect_karate_club(raw_dir, processed_dir)
    
    print("🧹 Preprocessing...")
    G = preprocess_graph(G, processed_dir)
    stats = graph_stats(G)
    print(f"   Graph: {stats.n_nodes} nodes, {stats.n_edges} edges")
    
    # 2. Visualize degree distribution (only once)
    if not (out_figures / "degree_distribution.png").exists():
        print("📊 Plotting degree distribution...")
        plot_degree_distribution(G, out_figures / "degree_distribution.png")
    
    # 3. Run algorithm
    print(f"🔍 Running {algorithm}...")
    
    membership = None
    modularity_Q = None
    gn_records = None
    stability_mean = None
    stability_std = None
    
    if algorithm == "Louvain":
        membership, modularity_Q = run_louvain(G, random_state=seed)
    
    elif algorithm == "Girvan-Newman":
        membership, modularity_Q, best_k, gn_records = run_girvan_newman_best_k(
            G, k_min=gn_k_min, k_max=gn_k_max
        )
    
    elif algorithm == "Label Propagation":
        # Run multiple times for stability
        results = []
        for i in range(lp_runs):
            m, q, _ = run_label_propagation(G, seed=seed + i)
            results.append({'membership': m, 'Q': q})
        
        # Select best
        best_idx = np.argmax([r['Q'] for r in results])
        membership = results[best_idx]['membership']
        modularity_Q = results[best_idx]['Q']
        
        # Compute stability
        Qs = [r['Q'] for r in results]
        stability_mean = float(np.mean(Qs))
        stability_std = float(np.std(Qs))
    
    # 4. Build node features for metrics
    print("📈 Computing metrics...")
    node_features = build_node_features(G, membership)
    
    # 5. Compute clustering metrics
    metrics = compute_clustering_metrics(G, membership, node_features)
    
    # 6. Compute ground truth metrics
    print("🎯 Computing ground truth comparison...")
    try:
        gt_metrics = compute_ground_truth_metrics(membership, nodes_csv)
    except Exception as e:
        print(f"⚠️  Ground truth error: {e}")
        gt_metrics = {'nmi': None, 'ari': None, 'purity': None}
    
    # 7. Save membership CSV
    print("💾 Saving results...")
    algo_name = algorithm.lower().replace(" ", "_").replace("-", "_")
    
    membership_df = node_features.copy()
    membership_df_path = out_tables / f"{algo_name}_membership.csv"
    membership_df.to_csv(membership_df_path, index=False)
    
    # 8. Visualize network
    network_png_path = out_figures / f"{algo_name}_network.png"
    plot_network_partition(G, membership, algorithm, network_png_path)
    
    # 9. Community profiles
    profiles_df = community_profile_table(G, membership, top_k=5)
    profiles_csv_path = out_tables / f"{algo_name}_community_profiles.csv"
    profiles_df.to_csv(profiles_csv_path, index=False)
    
    # 10. Node centrality
    centrality_df = node_centrality_table(G, membership)
    centrality_csv_path = out_tables / f"{algo_name}_node_centrality.csv"
    centrality_df.to_csv(centrality_csv_path, index=False)
    
    # 11. GN specific - plot Q vs K curve
    if algorithm == "Girvan-Newman" and gn_records:
        gn_curve_path = out_figures / "gn_modularity_by_k.png"
        plot_gn_modularity_curve(gn_records, gn_curve_path)
        
        # Save records to CSV
        gn_csv_path = out_tables / "gn_modularity_by_k.csv"
        pd.DataFrame(gn_records).to_csv(gn_csv_path, index=False)
    
    # 12. Community size distribution
    comm_size_path = out_figures / f"{algo_name}_community_sizes.png"
    plot_community_sizes(membership, algorithm, comm_size_path)
    
    # 13. ✨ Adjacency heatmap
    print("🔥 Plotting adjacency heatmap...")
    heatmap_path = out_figures / f"{algo_name}_adjacency_heatmap.png"
    plot_adjacency_heatmap(G, membership, algorithm, heatmap_path)
    
    # 14. ✨ Node feature correlation
    print("📈 Plotting feature correlation...")
    corr_path = out_figures / f"{algo_name}_feature_correlation.png"
    plot_node_feature_correlation(node_features, algorithm, corr_path)
    
    # 15. ✨ Bridge nodes analysis
    print("🌉 Detecting bridge nodes...")
    bridge_df = detect_bridge_nodes(G, membership, top_k=5)
    bridge_path = out_tables / f"{algo_name}_bridge_nodes.csv"
    bridge_df.to_csv(bridge_path, index=False)
    print(f"✓ Saved: {bridge_path}")
    print(f"\nTop 5 Bridge Nodes:")
    print(bridge_df.to_string(index=False))
    
    # 16. ✨ Inter-community edges
    print("\n📊 Analyzing inter-community edges...")
    inter_stats = analyze_inter_community_edges(G, membership)
    print(f"   Intra-edges (within communities): {inter_stats['intra_edges']}")
    print(f"   Inter-edges (between communities): {inter_stats['inter_edges']}")
    print(f"   Intra ratio: {inter_stats['intra_ratio']:.2%}")
    print(f"   Inter ratio: {inter_stats['inter_ratio']:.2%}")
    print(f"   Inter-edges by pair: {inter_stats['inter_by_pair']}")
    
    end_time = time.perf_counter()
    runtime = end_time - start_time
    
    # 17. Create result object
    result = RunResult(
        algorithm=algorithm,
        modularity_Q=float(modularity_Q),
        n_clusters=metrics['n_clusters'],
        silhouette=float(metrics['silhouette']),
        dunn=float(metrics['dunn']),
        davies_bouldin=float(metrics['davies_bouldin']),
        runtime_sec=float(runtime),
        nmi=gt_metrics.get('nmi'),
        ari=gt_metrics.get('ari'),
        purity=gt_metrics.get('purity'),
        membership_csv=str(membership_df_path),
        network_png=str(network_png_path),
        profiles_csv=str(profiles_csv_path),
        centrality_csv=str(centrality_csv_path),
        heatmap_png=str(heatmap_path),
        correlation_png=str(corr_path),
        bridge_nodes_csv=str(bridge_path),
        stability_runs=lp_runs if algorithm == "Label Propagation" else 1,
        modularity_mean=stability_mean,
        modularity_std=stability_std,
    )
    
    print(f"✓ Runtime: {runtime:.6f}s")
    
    return result