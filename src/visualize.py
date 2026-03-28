from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter


def plot_degree_distribution(G: nx.Graph, out_path: Path) -> None:
    """
    Vẽ histogram phân bố bậc (degree distribution)
    """
    degrees = [G.degree(n) for n in G.nodes()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    
    plt.xlabel('Degree (Number of Friends)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    plt.title('Degree Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {out_path}")


def plot_network_partition(
    G: nx.Graph,
    membership: Dict[int, int],
    algorithm: str,
    out_path: Path
) -> None:
    """
    Vẽ mạng với cộng đồng (node color = community)
    """
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    # Node colors by community
    colors = [membership[n] for n in G.nodes()]
    
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(
        G, pos,
        node_color=colors,
        node_size=300,
        cmap='tab20',
        alpha=0.8
    )
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f'{algorithm} Community Detection', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {out_path}")


def plot_gn_modularity_curve(gn_records: List[dict], out_path: Path) -> None:
    """
    ✨ THÊMMỚI: Vẽ curve Modularity vs K (Girvan-Newman)
    
    Giúp xác định best number of communities
    """
    if not gn_records:
        return
    
    k_values = [r['k'] for r in gn_records]
    q_values = [r['modularity_Q'] for r in gn_records]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, q_values, 'b-o', linewidth=2.5, markersize=8, label='Modularity Q')
    
    # Mark best k
    best_idx = np.argmax(q_values)
    best_k = k_values[best_idx]
    best_q = q_values[best_idx]
    plt.plot(best_k, best_q, 'r*', markersize=25, 
             label=f'Best: k={best_k}, Q={best_q:.4f}', zorder=5)
    
    plt.xlabel('Number of Communities (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Modularity Q', fontsize=12, fontweight='bold')
    plt.title('Girvan-Newman: Modularity vs k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {out_path}")


def plot_community_sizes(
    membership: Dict[int, int],
    algorithm: str,
    out_path: Path
) -> None:
    """
    ✨ THÊMMỚI: Vẽ histogram kích thước cộng đồng
    """
    community_sizes = list(Counter(membership.values()).values())
    community_sizes.sort(reverse=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(len(community_sizes)),
        community_sizes,
        color='steelblue',
        edgecolor='black',
        alpha=0.7
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Community (sorted by size)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    plt.title(f'{algorithm}: Community Size Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {out_path}")
    print(f"  Community sizes: {community_sizes}")


def plot_metrics_comparison(results_list: list, out_path: Path) -> None:
    """
    ✨ THÊMMỚI: So sánh metrics giữa 3 thuật toán (bar chart)
    
    Args:
        results_list: [RunResult, RunResult, RunResult]
        out_path: PNG file
    """
    algorithms = [r.algorithm for r in results_list]
    
    # Extract metrics
    qs = np.array([r.modularity_Q for r in results_list])
    sils = np.array([r.silhouette for r in results_list])
    runtimes = np.array([r.runtime_sec for r in results_list])
    
    # Normalize (0-1 scale)
    q_norm = (qs - qs.min()) / (qs.max() - qs.min() + 1e-6)
    sil_norm = (sils - sils.min()) / (sils.max() - sils.min() + 1e-6)
    
    # Speed: invert (faster = higher score)
    time_norm = 1 - (runtimes - runtimes.min()) / (runtimes.max() - runtimes.min() + 1e-6)
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, q_norm, width, label='Modularity Q', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, sil_norm, width, label='Silhouette', color='lightcoral', edgecolor='black')
    bars3 = ax.bar(x + width, time_norm, width, label='Speed', color='lightgreen', edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Metrics Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.15])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {out_path}")


def plot_adjacency_heatmap(
    G: nx.Graph,
    membership: Dict[int, int],
    algorithm: str,
    out_path: Path
) -> None:
    """
    ✨ HEATMAP: Vẽ ma trận kề (nodes sắp theo community)
    
    Giúp visual check: cạnh trong vs ngoài cộng đồng
    - Các khối đường chéo = edges trong community (tốt)
    - Phần còn lại = edges giữa communities (xấu)
    """
    import seaborn as sns
    
    # Sort nodes by community
    sorted_nodes = sorted(membership.keys(), key=lambda x: (membership[x], x))
    node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
    
    # Build adjacency matrix
    n = len(sorted_nodes)
    adj = np.zeros((n, n))
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            adj[i, j] = 1
            adj[j, i] = 1
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(adj, cmap='Blues', cbar_kws={'label': 'Edge exists'}, square=True)
    
    # Add community separators (red lines)
    current_comm = None
    prev_idx = 0
    for i, node in enumerate(sorted_nodes):
        comm = membership[node]
        if comm != current_comm:
            if current_comm is not None:
                plt.axvline(x=i, color='red', linewidth=2)
                plt.axhline(y=i, color='red', linewidth=2)
            current_comm = comm
    
    plt.title(f'{algorithm}: Adjacency Matrix (Nodes sorted by Community)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Node ID', fontsize=11, fontweight='bold')
    plt.ylabel('Node ID', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {out_path}")


def plot_node_feature_correlation(
    node_features: pd.DataFrame,
    algorithm: str,
    out_path: Path
) -> None:
    """
    ✨ CORRELATION: Vẽ heatmap tương quan giữa các centrality measures
    
    Features:
    - degree, pagerank, betweenness, closeness, clustering_coeff
    
    Giúp hiểu: các metrics có liên quan không?
    """
    import seaborn as sns
    
    # Select numeric columns
    numeric_cols = ['degree', 'pagerank', 'betweenness', 'closeness', 'clustering_coeff']
    corr_matrix = node_features[numeric_cols].corr()
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,  # Show correlation values
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        cbar_kws={'label': 'Correlation Coefficient'},
        vmin=-1, vmax=1
    )
    
    plt.title(f'{algorithm}: Node Centrality Correlation Matrix', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {out_path}")
    print(f"\nCorrelation Matrix:\n{corr_matrix}\n")