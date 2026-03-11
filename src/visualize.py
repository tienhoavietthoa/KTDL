from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # prevent GUI hang; we save PNG only

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns


def plot_degree_distribution(G: nx.Graph, out_path: Path) -> None:
    degrees = [d for _, d in G.degree()]
    plt.figure(figsize=(6, 4))
    sns.histplot(degrees, bins=10, kde=False)
    plt.title("Degree distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_network_partition(G: nx.Graph, membership: Dict[int, int], out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=42)

    labels = np.array([membership[int(n)] for n in G.nodes()])
    unique = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=max(10, len(unique)))

    node_colors = [palette[membership[int(n)] % len(palette)] for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=1.0)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=220,
        linewidths=0.5,
        edgecolors="white",
    )
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()