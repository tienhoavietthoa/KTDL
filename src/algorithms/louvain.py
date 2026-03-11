from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import community as community_louvain  # python-louvain


def run_louvain(G: nx.Graph, random_state: int = 42) -> Tuple[Dict[int, int], float]:
    partition = community_louvain.best_partition(G, random_state=random_state)
    Q = community_louvain.modularity(partition, G)
    return partition, float(Q)