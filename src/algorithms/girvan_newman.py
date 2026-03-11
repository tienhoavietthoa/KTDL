from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.quality import modularity


def _communities_to_membership(communities: Tuple[set, ...]) -> Dict[int, int]:
    membership: Dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            membership[int(node)] = int(cid)
    return membership


def run_girvan_newman_best_k(
    G: nx.Graph,
    k_min: int = 2,
    k_max: int = 8,
) -> Tuple[Dict[int, int], float, int, List[dict]]:
    """
    Scan k from 2..k_max (GN generator progressively splits),
    choose best modularity among k in [k_min, k_max].
    """
    comp_gen = girvan_newman(G)
    best_Q = float("-inf")
    best_k = None
    best_membership = None
    records: List[dict] = []

    for k in range(2, k_max + 1):
        communities = next(comp_gen)
        communities = tuple(sorted(communities, key=len, reverse=True))

        if k < k_min:
            continue

        Q = modularity(G, communities)
        records.append({"k": int(k), "modularity_Q": float(Q)})

        if float(Q) > best_Q:
            best_Q = float(Q)
            best_k = int(k)
            best_membership = _communities_to_membership(communities)

    if best_membership is None or best_k is None:
        raise RuntimeError("Girvan-Newman failed to produce a partition.")

    return best_membership, float(best_Q), int(best_k), records