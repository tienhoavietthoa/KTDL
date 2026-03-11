from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.community.quality import modularity


def _communities_to_membership(communities: List[set]) -> Dict[int, int]:
    membership: Dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            membership[int(node)] = int(cid)
    return membership


def run_label_propagation(G: nx.Graph, seed: int = 42) -> Tuple[Dict[int, int], float, int]:
    comms = list(asyn_lpa_communities(G, seed=seed))
    Q = modularity(G, comms)
    membership = _communities_to_membership(comms)
    return membership, float(Q), len(comms)