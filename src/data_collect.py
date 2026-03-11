from __future__ import annotations

from pathlib import Path
from typing import Tuple

import networkx as nx
import pandas as pd


def collect_karate_club(raw_dir: Path, processed_dir: Path) -> Tuple[nx.Graph, Path, Path, Path]:
    """
    Load Karate Club graph, export raw CSVs and a GraphML file.

    Returns:
      G, edges_csv, nodes_csv, graphml_path
    """
    G = nx.karate_club_graph()

    nodes_rows = []
    for n, data in G.nodes(data=True):
        nodes_rows.append({"node_id": int(n), "club": data.get("club", None)})

    nodes_df = pd.DataFrame(nodes_rows).sort_values("node_id")
    nodes_csv = raw_dir / "nodes.csv"
    nodes_df.to_csv(nodes_csv, index=False, encoding="utf-8")

    edges_rows = [{"source": int(u), "target": int(v)} for u, v in G.edges()]
    edges_df = pd.DataFrame(edges_rows).sort_values(["source", "target"])
    edges_csv = raw_dir / "edges.csv"
    edges_df.to_csv(edges_csv, index=False, encoding="utf-8")

    graphml_path = processed_dir / "karate.graphml"
    nx.write_graphml(G, graphml_path)

    return G, edges_csv, nodes_csv, graphml_path