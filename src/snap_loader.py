from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path
from typing import Tuple

import networkx as nx


FACEBOOK_COMBINED_URL = "https://snap.stanford.edu/data/facebook_combined.txt.gz"


def download_if_needed(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    # Download
    urllib.request.urlretrieve(url, out_path)
    return out_path


def load_facebook_combined(raw_dir: Path) -> Tuple[nx.Graph, Path]:
    """
    Loads SNAP Facebook combined graph (undirected) from .txt.gz edge list.
    Returns: (G, path)
    """
    gz_path = raw_dir / "facebook_combined.txt.gz"
    download_if_needed(FACEBOOK_COMBINED_URL, gz_path)

    edges_path = raw_dir / "facebook_combined.txt"
    if not edges_path.exists():
        with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f_in, open(
            edges_path, "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                f_out.write(line + "\n")

    G = nx.read_edgelist(edges_path, nodetype=int, data=False, create_using=nx.Graph())
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G, edges_path