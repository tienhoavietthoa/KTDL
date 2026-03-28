from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)


@dataclass
class GraphStats:
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    is_connected: bool


def graph_stats(G: nx.Graph) -> GraphStats:
    """
    Tính thống kê cơ bản của đồ thị
    
    Args:
        G: NetworkX Graph
    
    Returns:
        GraphStats object
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    avg_degree = (2 * n_edges / n_nodes) if n_nodes > 0 else 0
    is_connected = nx.is_connected(G)
    
    return GraphStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        density=float(density),
        avg_degree=float(avg_degree),
        is_connected=is_connected
    )


def build_node_features(G: nx.Graph, membership: Dict[int, int]) -> pd.DataFrame:
    """
    Tính 5 centrality measures cho mỗi node
    
    Features:
    - degree: số bạn
    - clustering: mức độ hình thành tam giác
    - pagerank: ảnh hưởng (dựa trên số bạn của bạn)
    - betweenness: số đường ngắn nhất đi qua node
    - closeness: khoảng cách trung bình tới các node khác
    
    Args:
        G: NetworkX Graph
        membership: {node_id: community_id}
    
    Returns:
        DataFrame: [node_id, community_id, degree, clustering, pagerank, betweenness, closeness]
    """
    # Degree
    degree = dict(G.degree())
    
    # Clustering coefficient
    clustering = nx.clustering(G)
    
    # PageRank
    pagerank = nx.pagerank(G)
    
    # Betweenness centrality (normalized)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    
    # Closeness centrality
    closeness = nx.closeness_centrality(G)
    
    rows = []
    for node in G.nodes():
        rows.append({
            'node_id': int(node),
            'community_id': int(membership.get(node, -1)),
            'degree': float(degree.get(node, 0)),
            'clustering_coeff': float(clustering.get(node, 0)),
            'pagerank': float(pagerank.get(node, 0)),
            'betweenness': float(betweenness.get(node, 0)),
            'closeness': float(closeness.get(node, 0)),
        })
    
    return pd.DataFrame(rows)


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Tính Dunn Index = min(inter-cluster distance) / max(intra-cluster diameter)
    
    Cao = tốt (clusters tách biệt rõ)
    Thấp = kém (clusters gần nhau)
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
    
    Returns:
        Dunn Index (float)
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    # Tính đường kính (diameter) của mỗi cluster
    intra_dists = []
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        if np.sum(cluster_mask) < 2:
            continue
        cluster_points = X[cluster_mask]
        # Diameter = max pairwise distance
        from scipy.spatial.distance import pdist, squareform
        dists = pdist(cluster_points, metric='euclidean')
        if len(dists) > 0:
            intra_dists.append(np.max(dists))
    
    if not intra_dists:
        return 0.0
    
    max_intra = np.max(intra_dists)
    
    # Tính khoảng cách tối thiểu giữa các cluster centroids
    from scipy.spatial.distance import cdist
    
    centroids = []
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        centroid = np.mean(X[cluster_mask], axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    if len(centroids) < 2:
        return 0.0
    
    inter_dists = pdist(centroids, metric='euclidean')
    min_inter = np.min(inter_dists) if len(inter_dists) > 0 else 1e-6
    
    if max_intra == 0:
        return 0.0
    
    return float(min_inter / max_intra)


def compute_clustering_metrics(
    G: nx.Graph,
    membership: Dict[int, int],
    features: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Tính 4 độ đo chất lượng phân hoạch
    
    Metrics:
    - Silhouette: -1 to 1 (1 = tốt, -1 = xấu)
    - Dunn: 0 to ∞ (cao = tốt)
    - Davies-Bouldin: 0 to ∞ (thấp = tốt)
    - n_clusters: số cộng đồng
    
    Args:
        G: NetworkX Graph
        membership: {node_id: community_id}
        features: Pre-computed features (if None, will compute)
    
    Returns:
        {silhouette, dunn, davies_bouldin, n_clusters}
    """
    if features is None:
        features = build_node_features(G, membership)
    
    # Prepare data
    X = features[['degree', 'clustering_coeff', 'pagerank', 'betweenness', 'closeness']].values
    nodes_order = features['node_id'].values
    labels = np.array([membership[n] for n in nodes_order])
    
    n_clusters = len(np.unique(labels))
    
    # Silhouette (có thể âm - bình thường cho networks)
    try:
        if n_clusters < 2 or n_clusters >= len(X):
            silhouette = -1.0
        else:
            silhouette = float(silhouette_score(X, labels))
    except Exception as e:
        print(f"⚠️  Silhouette computation error: {e}")
        silhouette = -1.0
    
    # Davies-Bouldin (thấp = tốt)
    try:
        if n_clusters < 2:
            davies_bouldin = float('inf')
        else:
            davies_bouldin = float(davies_bouldin_score(X, labels))
    except Exception as e:
        print(f"⚠️  Davies-Bouldin computation error: {e}")
        davies_bouldin = float('inf')
    
    # Dunn Index (cao = tốt)
    try:
        dunn = dunn_index(X, labels)
    except Exception as e:
        print(f"⚠️  Dunn Index computation error: {e}")
        dunn = 0.0
    
    return {
        'silhouette': silhouette,
        'dunn': dunn,
        'davies_bouldin': davies_bouldin,
        'n_clusters': n_clusters
    }


def compute_ground_truth_metrics(
    membership: Dict[int, int],
    ground_truth_csv: str
) -> Dict[str, float]:
    """
    ✨ THÊMMỚI: So sánh kết quả với ground truth (Mr. Hi vs Officer)
    
    Metrics:
    - NMI (Normalized Mutual Information): 0-1 (1 = perfect match)
    - ARI (Adjusted Rand Index): -1 to 1 (1 = perfect match)
    - Purity: 0-1 (1 = perfect match)
    
    Args:
        membership: {node_id: community_id} từ algorithm
        ground_truth_csv: đường dẫn nodes.csv (có cột 'club')
    
    Returns:
        {nmi, ari, purity}
    """
    import pandas as pd
    
    # Load ground truth
    df = pd.read_csv(ground_truth_csv)
    ground_truth = {}
    for _, row in df.iterrows():
        node = int(row['node_id'])
        club = row['club']  # "Mr. Hi" = 0, "Officer" = 1
        ground_truth[node] = 1 if club == "Officer" else 0
    
    # Convert to arrays (sorted order)
    nodes = sorted(membership.keys())
    pred_labels = np.array([membership[n] for n in nodes])
    true_labels = np.array([ground_truth[n] for n in nodes])
    
    # Map multi-label prediction to binary
    pred_binary = _map_to_binary(pred_labels, true_labels)
    
    # Compute metrics
    nmi = float(normalized_mutual_info_score(true_labels, pred_binary))
    ari = float(adjusted_rand_score(true_labels, pred_binary))
    purity = float(_purity_score(true_labels, pred_binary))
    
    return {
        'nmi': nmi,
        'ari': ari,
        'purity': purity
    }


def _map_to_binary(pred_labels: np.ndarray, true_labels: np.ndarray) -> List[int]:
    """
    Map multi-community prediction to binary (Mr. Hi=0 vs Officer=1)
    
    Tìm community nào match nhất với Mr. Hi, còn lại là Officer
    """
    pred = np.array(pred_labels, dtype=int)
    true = np.array(true_labels, dtype=int)
    
    # Tìm community có overlap lớn nhất với Mr. Hi (true=0)
    mr_hi_comm = None
    max_overlap = 0
    for comm in np.unique(pred):
        overlap = np.sum((pred == comm) & (true == 0))
        if overlap > max_overlap:
            max_overlap = overlap
            mr_hi_comm = comm
    
    if mr_hi_comm is None:
        mr_hi_comm = pred[0]
    
    # Map: mr_hi_comm -> 0, others -> 1
    binary = np.where(pred == mr_hi_comm, 0, 1)
    return binary.tolist()


def _purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Purity: % nodes assigned to correct class
    
    Công thức: Σ max(confusion_matrix) / n_samples
    """
    from sklearn.metrics import confusion_matrix
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Confusion matrix
    w = confusion_matrix(y_true, y_pred)
    
    # Optimal assignment (Hungarian algorithm)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    purity = w[row_ind, col_ind].sum() / len(y_true)
    return float(purity)