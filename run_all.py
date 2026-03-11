import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from networkx.algorithms import community as nx_comm
import pandas as pd
import os


def load_example_graph():
    # Dữ liệu thử: Karate Club graph (đã có sẵn trong NetworkX)
    return nx.karate_club_graph()


def preprocess_graph(G: nx.Graph) -> nx.Graph:
    # Loại self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # Loại node cô lập (nếu có)
    isolated = list(nx.isolates(G))
    if isolated:
        G.remove_nodes_from(isolated)

    # Đảm bảo graph là vô hướng
    if G.is_directed():
        G = G.to_undirected()
    return G


def run_louvain(G: nx.Graph):
    partition = community_louvain.best_partition(G)
    mod = community_louvain.modularity(partition, G)
    return partition, mod


def run_girvan_newman(G: nx.Graph, target_communities: int = 2):
    comp_gen = nx_comm.girvan_newman(G)
    communities = None
    for comps in comp_gen:
        communities = list(comps)
        if len(communities) >= target_communities:
            break
    partition = {node: cid for cid, comp in enumerate(communities) for node in comp}
    mod = community_louvain.modularity(partition, G)
    return partition, mod


def run_label_propagation(G: nx.Graph):
    comps = list(nx_comm.label_propagation_communities(G))
    partition = {node: cid for cid, comp in enumerate(comps) for node in comp}
    mod = community_louvain.modularity(partition, G)
    return partition, mod


def visualize_partition(G: nx.Graph, partition: dict, title: str, filename: str):
    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42)
    colors = [partition[n] for n in G.nodes()]
    nx.draw(G, pos, node_color=colors, cmap="tab20", with_labels=True, node_size=250)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def graph_stats(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    avg_degree = sum(d for _, d in G.degree()) / n if n else 0
    comps = list(nx.connected_components(G))
    giant = max(comps, key=len) if comps else set()
    diameter = nx.diameter(G.subgraph(giant)) if comps else 0

    return {
        "num_nodes": n,
        "num_edges": m,
        "density": density,
        "avg_degree": avg_degree,
        "num_components": len(comps),
        "giant_component_size": len(giant),
        "diameter": diameter,
    }


def save_stats(stats: dict, filename: str):
    pd.DataFrame([stats]).to_csv(filename, index=False)


def load_graph_from_edgelist(path: str) -> nx.Graph:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return nx.read_edgelist(path, nodetype=str, data=False)


def main():
    data_file = "network.edgelist"
    if os.path.exists(data_file):
        G = load_graph_from_edgelist(data_file)
        print("Load từ", data_file)
    else:
        G = load_example_graph()
        print("Dùng Karate Club làm demo")

    G = preprocess_graph(G)

    part_louvain, mod_louvain = run_louvain(G)
    part_gn, mod_gn = run_girvan_newman(G, target_communities=2)
    part_lp, mod_lp = run_label_propagation(G)

    print("Louvain: modularity =", mod_louvain, "num communities =", len(set(part_louvain.values())))
    print("Girvan-Newman: modularity =", mod_gn, "num communities =", len(set(part_gn.values())))
    print("Label Propagation: modularity =", mod_lp, "num communities =", len(set(part_lp.values())))

    visualize_partition(G, part_louvain, "Louvain", "louvain.png")
    visualize_partition(G, part_gn, "Girvan-Newman", "girvan_newman.png")
    visualize_partition(G, part_lp, "Label Propagation", "label_propagation.png")

    stats = graph_stats(G)
    print("Graph stats:", stats)
    save_stats(stats, "graph_stats.csv")


if __name__ == "__main__":
    main()