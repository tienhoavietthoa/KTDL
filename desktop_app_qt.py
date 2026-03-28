from __future__ import annotations

import gzip
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import networkx as nx

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# -------------------- Title / Topic --------------------
TOPIC_TITLE = "PHÁT HIỆN CỘNG ĐỒNG VÀ ỨNG DỤNG GỢI Ý KẾT NỐI TRONG MẠNG XÃ HỘI"
TOPIC_SUBTITLE = "Ứng dụng: PHÁT HIỆN CỘNG ĐỒNG VÀ ỨNG DỤNG GỢI Ý KẾT NỐI TRONG MẠNG XÃ HỘI"
DATASET_LABEL = "Dataset: SNAP Facebook (facebook_combined)"


# -------------------- SNAP Loader --------------------
FACEBOOK_COMBINED_URL = "https://snap.stanford.edu/data/facebook_combined.txt.gz"


def ensure_dirs() -> dict:
    root = Path(".").resolve()
    data_raw = root / "data" / "raw"
    data_raw.mkdir(parents=True, exist_ok=True)
    return {"data_raw": data_raw}


def download_if_needed(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    urllib.request.urlretrieve(url, out_path)
    return out_path


def load_facebook_combined(raw_dir: Path) -> nx.Graph:
    gz_path = raw_dir / "facebook_combined.txt.gz"
    download_if_needed(FACEBOOK_COMBINED_URL, gz_path)

    edges_path = raw_dir / "facebook_combined.txt"
    if not edges_path.exists() or edges_path.stat().st_size == 0:
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
    return G


# -------------------- Community Algorithms --------------------
def community_louvain_like(G: nx.Graph, seed: int) -> Dict[int, int]:
    from networkx.algorithms.community import louvain_communities

    comms = louvain_communities(G, seed=seed)
    membership: Dict[int, int] = {}
    for cid, nodes in enumerate(comms):
        for n in nodes:
            membership[int(n)] = int(cid)
    return membership


def community_label_propagation(G: nx.Graph, seed: int) -> Dict[int, int]:
    from networkx.algorithms.community import asyn_lpa_communities

    comms = list(asyn_lpa_communities(G, seed=seed))
    membership: Dict[int, int] = {}
    for cid, nodes in enumerate(comms):
        for n in nodes:
            membership[int(n)] = int(cid)
    return membership


def community_girvan_newman(G: nx.Graph, k: int) -> Dict[int, int]:
    from networkx.algorithms.community import girvan_newman

    if k < 2:
        raise ValueError("k phải >= 2")
    gen = girvan_newman(G)
    level = None
    for _ in range(k - 1):
        level = next(gen)

    comms = list(level)
    membership: Dict[int, int] = {}
    for cid, nodes in enumerate(comms):
        for n in nodes:
            membership[int(n)] = int(cid)
    return membership


def compute_modularity(G: nx.Graph, membership: Dict[int, int]) -> float:
    from networkx.algorithms.community.quality import modularity

    comms = {}
    for n, cid in membership.items():
        comms.setdefault(cid, set()).add(n)
    return float(modularity(G, list(comms.values())))


def compute_centrality(G: nx.Graph) -> pd.DataFrame:
    pr = nx.pagerank(G)
    deg = dict(G.degree())
    return pd.DataFrame(
        {"node_id": list(G.nodes()), "degree": [deg[n] for n in G.nodes()], "pagerank": [pr[n] for n in G.nodes()]}
    )


# -------------------- Recommendation (Community + Jaccard) --------------------
def _jaccard(u_neighbors: set, v_neighbors: set) -> float:
    inter = len(u_neighbors & v_neighbors)
    union = len(u_neighbors | v_neighbors)
    return 0.0 if union == 0 else inter / union


def recommend_friends(
    G: nx.Graph,
    membership: Dict[int, int],
    centrality_df: pd.DataFrame,
    u: int,
    top_k: int,
    w_comm: float,
    w_jacc: float,
    w_pr: float,
    candidate_pool: str = "same-community",
) -> pd.DataFrame:
    """
    score(u,v) = w_comm*same_community + w_jacc*Jaccard + w_pr*norm(PageRank(v))
    """
    if u not in G:
        raise ValueError(f"Node {u} không tồn tại trong graph.")

    pr_map = dict(zip(centrality_df["node_id"].astype(int), centrality_df["pagerank"].astype(float)))
    pr_vals = np.array(list(pr_map.values()), dtype=float)
    pr_min, pr_max = float(pr_vals.min()), float(pr_vals.max())

    def pr_norm(x: float) -> float:
        if pr_max - pr_min < 1e-12:
            return 0.0
        return (x - pr_min) / (pr_max - pr_min)

    u_comm = membership.get(u, -1)
    u_neighbors = set(G.neighbors(u))

    if candidate_pool == "two-hop":
        candidates = set()
        for n in u_neighbors:
            candidates.update(G.neighbors(n))
        candidates.discard(u)
        candidates.difference_update(u_neighbors)
    else:
        candidates = [v for v, cid in membership.items() if cid == u_comm and v != u and not G.has_edge(u, v)]

    rows = []
    for v in candidates:
        v = int(v)
        v_neighbors = set(G.neighbors(v))
        jac = _jaccard(u_neighbors, v_neighbors)
        cn = len(u_neighbors & v_neighbors)
        same = 1 if membership.get(v, -999999) == u_comm else 0
        vpr = float(pr_map.get(v, 0.0))
        score = w_comm * same + w_jacc * jac + w_pr * pr_norm(vpr)

        rows.append(
            {
                "rank": 0,
                "v": v,
                "score": score,
                "same_comm": same,
                "jaccard": jac,
                "common_neighbors": cn,
                "pagerank": vpr,
                "comm_v": membership.get(v, -1),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["rank", "v", "score", "same_comm", "jaccard", "common_neighbors", "pagerank", "comm_v"]
        )

    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# -------------------- Worker --------------------
@dataclass
class CommunityRun:
    algorithm: str
    n_nodes: int
    n_edges: int
    n_communities: int
    modularity_Q: float
    membership: Dict[int, int]
    centrality: pd.DataFrame
    runtime_sec: float


class WorkerRun(QThread):
    finished_ok = pyqtSignal(object)  # CommunityRun
    failed = pyqtSignal(str)

    def __init__(self, algo: str, seed: int, gn_k: int, raw_dir: Path):
        super().__init__()
        self.algo = algo
        self.seed = seed
        self.gn_k = gn_k
        self.raw_dir = raw_dir

    def run(self):
        try:
            t0 = time.time()
            G = load_facebook_combined(self.raw_dir)

            if self.algo == "Louvain":
                membership = community_louvain_like(G, seed=self.seed)
            elif self.algo == "Label Propagation":
                membership = community_label_propagation(G, seed=self.seed)
            elif self.algo == "Girvan-Newman":
                membership = community_girvan_newman(G, k=self.gn_k)
            else:
                raise ValueError("Thuật toán không hỗ trợ.")

            Q = compute_modularity(G, membership)
            cent = compute_centrality(G)
            ncom = len(set(membership.values()))
            dt = time.time() - t0

            self.finished_ok.emit(
                CommunityRun(
                    algorithm=self.algo,
                    n_nodes=G.number_of_nodes(),
                    n_edges=G.number_of_edges(),
                    n_communities=ncom,
                    modularity_Q=Q,
                    membership=membership,
                    centrality=cent,
                    runtime_sec=dt,
                )
            )
        except Exception as e:
            self.failed.emit(str(e))


# -------------------- GUI --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{TOPIC_TITLE} — Ứng dụng Desktop")
        self.resize(1420, 900)

        self.paths = ensure_dirs()
        self.G: nx.Graph | None = None
        self.current: CommunityRun | None = None
        self.worker: WorkerRun | None = None

        root = QWidget()
        self.setCentralWidget(root)

        # Pink base / light cute theme + readable table text
        self.setStyleSheet(
            """
            QMainWindow { background: #fff1f6; } /* pink-50 */
            QLabel { color: #3f1d2a; }

            QGroupBox {
                background: #ffffff;
                border: 1px solid #ffd0e1;
                border-radius: 16px;
                margin-top: 10px;
                font-weight: 800;
                color: #3f1d2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 10px 0 10px;
            }

            QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                background: #ffffff;
                border: 1px solid #ffc2d8;
                border-radius: 14px;
                padding: 8px 10px;
                color: #3f1d2a;
            }

            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #fb7185, stop:1 #a78bfa);
                color: #ffffff;
                border: none;
                border-radius: 16px;
                padding: 11px 14px;
                font-weight: 900;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #f43f5e, stop:1 #8b5cf6);
            }
            QPushButton:disabled { background: #fbcfe8; color: #7c2d12; }

            QTableWidget {
                background: #ffffff;
                border: 1px solid #ffd0e1;
                border-radius: 16px;
                gridline-color: #ffe4ef;
                selection-background-color: #ffe4ef;
                selection-color: #3f1d2a;
                alternate-background-color: #fff7fb;
                color: #3f1d2a;
            }
            QHeaderView::section {
                background: #ffe4ef;
                padding: 10px;
                border: none;
                border-right: 1px solid #ffd0e1;
                border-bottom: 1px solid #ffd0e1;
                font-weight: 900;
                color: #3f1d2a;
            }
            """
        )

        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QGroupBox()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(4)

        title = QLabel(TOPIC_TITLE)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.DemiBold))

        subtitle = QLabel(TOPIC_SUBTITLE)
        subtitle.setStyleSheet("color: #7a324c; font-weight: 700;")

        dataset = QLabel(DATASET_LABEL)
        dataset.setStyleSheet("color: #9f4c6a; font-weight: 700;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(dataset)
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        # LEFT
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        box1 = QGroupBox("Bước 1: Phát hiện cộng đồng (Community Detection)")
        f1 = QFormLayout(box1)
        f1.setHorizontalSpacing(12)
        f1.setVerticalSpacing(10)

        self.algo_cb = QComboBox()
        self.algo_cb.addItems(["Louvain", "Label Propagation", "Girvan-Newman"])

        self.seed = QSpinBox()
        self.seed.setRange(0, 10_000_000)
        self.seed.setValue(42)

        self.gn_k = QSpinBox()
        self.gn_k.setRange(2, 80)
        self.gn_k.setValue(10)

        self.btn_run = QPushButton("Chạy Bước 1")
        self.btn_run.clicked.connect(self.on_run)

        f1.addRow("Thuật toán", self.algo_cb)
        f1.addRow("Seed", self.seed)
        f1.addRow("GN số cộng đồng k", self.gn_k)
        f1.addRow("", self.btn_run)

        left_layout.addWidget(box1)

        box2 = QGroupBox("Bước 2: Gợi ý kết bạn")
        f2 = QFormLayout(box2)
        f2.setHorizontalSpacing(12)
        f2.setVerticalSpacing(10)

        self.node_u = QSpinBox()
        self.node_u.setRange(0, 10_000_000)
        self.node_u.setValue(3)

        self.topk = QSpinBox()
        self.topk.setRange(1, 100)
        self.topk.setValue(10)

        self.w_comm = QDoubleSpinBox()
        self.w_comm.setRange(0.0, 1.0)
        self.w_comm.setSingleStep(0.05)
        self.w_comm.setValue(0.7)

        self.w_jacc = QDoubleSpinBox()
        self.w_jacc.setRange(0.0, 1.0)
        self.w_jacc.setSingleStep(0.05)
        self.w_jacc.setValue(0.3)

        self.w_pr = QDoubleSpinBox()
        self.w_pr.setRange(0.0, 1.0)
        self.w_pr.setSingleStep(0.05)
        self.w_pr.setValue(0.0)

        self.pool_cb = QComboBox()
        self.pool_cb.addItems(["same-community", "two-hop"])

        self.btn_rec = QPushButton("Chạy Bước 2 (Gợi ý + Giải thích)")
        self.btn_rec.clicked.connect(self.on_recommend)
        self.btn_rec.setEnabled(False)

        f2.addRow("Node u", self.node_u)
        f2.addRow("Top‑K", self.topk)
        f2.addRow("Ưu tiên cùng nhóm", self.w_comm)
        f2.addRow("Ưu tiên bạn chung", self.w_jacc)
        f2.addRow("Ưu tiên PageRank (tuỳ chọn)", self.w_pr)
        f2.addRow("Ứng viên", self.pool_cb)
        f2.addRow("", self.btn_rec)

        left_layout.addWidget(box2)

        self.status = QLabel("Sẵn sàng.")
        self.status.setStyleSheet("color:#9f4c6a; font-weight: 800;")
        self.status.setWordWrap(True)
        left_layout.addWidget(self.status)

        left_layout.addStretch(1)
        splitter.addWidget(left)

        # RIGHT
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFont("Segoe UI", 10))
        self.summary.setMinimumHeight(250)
        right_layout.addWidget(self.summary)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["#", "Gợi ý (v)", "Điểm", "Cùng nhóm", "Jaccard", "Bạn chung", "PR(v)", "Nhóm(v)"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        right_layout.addWidget(self.table, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 920])

    def _set_table(self, df: pd.DataFrame):
        self.table.setRowCount(0)
        if df is None or df.empty:
            return

        self.table.setRowCount(len(df))

        # ensure readable text colors
        text_brush = QBrush(QColor("#3f1d2a"))  # dark rose text

        def put(r: int, c: int, val: str, align: Qt.AlignmentFlag | None = None):
            item = QTableWidgetItem(val)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setForeground(text_brush)
            if align is not None:
                item.setTextAlignment(int(align))
            self.table.setItem(r, c, item)

        for i, row in df.iterrows():
            put(i, 0, str(int(row["rank"])), Qt.AlignmentFlag.AlignCenter)
            put(i, 1, str(int(row["v"])), Qt.AlignmentFlag.AlignCenter)
            put(i, 2, f"{float(row['score']):.4f}", Qt.AlignmentFlag.AlignCenter)
            put(i, 3, "Có" if int(row["same_comm"]) == 1 else "Không", Qt.AlignmentFlag.AlignCenter)
            put(i, 4, f"{float(row['jaccard']):.4f}", Qt.AlignmentFlag.AlignCenter)
            put(i, 5, str(int(row["common_neighbors"])), Qt.AlignmentFlag.AlignCenter)
            put(i, 6, f"{float(row['pagerank']):.6f}", Qt.AlignmentFlag.AlignCenter)
            put(i, 7, str(int(row["comm_v"])), Qt.AlignmentFlag.AlignCenter)

        self.table.resizeColumnsToContents()

    def on_run(self):
        algo = self.algo_cb.currentText().strip()
        seed = int(self.seed.value())
        gn_k = int(self.gn_k.value())

        self.btn_run.setEnabled(False)
        self.btn_rec.setEnabled(False)
        self.status.setText("Đang chạy bước 1... (lần đầu có thể tải dataset + tính PageRank)")
        self.summary.setPlainText("Đang chạy...")
        self._set_table(pd.DataFrame())

        self.worker = WorkerRun(algo=algo, seed=seed, gn_k=gn_k, raw_dir=self.paths["data_raw"])
        self.worker.finished_ok.connect(self.on_run_ok)
        self.worker.failed.connect(self.on_run_fail)
        self.worker.start()

    def on_run_ok(self, res: CommunityRun):
        if self.G is None:
            self.G = load_facebook_combined(self.paths["data_raw"])
        self.current = res

        sizes = pd.Series(list(res.membership.values())).value_counts().sort_values(ascending=False)
        largest_cid = int(sizes.index[0])
        largest_size = int(sizes.iloc[0])

        top_pr = res.centrality.sort_values("pagerank", ascending=False).head(1).iloc[0]

        # Short + easy explanation
        txt = []
        txt.append("BƯỚC 1: KẾT QUẢ PHÁT HIỆN CỘNG ĐỒNG")
        txt.append(f"• Thuật toán: {res.algorithm}")
        txt.append(f"• Nodes/Edges: {res.n_nodes} / {res.n_edges}")
        txt.append(f"• Số cộng đồng: {res.n_communities}")
        txt.append(f"• Modularity Q: {res.modularity_Q:.4f}")
        txt.append(f"• Runtime: {res.runtime_sec:.2f}s")
        txt.append(f"• Cộng đồng lớn nhất: #{largest_cid} ({largest_size} nodes)")
        txt.append(f"• Node nổi bật: {int(top_pr.node_id)} (PageRank={float(top_pr.pagerank):.6f})")
        txt.append("")
        txt.append("Chạy BƯỚC 2 để tìm gợi ý kết bạn.")

        self.summary.setPlainText("\n".join(txt))
        self.status.setText("Xong bước 1. Bấm 'Chạy Bước 2' để gợi ý.")
        self.btn_run.setEnabled(True)
        self.btn_rec.setEnabled(True)

    def on_run_fail(self, msg: str):
        QMessageBox.critical(self, "Lỗi", msg)
        self.status.setText("Thất bại.")
        self.btn_run.setEnabled(True)

    def on_recommend(self):
        if self.G is None or self.current is None:
            QMessageBox.warning(self, "Chưa có community", "Bạn cần chạy Bước 1 trước.")
            return

        u = int(self.node_u.value())
        top_k = int(self.topk.value())
        w_comm = float(self.w_comm.value())
        w_jacc = float(self.w_jacc.value())
        w_pr = float(self.w_pr.value())
        pool = self.pool_cb.currentText().strip()

        try:
            rec = recommend_friends(
                G=self.G,
                membership=self.current.membership,
                centrality_df=self.current.centrality,
                u=u,
                top_k=top_k,
                w_comm=w_comm,
                w_jacc=w_jacc,
                w_pr=w_pr,
                candidate_pool=pool,
            )
        except Exception as e:
            QMessageBox.critical(self, "Lỗi gợi ý", str(e))
            return

        self._set_table(rec)

        # super simple explanation
        base = self.summary.toPlainText().split("\n\nBƯỚC 2", 1)[0]
        exp = []
        exp.append("BƯỚC 2: GỢI Ý KẾT BẠN ")
        exp.append(f"• Node u = {u} | Top‑K = {top_k} | Ứng viên = {pool}")
        exp.append("• Ưu tiên cùng cộng đồng + nhiều bạn chung.")
        exp.append("")

        if rec.empty:
            exp.append("Không có gợi ý phù hợp.")
            exp.append("Gợi ý: đổi node u hoặc chọn 'two-hop' để lấy 'bạn của bạn'.")
            self.summary.setPlainText(base + "\n\n" + "\n".join(exp))
            self.status.setText("Không có ứng viên. Thử two-hop hoặc đổi u.")
            return

        same_cnt = int(rec["same_comm"].sum())
        top1 = rec.iloc[0]
        exp.append(f"• {same_cnt}/{len(rec)} gợi ý cùng cộng đồng với u.")
        exp.append(
            f"• Top #1: v={int(top1.v)} vì cùng cộng đồng và có {int(top1.common_neighbors)} bạn chung "
            f"(Jaccard={float(top1.jaccard):.3f})."
        )

        self.summary.setPlainText(base + "\n\n" + "\n".join(exp))
        self.status.setText(f"Đã gợi ý Top-{len(rec)} cho node {u}.")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()