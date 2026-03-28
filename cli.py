from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from src.utils import ensure_dirs, append_text
from src.pipeline import run_once
from src.visualize import plot_metrics_comparison


def parse_args():
    p = argparse.ArgumentParser(description="Community Detection CLI (Karate Club)")
    p.add_argument("--algo", choices=["louvain", "gn", "lp", "all"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gn-kmin", type=int, default=2)
    p.add_argument("--gn-kmax", type=int, default=8)
    p.add_argument("--lp-runs", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    paths = ensure_dirs()
    notes_path = paths["out_logs"] / "experiment_notes.md"

    append_text(notes_path, f"\n# CLI run ({date.today().isoformat()})\n")
    append_text(notes_path, f"- algo={args.algo}, seed={args.seed}, gn_kmin={args.gn_kmin}, gn_kmax={args.gn_kmax}, lp_runs={args.lp_runs}\n")

    def run_and_print(label: str):
        res = run_once(
            algorithm=label,
            raw_dir=paths["data_raw"],
            processed_dir=paths["data_processed"],
            out_figures=paths["out_figures"],
            out_tables=paths["out_tables"],
            seed=args.seed,
            gn_k_min=args.gn_kmin,
            gn_k_max=args.gn_kmax,
            lp_runs=args.lp_runs if label == "Label Propagation" else 1,
        )
        print(f"\n{'='*60}")
        print(f"  {res.algorithm}")
        print(f"{'='*60}")
        print(f"Runtime (sec)    : {res.runtime_sec:.6f}")
        print(f"Modularity Q     : {res.modularity_Q:.6f}")
        
        if res.nmi is not None:
            print(f"NMI (ground truth): {res.nmi:.6f}")
        if res.ari is not None:
            print(f"ARI (ground truth): {res.ari:.6f}")
        if res.purity is not None:
            print(f"Purity (ground tr): {res.purity:.6f}")
        
        if res.modularity_mean is not None:
            print(f"LP Q mean±std    : {res.modularity_mean:.6f} ± {res.modularity_std:.6f} (runs={res.stability_runs})")
        print(f"#Clusters        : {res.n_clusters}")
        print(f"Silhouette       : {res.silhouette:.6f}")
        print(f"Dunn             : {res.dunn:.6f}")
        print(f"Davies-Bouldin   : {res.davies_bouldin:.6f}")
        print("Saved:")
        print(f" - {res.network_png}")
        print(f" - {res.membership_csv}")
        print(f" - {res.profiles_csv}")
        print(f" - {res.centrality_csv}")
        print(f" - {res.heatmap_png}")
        print(f" - {res.correlation_png}")
        print(f" - {res.bridge_nodes_csv}")
        return res

    results = []
    if args.algo == "louvain":
        results.append(run_and_print("Louvain"))
    elif args.algo == "gn":
        results.append(run_and_print("Girvan-Newman"))
    elif args.algo == "lp":
        results.append(run_and_print("Label Propagation"))
    else:
        results.append(run_and_print("Louvain"))
        results.append(run_and_print("Girvan-Newman"))
        results.append(run_and_print("Label Propagation"))

    # Plot metrics comparison
    if len(results) > 1:
        print("\n📊 Plotting metrics comparison...")
        plot_metrics_comparison(
            results,
            paths["out_figures"] / "metrics_comparison.png"
        )

    # Save summary
    df = pd.DataFrame([r.__dict__ for r in results])
    summary_path = paths["out_tables"] / "metrics_summary_cli.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n✓ Summary saved: {summary_path}\n")


if __name__ == "__main__":
    main()