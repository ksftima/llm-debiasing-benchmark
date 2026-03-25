"""
Plot bias-variance tradeoff as a function of L2 regularization strength λ.

Loads summary CSVs produced by evaluate_full_logistic.py for each λ value,
and plots sRMSE, bias, and variance vs λ at 3 representative n values.

Usage:
    python plot_lambda_sensitivity.py --dataset misogynistic
    python plot_lambda_sensitivity.py --dataset cuad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser


LAMBDAS   = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
LLM_ORDER = ["llama", "deepseek", "gpt54", "mistral", "claude"]
METHODS   = ["expert_only", "dsl", "ppi"]
COLORS    = {"expert_only": "#1f77b4", "dsl": "#ff7f0e", "ppi": "#2ca02c"}
LABELS    = {"expert_only": "θ†", "dsl": "DSL", "ppi": "PPI"}

# 3 representative n values (log-spaced: small, medium, large)
N_REPRESENTATIVE = [26, 72, 200]


def load_for_lambda(summaries_dir: Path, dataset: str, lam: float) -> pd.DataFrame:
    lam_tag = "" if lam == 0.01 else f"_lam{str(lam).replace('.', '')}"
    frames = []
    for llm in LLM_ORDER:
        path = summaries_dir / f"{dataset}_{llm}_full_logistic{lam_tag}.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["lam"] = lam
    return df


def compute_variance(srmse, bias):
    """variance component = sRMSE² - bias²  (clamped to 0)"""
    return np.maximum(srmse**2 - bias**2, 0.0)


def make_plots(all_df: pd.DataFrame, dataset: str, fig_dir: Path):

    # Average over LLMs
    grp = all_df.groupby(["lam", "method", "n_expert"])[["sRMSE_eucl", "bias_eucl"]].mean().reset_index()
    grp["variance"] = compute_variance(grp["sRMSE_eucl"], grp["bias_eucl"])

    for n in N_REPRESENTATIVE:
        # Find closest available n
        available_n = grp["n_expert"].dropna().unique()
        closest_n = int(available_n[np.argmin(np.abs(available_n - n))])
        sub_n = grp[grp["n_expert"] == closest_n]

        # --- Plot 1: sRMSE vs λ ---
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in METHODS:
            sub = sub_n[sub_n["method"] == method].sort_values("lam")
            ax.plot(sub["lam"], sub["sRMSE_eucl"], marker="o",
                    color=COLORS[method], label=LABELS[method])
        ax.set_xscale("log")
        ax.set_xlabel("λ (log scale)")
        ax.set_ylabel("sRMSE (Euclidean)")
        ax.set_title(f"Full Logistic — sRMSE vs λ at n={closest_n} ({dataset.upper()})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = fig_dir / f"{dataset}_lam_srmse_n{closest_n}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close(fig)

        # --- Plot 2: bias vs λ ---
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in METHODS:
            sub = sub_n[sub_n["method"] == method].sort_values("lam")
            ax.plot(sub["lam"], sub["bias_eucl"], marker="o",
                    color=COLORS[method], label=LABELS[method])
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("λ (log scale)")
        ax.set_ylabel("Standardised Bias (Euclidean)")
        ax.set_title(f"Full Logistic — Bias vs λ at n={closest_n} ({dataset.upper()})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = fig_dir / f"{dataset}_lam_bias_n{closest_n}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close(fig)

        # --- Plot 3: bias-variance decomposition per method ---
        fig, axes = plt.subplots(1, len(METHODS), figsize=(14, 4), sharey=False)
        for ax, method in zip(axes, METHODS):
            sub = sub_n[sub_n["method"] == method].sort_values("lam")
            ax.plot(sub["lam"], sub["sRMSE_eucl"]**2, marker="o",
                    color="black",     label="sRMSE²")
            ax.plot(sub["lam"], sub["bias_eucl"]**2,  marker="s",
                    color="red",       label="Bias²",    linestyle="--")
            ax.plot(sub["lam"], sub["variance"],       marker="^",
                    color="steelblue", label="Variance", linestyle=":")
            ax.set_xscale("log")
            ax.set_xlabel("λ")
            ax.set_title(LABELS[method])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Full Logistic — Bias-Variance Decomposition at n={closest_n} ({dataset.upper()})")
        fig.tight_layout()
        out = fig_dir / f"{dataset}_lam_decomposition_n{closest_n}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--summaries-dir", type=Path,
        default=Path("thesis/results/summaries"))
    parser.add_argument("--dataset", type=str, default="misogynistic")
    args = parser.parse_args()

    fig_dir = Path("thesis/results/figures/lambda_sensitivity")

    frames = []
    for lam in LAMBDAS:
        df = load_for_lambda(args.summaries_dir, args.dataset, lam)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No summary CSVs found for any λ value.")

    all_df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(all_df)} rows across {all_df['lam'].nunique()} λ values")

    make_plots(all_df, args.dataset, fig_dir)
