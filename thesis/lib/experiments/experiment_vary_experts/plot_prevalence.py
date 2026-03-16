"""
Plot: Vary Expert Sample Size — Phase 1: Class Prevalence

Creates a 2×2 figure with one subplot per LLM.
Each subplot shows sRMSE (or standardized bias) vs n_expert, with ±2 SE
confidence bands matching the style of plot_test_fitting.py.

Usage:
    python3 plot_prevalence.py \
        --summaries-dir thesis/results/summaries \
        --dataset cuad \
        --output-srmse thesis/results/figures/cuad_prevalence_srmse.pdf \
        --output-bias  thesis/results/figures/cuad_prevalence_bias.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from argparse import ArgumentParser


LLM_ORDER  = ["llama", "deepseek", "gpt54", "mistral"]
LLM_TITLES = {"llama": "Llama", "deepseek": "DeepSeek",
               "gpt54": "GPT-5.4", "mistral": "Mistral"}

# Method display order and labels
METHODS = ["expert_only", "dsl", "ppi", "llm_only"]
METHOD_LABELS = {
    "expert_only": "Expert only",
    "dsl":         "DSL",
    "ppi":         "PPI",
    "llm_only":    "LLM only",
}


def load_summaries(summaries_dir: Path, dataset: str) -> pd.DataFrame:
    frames = []
    for llm in LLM_ORDER:
        path = summaries_dir / f"{dataset}_{llm}_prevalence.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No prevalence CSVs found in {summaries_dir}")
    return pd.concat(frames, ignore_index=True)


def _plot_panel(ax, sub: pd.DataFrame, colors: list, metric: str, se_col: str,
                ylabel: str, title: str, n_values: np.ndarray):
    """
    Draw one LLM subplot for either sRMSE or standardized_bias.
    Falls back gracefully if se_col is absent (old CSVs without SE).
    """
    has_se = se_col in sub.columns

    for idx, method in enumerate(METHODS):
        mdf = sub[sub["method"] == method].copy()
        if mdf.empty:
            continue

        color = colors[idx]
        label = METHOD_LABELS[method]

        if method == "llm_only":
            val = mdf[metric].iloc[0]
            ax.axhline(val, color=color, linestyle="--", linewidth=1.8,
                       label=label, zorder=3)
            if has_se:
                se = mdf[se_col].iloc[0]
                ax.axhspan(val - 2 * se, val + 2 * se,
                           color=color, alpha=0.12, linewidth=0)
        else:
            mdf = pd.DataFrame(mdf).sort_values(by="n_expert")
            x   = mdf["n_expert"].values
            y   = mdf[metric].values

            if has_se:
                se = mdf[se_col].to_numpy(dtype=float)
                ax.fill_between(x, y - 2 * se, y + 2 * se,
                                color=color, alpha=0.2, linewidth=0)

            ax.plot(x, y, "o-", color=color, linewidth=1.8,
                    markersize=5, label=label, zorder=4)

    # x-axis ticks: actual n values, rotated
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(int(n)) for n in n_values], rotation=45, ha="right")

    ax.set_xlabel("Number of expert labels (n)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=n_values[0] * 0.85, right=n_values[-1] * 1.05)


def make_figure(df: pd.DataFrame, _dataset: str,
                metric: str, se_col: str, ylabel: str,
                suptitle: str, output: Path):
    llms_present = [llm for llm in LLM_ORDER if llm in df["llm"].unique()]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # n_values for x-axis ticks (same for all LLMs)
    n_values = np.sort(
        df[df["method"] == "expert_only"]["n_expert"].dropna().unique()
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, llm in zip(axes, llms_present):
        sub = df[df["llm"] == llm]
        _plot_panel(ax, sub, colors, metric, se_col,
                    ylabel, LLM_TITLES.get(llm, llm), n_values)

    # Hide any unused panels
    for ax in axes[len(llms_present):]:
        ax.set_visible(False)

    # Shared legend below figure
    handles = []
    for idx, method in enumerate(METHODS):
        ls = "--" if method == "llm_only" else "-"
        h = mlines.Line2D([], [], color=colors[idx], marker="o" if method != "llm_only" else "",
                          linestyle=ls, linewidth=1.8, markersize=5,
                          label=METHOD_LABELS[method])
        handles.append(h)

    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(suptitle, fontsize=13, y=1.01)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    print(f"Saved → {output}")
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--summaries-dir", type=Path,
        default=Path("thesis/results/summaries"))
    parser.add_argument("--dataset", type=str, default="cuad")
    parser.add_argument("--output-srmse", type=Path,
        default=Path("thesis/results/figures/cuad_prevalence_srmse.pdf"))
    parser.add_argument("--output-bias", type=Path,
        default=Path("thesis/results/figures/cuad_prevalence_bias.pdf"))
    args = parser.parse_args()

    df = load_summaries(args.summaries_dir, args.dataset)
    print(f"Loaded {len(df)} rows  |  dataset={args.dataset}")

    make_figure(
        df, args.dataset,
        metric="sRMSE", se_col="sRMSE_se",
        ylabel="sRMSE",
        suptitle=f"Class Prevalence — sRMSE vs Expert Sample Size  ({args.dataset.upper()})",
        output=args.output_srmse,
    )

    make_figure(
        df, args.dataset,
        metric="standardized_bias", se_col="bias_se",
        ylabel="Standardized Bias",
        suptitle=f"Class Prevalence — Standardized Bias vs Expert Sample Size  ({args.dataset.upper()})",
        output=args.output_bias,
    )