"""
Plotting script for Experiment 2 — Phase 1: Class Prevalence
Vary total dataset size N, fixed n_expert.

X-axis: N_total (log scale)
Usage:
    python plot_prevalence.py --dataset misogynistic --n-expert 50
    python plot_prevalence.py --dataset cuad --n-expert 100
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from argparse import ArgumentParser


LLM_ORDER  = ["llama", "deepseek", "gpt54", "mistral", "claude"]
LLM_TITLES = {"llama": "Llama", "deepseek": "DeepSeek",
               "gpt54": "GPT-5.4", "mistral": "Mistral", "claude": "Claude"}

METHODS = ["expert_only", "dsl", "ppi", "ppipp", "llm_only"]
METHOD_LABELS = {
    "expert_only": r"$\theta_\dagger$",
    "dsl":         "DSL",
    "ppi":         "PPI",
    "ppipp":       "PPI++",
    "llm_only":    "LLM only",
}


def load_summaries(summaries_dir: Path, dataset: str, n_expert: int) -> pd.DataFrame:
    frames = []
    for llm in LLM_ORDER:
        path = summaries_dir / f"{dataset}_{llm}_n{n_expert}_prevalence_total.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No prevalence CSVs found in {summaries_dir}")
    return pd.concat(frames, ignore_index=True)


def _plot_panel(ax, sub: pd.DataFrame, colors: list, metric: str, se_col: str,
                ylabel: str, title: str, N_values: np.ndarray,
                methods: list[str] | None = None):
    active = methods if methods is not None else METHODS
    has_se = se_col in sub.columns

    for idx, method in enumerate(active):
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
            mdf = mdf.sort_values("N_total")
            x = mdf["N_total"].values
            y = mdf[metric].values

            if has_se:
                se = mdf[se_col].to_numpy(dtype=float)
                ax.fill_between(x, y - 2 * se, y + 2 * se,
                                color=color, alpha=0.2, linewidth=0)

            ax.plot(x, y, "o-", color=color, linewidth=1.8,
                    markersize=5, label=label, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(N_values)
    ax.set_xticklabels([str(int(n)) for n in N_values], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(left=N_values[0] * 0.85, right=N_values[-1] * 1.05)

    ax.set_xlabel("Total dataset size N (log)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")


def make_figure(df: pd.DataFrame, dataset: str,
                metric: str, se_col: str, ylabel: str,
                suptitle: str, output: Path,
                methods: list[str] | None = None):
    active_methods = methods if methods is not None else METHODS
    llms_present   = [llm for llm in LLM_ORDER if llm in df["llm"].unique()]
    colors         = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map      = {m: colors[i] for i, m in enumerate(METHODS)}

    ref_method = next(m for m in active_methods if m != "llm_only")

    n_llms = len(llms_present)
    ncols = 3 if n_llms > 4 else 2
    nrows = (n_llms + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5 * nrows))
    axes = axes.flatten()

    for ax, llm in zip(axes, llms_present):
        sub = pd.DataFrame(df[df["llm"] == llm])
        N_values = np.sort(
            sub[sub["method"] == ref_method]["N_total"].dropna().unique()
        )
        _plot_panel(ax, sub, [color_map[m] for m in active_methods],
                    metric, se_col,
                    ylabel, LLM_TITLES.get(llm, llm), N_values,
                    methods=active_methods)

    for ax in axes[len(llms_present):]:
        ax.set_visible(False)

    handles = []
    for method in active_methods:
        ls = "--" if method == "llm_only" else "-"
        h = mlines.Line2D([], [], color=color_map[method],
                          marker="o" if method != "llm_only" else "",
                          linestyle=ls, linewidth=1.8, markersize=5,
                          label=METHOD_LABELS[method])
        handles.append(h)

    fig.legend(handles=handles, loc="lower center", ncol=len(active_methods),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), bbox_inches="tight", dpi=300)
    print(f"Saved → {output}")
    plt.close(fig)


def make_averaged_figure(df: pd.DataFrame, dataset: str,
                         metric: str, se_col: str, ylabel: str,
                         suptitle: str, output: Path,
                         methods: list[str] | None = None):
    """Single-panel figure averaged over LLMs."""
    active_methods = methods if methods is not None else ["expert_only", "dsl", "ppi", "ppipp"]
    colors         = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map      = {m: colors[i] for i, m in enumerate(METHODS)}

    fig, ax = plt.subplots(figsize=(7, 5))

    ref_method  = next(m for m in active_methods if m != "llm_only")
    N_values    = np.sort(df[df["method"] == ref_method]["N_total"].dropna().unique())
    n_llms      = df["llm"].nunique()

    for method in active_methods:
        color = color_map[method]
        label = METHOD_LABELS[method]
        mdf   = df[df["method"] == method].copy()

        if method == "llm_only":
            mean_val = mdf[metric].mean()
            se_val   = mdf[metric].std() / np.sqrt(n_llms) if n_llms > 1 else 0.0
            ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.8,
                       label=label, zorder=3)
            ax.axhspan(mean_val - 2 * se_val, mean_val + 2 * se_val,
                       color=color, alpha=0.12, linewidth=0)
        else:
            grp = (mdf.groupby("N_total")[metric]
                   .agg(["mean", "std"])
                   .reset_index()
                   .sort_values("N_total"))
            grp["se"] = grp["std"] / np.sqrt(n_llms)

            x  = grp["N_total"].to_numpy(dtype=float)
            y  = grp["mean"].to_numpy(dtype=float)
            se = grp["se"].fillna(0).to_numpy(dtype=float)

            ax.fill_between(x, y - 2 * se, y + 2 * se,
                            color=color, alpha=0.2, linewidth=0)
            ax.plot(x, y, "o-", color=color, linewidth=1.8,
                    markersize=5, label=label, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(N_values)
    ax.set_xticklabels([str(int(n)) for n in N_values], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(left=N_values[0] * 0.85, right=N_values[-1] * 1.05)
    ax.set_xlabel("Total dataset size N (log)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, frameon=True)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), bbox_inches="tight", dpi=300)
    print(f"Saved → {output}")
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--summaries-dir", type=Path,
        default=Path("thesis/results/experiment_2/summaries"))
    parser.add_argument("--dataset",  type=str, default="misogynistic")
    parser.add_argument("--n-expert", type=int, default=50,
        help="Fixed number of expert annotations (50, 100, or 200)")
    parser.add_argument("--fig-dir",  type=Path,
        default=Path("thesis/results/experiment_2/figures"))
    args = parser.parse_args()

    fig_dir  = args.fig_dir
    ds       = args.dataset
    n        = args.n_expert
    tag      = f"{ds}_n{n}"

    df = load_summaries(args.summaries_dir, ds, n)
    print(f"Loaded {len(df)} rows  |  dataset={ds}  |  n_expert={n}")

    make_figure(
        df, ds,
        metric="sRMSE", se_col="sRMSE_se",
        ylabel="sRMSE",
        suptitle=f"Class Prevalence — sRMSE ({ds.upper()}, n_expert={n})",
        output=fig_dir / f"{tag}_prevalence_srmse.png",
        methods=["expert_only", "dsl", "ppi", "ppipp"],
    )

    make_figure(
        df, ds,
        metric="bias", se_col="bias_se",
        ylabel="Standardised Bias",
        suptitle=f"Class Prevalence — Standardised Bias ({ds.upper()}, n_expert={n})",
        output=fig_dir / f"{tag}_prevalence_bias.png",
        methods=["expert_only", "dsl", "ppi", "ppipp"],
    )

    make_figure(
        df, ds,
        metric="sRMSE", se_col="sRMSE_se",
        ylabel="sRMSE",
        suptitle=f"Class Prevalence — sRMSE with LLM baseline ({ds.upper()}, n_expert={n})",
        output=fig_dir / f"{tag}_prevalence_full.png",
        methods=["expert_only", "dsl", "ppi", "ppipp", "llm_only"],
    )

    make_averaged_figure(
        df, ds,
        metric="sRMSE", se_col="sRMSE_se",
        ylabel="sRMSE",
        suptitle=f"Class Prevalence — sRMSE averaged over LLMs ({ds.upper()}, n_expert={n})",
        output=fig_dir / f"{tag}_prevalence_avg.png",
        methods=["expert_only", "dsl", "ppi", "ppipp"],
    )
