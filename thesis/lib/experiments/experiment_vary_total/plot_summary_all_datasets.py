"""
Summary plots for Experiment 2 — all datasets, all n_expert values.

Produces one figure per phase showing a grid:
  rows    = datasets
  columns = n_expert ∈ {50, 100, 200}
Each panel: LLM-averaged sRMSE (or bias) vs N_total.

Usage:
    python plot_summary_all_datasets.py --phase prevalence
    python plot_summary_all_datasets.py --phase low
    python plot_summary_all_datasets.py --phase high
    python plot_summary_all_datasets.py --phase full
    python plot_summary_all_datasets.py --phase all   # runs all four
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from argparse import ArgumentParser

# ── constants ─────────────────────────────────────────────────────────────────

LLM_ORDER = ["llama", "deepseek", "gpt54", "mistral", "claude"]

DATASETS = ["cuad", "misogynistic", "vuamc", "fomc"]
DATASET_TITLES = {
    "cuad":          "CUAD",
    "misogynistic":  "Misogynistic",
    "vuamc":         "VUAMC",
    "fomc":          "FOMC",
}
N_EXPERTS = [50, 100, 200]

METHODS = ["expert_only", "dsl", "ppi", "ppipp", "llm_only"]
DEBIASING = ["expert_only", "dsl", "ppi", "ppipp"]
METHOD_LABELS = {
    "expert_only": r"$\theta_\dagger$",
    "dsl":         "DSL",
    "ppi":         "PPI",
    "ppipp":       "PPI++",
    "llm_only":    "LLM only",
}

PHASE_METRIC = {
    "prevalence": ("sRMSE",      "sRMSE_se",      "sRMSE",       "bias",      "bias_se"),
    "low":        ("sRMSE_beta2","sRMSE_beta2_se", r"sRMSE ($\beta$)", "bias_beta2","bias_beta2_se"),
    "high":       ("sRMSE_beta2","sRMSE_beta2_se", r"sRMSE ($\beta$)", "bias_beta2","bias_beta2_se"),
    "full":       ("sRMSE_eucl", "sRMSE_eucl_se",  "sRMSE (Eucl.)",   "bias_eucl", "bias_eucl_se"),
}
# (metric, metric_se, ylabel_srmse, bias_metric, bias_se)

PHASE_LABELS = {
    "prevalence": "Phase 1: Class Prevalence",
    "low":        "Phase 2: Low-Variance Feature",
    "high":       "Phase 3: High-Variance Feature",
    "full":       "Phase 4: Full Logistic Regression",
}
PHASE_FILE_SUFFIX = {
    "prevalence": "prevalence",
    "low":        "low_variance",
    "high":       "high_variance",
    "full":       "full_logistic",
}


# ── data loading ───────────────────────────────────────────────────────────────

def load_phase(summaries_dir: Path, dataset: str, n_expert: int, phase: str) -> pd.DataFrame | None:
    suffix = PHASE_FILE_SUFFIX[phase]
    frames = []
    for llm in LLM_ORDER:
        path = summaries_dir / f"{dataset}_{llm}_n{n_expert}_{suffix}_total.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ── single averaged panel ──────────────────────────────────────────────────────

def _draw_averaged_panel(ax, df: pd.DataFrame, metric: str, se_col: str,
                         ylabel: str, title: str,
                         include_llm_only: bool = True):
    colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: colors[i] for i, m in enumerate(METHODS)}
    active    = DEBIASING + (["llm_only"] if include_llm_only else [])
    n_llms    = df["llm"].nunique()

    ref = next(m for m in active if m != "llm_only")
    N_values = np.sort(df[df["method"] == ref]["N_total"].dropna().unique())

    for method in active:
        color = color_map[method]
        label = METHOD_LABELS[method]
        mdf   = df[df["method"] == method].copy()
        if mdf.empty:
            continue

        if method == "llm_only":
            mean_val = mdf[metric].mean()
            se_val   = mdf[metric].std() / np.sqrt(n_llms) if n_llms > 1 else 0.0
            ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.5,
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
            ax.fill_between(x, y - 2 * se, y + 2 * se, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, y, "o-", color=color, linewidth=1.6, markersize=4,
                    label=label, zorder=4)

    if len(N_values) > 0:
        ax.set_xscale("log")
        ax.set_xticks(N_values)
        ax.set_xticklabels([str(int(v)) for v in N_values], rotation=45, ha="right", fontsize=7)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(N_values[0] * 0.85, N_values[-1] * 1.05)

    ax.set_xlabel("Total N (log)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(axis="y", labelsize=7)


# ── main grid figure ───────────────────────────────────────────────────────────

def make_grid_figure(summaries_dir: Path, phase: str, fig_dir: Path,
                     which: str = "srmse"):
    """
    which: 'srmse' or 'bias'
    Grid: rows = datasets, cols = n_expert values.
    """
    metric_col, se_col, ylabel_srmse, bias_col, bias_se_col = PHASE_METRIC[phase]
    if which == "srmse":
        metric, se, ylabel = metric_col, se_col, ylabel_srmse
    else:
        metric, se, ylabel = bias_col, bias_se_col, "Standardised Bias"

    # determine which datasets are actually available
    available = []
    for ds in DATASETS:
        if load_phase(summaries_dir, ds, 50, phase) is not None:
            available.append(ds)
    if not available:
        print(f"No data found for phase={phase}. Skipping.")
        return

    nrows = len(available)
    ncols = len(N_EXPERTS)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for r, ds in enumerate(available):
        for c, n in enumerate(N_EXPERTS):
            ax = axes[r][c]
            df = load_phase(summaries_dir, ds, n, phase)
            if df is None:
                ax.set_visible(False)
                continue
            # column header (top row only)
            col_title = f"n_expert = {n}"
            # row label (left col only): dataset name in y-label area
            row_prefix = f"{DATASET_TITLES[ds]}\n" if c == 0 else ""
            _draw_averaged_panel(ax, df, metric, se, ylabel,
                                 title=f"{row_prefix}{col_title}",
                                 include_llm_only=(which == "srmse"))

    # shared legend at bottom
    colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: colors[i] for i, m in enumerate(METHODS)}
    legend_methods = DEBIASING + (["llm_only"] if which == "srmse" else [])
    handles = []
    for method in legend_methods:
        ls = "--" if method == "llm_only" else "-"
        h = mlines.Line2D([], [], color=color_map[method],
                          marker="o" if method != "llm_only" else "",
                          linestyle=ls, linewidth=1.6, markersize=4,
                          label=METHOD_LABELS[method])
        handles.append(h)

    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

    phase_label = PHASE_LABELS[phase]
    metric_label = ylabel_srmse if which == "srmse" else "Standardised Bias"
    fig.suptitle(f"{phase_label} — {metric_label} (averaged over LLMs)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    suffix_tag = PHASE_FILE_SUFFIX[phase]
    out = fig_dir / f"all_datasets_{suffix_tag}_{which}_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    print(f"Saved → {out}")
    plt.close(fig)


# ── dataset-averaged grid (rows=LLMs, cols=n_expert) ─────────────────────────

def make_grid_figure_by_llm(summaries_dir: Path, phase: str, fig_dir: Path,
                             which: str = "srmse"):
    """
    Average over datasets instead of LLMs.
    Grid: rows = LLMs, cols = n_expert values.
    Each panel shows method curves averaged over all available datasets.
    """
    metric_col, se_col, ylabel_srmse, bias_col, bias_se_col = PHASE_METRIC[phase]
    if which == "srmse":
        metric, se_c, ylabel = metric_col, se_col, ylabel_srmse
    else:
        metric, se_c, ylabel = bias_col, bias_se_col, "Standardised Bias"

    # collect available datasets
    available_ds = [ds for ds in DATASETS
                    if load_phase(summaries_dir, ds, 50, phase) is not None]
    if not available_ds:
        print(f"No data found for phase={phase}. Skipping.")
        return
    n_ds = len(available_ds)

    colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: colors[i] for i, m in enumerate(METHODS)}
    active    = DEBIASING + (["llm_only"] if which == "srmse" else [])

    nrows = len(LLM_ORDER)
    ncols = len(N_EXPERTS)
    LLM_TITLES = {"llama": "Llama", "deepseek": "DeepSeek",
                  "gpt54": "GPT-5.4", "mistral": "Mistral", "claude": "Claude"}

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for r, llm in enumerate(LLM_ORDER):
        for c, n in enumerate(N_EXPERTS):
            ax = axes[r][c]

            # concatenate all available datasets for this llm/n
            frames = []
            for ds in available_ds:
                df_ds = load_phase(summaries_dir, ds, n, phase)
                if df_ds is not None:
                    sub = df_ds[df_ds["llm"] == llm].copy()
                    frames.append(sub)
            if not frames:
                ax.set_visible(False)
                continue
            df = pd.concat(frames, ignore_index=True)

            ref = next(m for m in active if m != "llm_only")
            N_values = np.sort(df[df["method"] == ref]["N_total"].dropna().unique())

            for method in active:
                color = color_map[method]
                label = METHOD_LABELS[method]
                mdf   = df[df["method"] == method].copy()
                if mdf.empty:
                    continue

                if method == "llm_only":
                    mean_val = mdf[metric].mean()
                    se_val   = mdf[metric].std() / np.sqrt(n_ds) if n_ds > 1 else 0.0
                    ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.5,
                               label=label, zorder=3)
                    ax.axhspan(mean_val - 2 * se_val, mean_val + 2 * se_val,
                               color=color, alpha=0.12, linewidth=0)
                else:
                    grp = (mdf.groupby("N_total")[metric]
                           .agg(["mean", "std"])
                           .reset_index()
                           .sort_values("N_total"))
                    grp["se"] = grp["std"] / np.sqrt(n_ds)
                    x  = grp["N_total"].to_numpy(dtype=float)
                    y  = grp["mean"].to_numpy(dtype=float)
                    se = grp["se"].fillna(0).to_numpy(dtype=float)
                    ax.fill_between(x, y - 2 * se, y + 2 * se,
                                    color=color, alpha=0.18, linewidth=0)
                    ax.plot(x, y, "o-", color=color, linewidth=1.6,
                            markersize=4, label=label, zorder=4)

            if len(N_values) > 0:
                ax.set_xscale("log")
                ax.set_xticks(N_values)
                ax.set_xticklabels([str(int(v)) for v in N_values],
                                   rotation=45, ha="right", fontsize=7)
                ax.xaxis.set_minor_locator(plt.NullLocator())
                ax.set_xlim(N_values[0] * 0.85, N_values[-1] * 1.05)

            row_prefix = f"{LLM_TITLES[llm]}\n" if c == 0 else ""
            ax.set_xlabel("Total N (log)", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{row_prefix}n_expert = {n}", fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.3, which="both")
            ax.tick_params(axis="y", labelsize=7)

    # shared legend
    handles = []
    for method in active:
        ls = "--" if method == "llm_only" else "-"
        h = mlines.Line2D([], [], color=color_map[method],
                          marker="o" if method != "llm_only" else "",
                          linestyle=ls, linewidth=1.6, markersize=4,
                          label=METHOD_LABELS[method])
        handles.append(h)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

    ds_str = ", ".join(DATASET_TITLES[d] for d in available_ds)
    phase_label  = PHASE_LABELS[phase]
    metric_label = ylabel_srmse if which == "srmse" else "Standardised Bias"
    fig.suptitle(f"{phase_label} — {metric_label} (averaged over datasets: {ds_str})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    suffix_tag = PHASE_FILE_SUFFIX[phase]
    out = fig_dir / f"all_llms_{suffix_tag}_{which}_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    print(f"Saved → {out}")
    plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--summaries-dir", type=Path,
        default=Path("thesis/results/experiment2/summaries/N=997"))
    parser.add_argument("--phase", type=str,
        choices=["prevalence", "low", "high", "full", "all"],
        default="all",
        help="Which phase to plot, or 'all' for all four.")
    parser.add_argument("--fig-dir", type=Path,
        default=Path("thesis/results/experiment2/figures/summary"),
        help="Output directory for figures.")
    parser.add_argument("--no-ppi", action="store_true",
        help="Exclude PPI from plots; outputs go to a 'minus PPI' subfolder.")
    args = parser.parse_args()

    if args.no_ppi:
        DEBIASING[:] = [m for m in DEBIASING if m != "ppi"]
        args.fig_dir = args.fig_dir / "minus PPI"

    phases = (["prevalence", "low", "high", "full"]
              if args.phase == "all" else [args.phase])

    for ph in phases:
        print(f"\n── Phase: {ph} ──")
        make_grid_figure(args.summaries_dir, ph, args.fig_dir, which="srmse")
        make_grid_figure(args.summaries_dir, ph, args.fig_dir, which="bias")
        make_grid_figure_by_llm(args.summaries_dir, ph, args.fig_dir, which="srmse")
        make_grid_figure_by_llm(args.summaries_dir, ph, args.fig_dir, which="bias")
