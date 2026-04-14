"""
Summary plots for Experiment 1 — all datasets.

Two grid types per phase:
  1. rows = datasets,  cols = LLMs    → averaged over LLMs
  2. rows = LLMs,      cols = datasets → averaged over datasets

Usage:
    python plot_summary_all_datasets.py --phase all
    python plot_summary_all_datasets.py --phase prevalence
    python plot_summary_all_datasets.py --phase low
    python plot_summary_all_datasets.py --phase high
    python plot_summary_all_datasets.py --phase full
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from argparse import ArgumentParser


# ── constants ─────────────────────────────────────────────────────────────────

LLM_ORDER  = ["llama", "deepseek", "gpt54", "mistral", "claude"]
LLM_TITLES = {"llama": "Llama", "deepseek": "DeepSeek",
               "gpt54": "GPT-5.4", "mistral": "Mistral", "claude": "Claude"}

DATASETS = ["cuad", "misogynistic", "vuamc", "fomc"]
DATASET_TITLES = {
    "cuad":         "CUAD",
    "misogynistic": "Misogynistic",
    "vuamc":        "VUAMC",
    "fomc":         "FOMC",
}

# Total dataset size N per (dataset, llm) for x-axis normalisation
N_MAX = 997  # all datasets capped at 997 rows

DATASET_N: dict[tuple[str, str], int] = {
    (ds, llm): N_MAX
    for ds in ["cuad", "misogynistic", "fomc", "pubmedqa", "vuamc"]
    for llm in ["llama", "deepseek", "gpt54", "mistral", "claude"]
}

METHODS = ["expert_only", "dsl", "ppi", "ppipp", "llm_only"]
DEBIASING = ["expert_only", "dsl", "ppi", "ppipp"]
METHOD_LABELS = {
    "expert_only": r"$\theta_\dagger$",
    "dsl":         "DSL",
    "ppi":         "PPI",
    "ppipp":       "PPI++",
    "llm_only":    "LLM only",
}

# (srmse_col, srmse_se_col, ylabel_srmse, bias_col, bias_se_col)
PHASE_METRIC = {
    "prevalence": ("sRMSE",       "sRMSE_se",       "sRMSE",
                   "standardized_bias", "bias_se"),
    "low":        ("sRMSE_beta2", "sRMSE_beta2_se",  r"sRMSE ($\beta$)",
                   "bias_beta2",  "bias_beta2_se"),
    "high":       ("sRMSE_beta2", "sRMSE_beta2_se",  r"sRMSE ($\beta$)",
                   "bias_beta2",  "bias_beta2_se"),
    "full":       ("sRMSE_eucl",  "sRMSE_eucl_se",   "sRMSE (Eucl.)",
                   "bias_eucl",   "bias_eucl_se"),
}

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


# ── summaries directory lookup ─────────────────────────────────────────────────

def _summaries_dir_for(base: Path, dataset: str) -> Path:
    return base / dataset


# ── data loading ───────────────────────────────────────────────────────────────

def load_phase(summaries_base: Path, dataset: str, phase: str) -> pd.DataFrame | None:
    suffix = PHASE_FILE_SUFFIX[phase]
    sdir   = _summaries_dir_for(summaries_base, dataset)
    frames = []
    for llm in LLM_ORDER:
        path = sdir / f"{dataset}_{llm}_{suffix}.csv"
        if path.exists():
            df = pd.read_csv(path)
            # normalise n_expert → proportion using dataset N
            N = DATASET_N.get((dataset, llm), None)
            if N and "n_expert" in df.columns:
                df["n_prop"] = df["n_expert"] / N
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ── shared panel drawing ───────────────────────────────────────────────────────

def _draw_panel(ax, df: pd.DataFrame, metric: str, se_col: str,
                ylabel: str, title: str, n_avg: int,
                include_llm_only: bool = True):
    """
    df has already been filtered to the relevant rows (one LLM or one dataset).
    n_avg: number of items being averaged (for SE calculation).
    x-axis: n_prop (n_expert / N).
    """
    colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: colors[i] for i, m in enumerate(METHODS)}
    active    = DEBIASING + (["llm_only"] if include_llm_only else [])

    ref      = next(m for m in active if m != "llm_only")
    x_values = np.sort(df[df["method"] == ref]["n_prop"].dropna().unique())

    for method in active:
        color = color_map[method]
        label = METHOD_LABELS[method]
        mdf   = df[df["method"] == method].copy()
        if mdf.empty:
            continue

        if method == "llm_only":
            mean_val = mdf[metric].mean()
            se_val   = mdf[metric].std() / np.sqrt(n_avg) if n_avg > 1 else 0.0
            ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.5,
                       label=label, zorder=3)
            ax.axhspan(mean_val - 2 * se_val, mean_val + 2 * se_val,
                       color=color, alpha=0.12, linewidth=0)
        else:
            grp = (mdf.groupby("n_prop")[metric]
                   .agg(["mean", "std"])
                   .reset_index()
                   .sort_values("n_prop"))
            grp["se"] = grp["std"] / np.sqrt(n_avg)
            x  = grp["n_prop"].to_numpy(dtype=float)
            y  = grp["mean"].to_numpy(dtype=float)
            se = grp["se"].fillna(0).to_numpy(dtype=float)
            ax.fill_between(x, y - 2 * se, y + 2 * se,
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(x, y, "o-", color=color, linewidth=1.6,
                    markersize=4, label=label, zorder=4)

    if len(x_values) > 0:
        ax.set_xscale("log")
        step = max(1, len(x_values) // 6)
        ticks = x_values[::step]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:.3f}" for v in ticks],
                           rotation=45, ha="right", fontsize=7)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(x_values[0] * 0.85, x_values[-1] * 1.05)

    ax.set_xlabel("n / N (log)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(axis="y", labelsize=7)


def _add_legend(fig, active, color_map):
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


# ── grid 1: rows=datasets, averaged over LLMs ─────────────────────────────────

def make_grid_by_dataset(summaries_base: Path, phase: str, fig_dir: Path,
                         which: str = "srmse"):
    metric_col, se_col, ylabel_srmse, bias_col, bias_se_col = PHASE_METRIC[phase]
    metric, se_c, ylabel = (
        (metric_col, se_col, ylabel_srmse) if which == "srmse"
        else (bias_col, bias_se_col, "Standardised Bias")
    )

    available = [ds for ds in DATASETS
                 if load_phase(summaries_base, ds, phase) is not None]
    if not available:
        print(f"No data for phase={phase}.")
        return

    ncols = 2
    nrows = (len(available) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows), squeeze=False)
    colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: colors[i] for i, m in enumerate(METHODS)}
    active    = DEBIASING + (["llm_only"] if which == "srmse" else [])
    n_llms    = len(LLM_ORDER)

    for i, ds in enumerate(available):
        ax = axes[i // ncols][i % ncols]
        df = load_phase(summaries_base, ds, phase)
        _draw_panel(ax, df, metric, se_c, ylabel,
                    title=DATASET_TITLES[ds],
                    n_avg=n_llms,
                    include_llm_only=(which == "srmse"))

    for i in range(len(available), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    _add_legend(fig, active, color_map)
    metric_label = ylabel_srmse if which == "srmse" else "Standardised Bias"
    fig.suptitle(f"{PHASE_LABELS[phase]} — {metric_label} (averaged over LLMs)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out = (fig_dir / "Dataset" / f"all_datasets_{PHASE_FILE_SUFFIX[phase]}_{which}_avg_llms.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    print(f"Saved → {out}")
    plt.close(fig)


# ── grid 2: rows=LLMs, averaged over datasets ─────────────────────────────────

def make_grid_by_llm(summaries_base: Path, phase: str, fig_dir: Path,
                     which: str = "srmse"):
    metric_col, se_col, ylabel_srmse, bias_col, bias_se_col = PHASE_METRIC[phase]
    metric, se_c, ylabel = (
        (metric_col, se_col, ylabel_srmse) if which == "srmse"
        else (bias_col, bias_se_col, "Standardised Bias")
    )

    available_ds = [ds for ds in DATASETS
                    if load_phase(summaries_base, ds, phase) is not None]
    if not available_ds:
        print(f"No data for phase={phase}.")
        return
    n_ds = len(available_ds)

    colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: colors[i] for i, m in enumerate(METHODS)}
    active    = DEBIASING + (["llm_only"] if which == "srmse" else [])

    ncols = 2
    nrows = (len(LLM_ORDER) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows), squeeze=False)

    for r, llm in enumerate(LLM_ORDER):
        ax = axes[r // ncols][r % ncols]
        frames = []
        for ds in available_ds:
            df_ds = load_phase(summaries_base, ds, phase)
            if df_ds is not None:
                frames.append(df_ds[df_ds["llm"] == llm].copy())
        if not frames:
            ax.set_visible(False)
            continue
        df = pd.concat(frames, ignore_index=True)
        _draw_panel(ax, df, metric, se_c, ylabel,
                    title=LLM_TITLES[llm],
                    n_avg=n_ds,
                    include_llm_only=(which == "srmse"))

    for i in range(len(LLM_ORDER), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    _add_legend(fig, active, color_map)
    ds_str       = ", ".join(DATASET_TITLES[d] for d in available_ds)
    metric_label = ylabel_srmse if which == "srmse" else "Standardised Bias"
    fig.suptitle(
        f"{PHASE_LABELS[phase]} — {metric_label}\n(averaged over datasets: {ds_str})",
        fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out = (fig_dir / "LLMs" / f"all_llms_{PHASE_FILE_SUFFIX[phase]}_{which}_avg_datasets.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    print(f"Saved → {out}")
    plt.close(fig)


# ── helper: load all phases for a dataset/llm, normalise metric to common col ─

def load_all_phases(summaries_base: Path, dataset: str,
                    which: str = "srmse") -> pd.DataFrame | None:
    """
    Load all 4 phases for a dataset, rename the relevant sRMSE (or bias) column
    to 'metric' so they can be pooled and averaged across phases.
    """
    frames = []
    for phase in ["prevalence", "low", "high", "full"]:
        df = load_phase(summaries_base, dataset, phase)
        if df is None:
            continue
        metric_col, _, _, bias_col, _ = PHASE_METRIC[phase]
        col = metric_col if which == "srmse" else bias_col
        if col not in df.columns:
            continue
        sub = df.copy()
        sub["metric"] = sub[col]
        sub["phase"]  = phase
        frames.append(sub[["dataset", "llm", "method", "n_expert", "n_prop",
                            "metric", "phase"]])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ── grid: averaged over phases (and LLMs) — rows=datasets ─────────────────────

def make_phase_avg_by_dataset(summaries_base: Path, fig_dir: Path,
                               which: str = "srmse"):
    ylabel      = "sRMSE (avg. over phases)" if which == "srmse" else "Std. Bias (avg. over phases)"
    available   = [ds for ds in DATASETS
                   if load_all_phases(summaries_base, ds, which) is not None]
    if not available:
        return

    ncols = 2
    nrows = (len(available) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows), squeeze=False)
    colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: colors[i] for i, m in enumerate(METHODS)}
    active    = DEBIASING + (["llm_only"] if which == "srmse" else [])
    n_avg     = len(LLM_ORDER) * 4  # averaging over LLMs × phases

    for i, ds in enumerate(available):
        ax = axes[i // ncols][i % ncols]
        df = load_all_phases(summaries_base, ds, which)
        _draw_panel(ax, df, "metric", "metric", ylabel,
                    title=DATASET_TITLES[ds], n_avg=n_avg,
                    include_llm_only=(which == "srmse"))

    for i in range(len(available), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    _add_legend(fig, active, color_map)
    fig.suptitle(f"All Phases Combined — {ylabel} (averaged over LLMs)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out = fig_dir / "Dataset" / f"all_datasets_all_phases_{which}_avg_llms.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    print(f"Saved → {out}")
    plt.close(fig)


# ── grid: averaged over phases (and datasets) — rows=LLMs ─────────────────────

def make_phase_avg_by_llm(summaries_base: Path, fig_dir: Path,
                           which: str = "srmse"):
    ylabel       = "sRMSE (avg. over phases)" if which == "srmse" else "Std. Bias (avg. over phases)"
    available_ds = [ds for ds in DATASETS
                    if load_all_phases(summaries_base, ds, which) is not None]
    if not available_ds:
        return
    n_avg = len(available_ds) * 4  # averaging over datasets × phases

    ncols = 2
    nrows = (len(LLM_ORDER) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows), squeeze=False)
    colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: colors[i] for i, m in enumerate(METHODS)}
    active    = DEBIASING + (["llm_only"] if which == "srmse" else [])

    for r, llm in enumerate(LLM_ORDER):
        ax = axes[r // ncols][r % ncols]
        frames = []
        for ds in available_ds:
            df_ds = load_all_phases(summaries_base, ds, which)
            if df_ds is not None:
                frames.append(df_ds[df_ds["llm"] == llm].copy())
        if not frames:
            ax.set_visible(False)
            continue
        df = pd.concat(frames, ignore_index=True)
        _draw_panel(ax, df, "metric", "metric", ylabel,
                    title=LLM_TITLES[llm], n_avg=n_avg,
                    include_llm_only=(which == "srmse"))

    for i in range(len(LLM_ORDER), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    _add_legend(fig, active, color_map)
    ds_str = ", ".join(DATASET_TITLES[d] for d in available_ds)
    fig.suptitle(f"All Phases Combined — {ylabel}\n(averaged over datasets: {ds_str})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out = fig_dir / "LLMs" / f"all_llms_all_phases_{which}_avg_datasets.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    print(f"Saved → {out}")
    plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--summaries-dir", type=Path,
        default=Path("thesis/results/experiment 1/summaries"))
    parser.add_argument("--phase", type=str,
        choices=["prevalence", "low", "high", "full", "all"],
        default="all")
    parser.add_argument("--fig-dir", type=Path,
        default=Path("thesis/results/experiment 1/figures/Aggregated plots"))
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
        for which in ("srmse", "bias"):
            make_grid_by_dataset(args.summaries_dir, ph, args.fig_dir, which=which)
            make_grid_by_llm(args.summaries_dir, ph, args.fig_dir, which=which)

    print("\n── All phases combined ──")
    for which in ("srmse", "bias"):
        make_phase_avg_by_dataset(args.summaries_dir, args.fig_dir, which=which)
        make_phase_avg_by_llm(args.summaries_dir, args.fig_dir, which=which)
