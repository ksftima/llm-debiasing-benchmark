import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from argparse import ArgumentParser


LLM_ORDER  = ["llama", "deepseek", "gpt54", "mistral"]
LLM_TITLES = {"llama": "Llama", "deepseek": "DeepSeek",
               "gpt54": "GPT-5.4", "mistral": "Mistral"}

# Total dataset size N per (dataset, llm) — used to convert n → proportion
DATASET_N: dict[tuple[str, str], int] = {
    ("cuad", "llama"):           1395,
    ("cuad", "deepseek"):        1396,
    ("cuad", "gpt54"):           1396,
    ("cuad", "mistral"):         1391,
    ("misogynistic", "llama"):   997,
    ("misogynistic", "deepseek"): 1000,
    ("misogynistic", "gpt54"):   1000,
    ("misogynistic", "mistral"): 997,
}

# Method display order and labels
METHODS = ["expert_only", "dsl", "ppi", "llm_only"]
METHOD_LABELS = {
    "expert_only": r"$\theta_\dagger$",
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
                ylabel: str, title: str, prop_values: np.ndarray,
                methods: list[str] | None = None):
    """
    Draw one LLM subplot.  prop_values are the x-axis positions as proportions n/N.
    Falls back gracefully if se_col is absent (old CSVs without SE).
    colors must be aligned with the methods list passed in.
    """
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
            mdf = pd.DataFrame(mdf).sort_values(by="n_expert")
            x   = mdf["prop"].values      # proportion n/N
            y   = mdf[metric].values

            if has_se:
                se = mdf[se_col].to_numpy(dtype=float)
                ax.fill_between(x, y - 2 * se, y + 2 * se,
                                color=color, alpha=0.2, linewidth=0)

            ax.plot(x, y, "o-", color=color, linewidth=1.8,
                    markersize=5, label=label, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(prop_values)
    ax.set_xticklabels([f"{p:.3f}" for p in prop_values], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(left=prop_values[0] * 0.85, right=prop_values[-1] * 1.05)

    ax.set_xlabel("Proportion of expert samples (log)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")


def make_averaged_figure(df: pd.DataFrame, _dataset: str,
                         metric: str, _se_col: str, ylabel: str,
                         suptitle: str, output: Path,
                         methods: list[str] | None = None):
    """
    Single-panel figure averaged over all LLMs — analogous to Figure 3 in the paper.
    CI bands show ±2 SE across LLMs (inter-LLM variability).
    """
    active_methods = methods if methods is not None else ["expert_only", "dsl", "ppi"]
    colors         = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map      = {m: colors[i] for i, m in enumerate(METHODS)}

    # Add proportion using mean N across LLMs (they're nearly identical ~1394)
    mean_N = int(np.mean(list(DATASET_N.values())))
    df = df.copy()
    df["prop"] = df["n_expert"].apply(
        lambda n: n / mean_N if pd.notna(n) else np.nan
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ref_method = next(m for m in active_methods if m != "llm_only")
    prop_values = np.sort(
        df[df["method"] == ref_method]["prop"].dropna().unique()
    )

    n_llms = df["llm"].nunique()

    for method in active_methods:
        color = color_map[method]
        label = METHOD_LABELS[method]
        mdf   = df[df["method"] == method].copy()

        if method == "llm_only":
            # Average the constant LLM-only value over LLMs
            mean_val = mdf[metric].mean()
            se_val   = mdf[metric].std() / np.sqrt(n_llms) if n_llms > 1 else 0.0
            ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.8,
                       label=label, zorder=3)
            ax.axhspan(mean_val - 2 * se_val, mean_val + 2 * se_val,
                       color=color, alpha=0.12, linewidth=0)
        else:
            # Average sRMSE over LLMs at each n_expert
            grp = (mdf.groupby("n_expert")[metric]
                   .agg(["mean", "std"])
                   .reset_index()
                   .sort_values("n_expert"))
            grp["prop"] = grp["n_expert"] / mean_N
            grp["se"]   = grp["std"] / np.sqrt(n_llms)

            x  = grp["prop"].to_numpy(dtype=float)
            y  = grp["mean"].to_numpy(dtype=float)
            se = grp["se"].fillna(0).to_numpy(dtype=float)

            ax.fill_between(x, y - 2 * se, y + 2 * se,
                            color=color, alpha=0.2, linewidth=0)
            ax.plot(x, y, "o-", color=color, linewidth=1.8,
                    markersize=5, label=label, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(prop_values)
    ax.set_xticklabels([f"{p:.3f}" for p in prop_values], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(left=prop_values[0] * 0.85, right=prop_values[-1] * 1.05)
    ax.set_xlabel("Proportion of expert samples (log)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, frameon=True)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), bbox_inches="tight", dpi=300)
    print(f"Saved → {output}")
    plt.close(fig)


def make_figure(df: pd.DataFrame, dataset: str,
                metric: str, se_col: str, ylabel: str,
                suptitle: str, output: Path,
                methods: list[str] | None = None):
    """
    methods: subset of METHODS to plot. Defaults to all four.
             Pass ["llm_only", "dsl", "ppi"] to zoom in on debiasing methods.
    """
    active_methods = methods if methods is not None else METHODS
    llms_present   = [llm for llm in LLM_ORDER if llm in df["llm"].unique()]
    colors         = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map      = {m: colors[i] for i, m in enumerate(METHODS)}

    # Add proportion column n/N per LLM row
    df = df.copy()
    df["prop"] = df.apply(
        lambda r: r["n_expert"] / DATASET_N.get((dataset, r["llm"]), r["n_expert"])
        if pd.notna(r["n_expert"]) else np.nan,
        axis=1,
    )

    ref_method = next(m for m in active_methods if m != "llm_only")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for ax, llm in zip(axes, llms_present):
        sub = pd.DataFrame(df[df["llm"] == llm])
        prop_values = np.sort(
            sub[sub["method"] == ref_method]["prop"].dropna().unique()
        )
        _plot_panel(ax, sub, [color_map[m] for m in active_methods],
                    metric, se_col,
                    ylabel, LLM_TITLES.get(llm, llm), prop_values,
                    methods=active_methods)

    for ax in axes[len(llms_present):]:
        ax.set_visible(False)

    # Shared legend
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--summaries-dir", type=Path,
        default=Path("thesis/results/summaries"))
    parser.add_argument("--dataset", type=str, default="cuad")
    parser.add_argument("--output-srmse",     type=Path, default=None)
    parser.add_argument("--output-bias",      type=Path, default=None)
    parser.add_argument("--output-debiasing", type=Path, default=None)
    parser.add_argument("--output-avg",       type=Path, default=None)
    args = parser.parse_args()

    fig_dir = Path("thesis/results/figures")
    ds = args.dataset
    if args.output_srmse     is None: args.output_srmse     = fig_dir / f"{ds}_prevalence_srmse.png"
    if args.output_bias      is None: args.output_bias      = fig_dir / f"{ds}_prevalence_bias.png"
    if args.output_debiasing is None: args.output_debiasing = fig_dir / f"{ds}_prevalence_debiasing.png"
    if args.output_avg       is None: args.output_avg       = fig_dir / f"{ds}_prevalence_avg.png"

    df = load_summaries(args.summaries_dir, args.dataset)
    df = df[df["n_expert"].isna() | (df["n_expert"] > 50)]
    print(f"Loaded {len(df)} rows  |  dataset={args.dataset}")

    # Main figure: θ†, DSL, PPI — matching paper Figure 3 style
    make_figure(
        df, args.dataset,
        metric="sRMSE", se_col="sRMSE_se",
        ylabel="sRMSE",
        suptitle=f"Class Prevalence — sRMSE ({args.dataset.upper()})",
        output=args.output_srmse,
        methods=["expert_only", "dsl", "ppi"],
    )

    make_figure(
        df, args.dataset,
        metric="standardized_bias", se_col="bias_se",
        ylabel="Standardised Bias",
        suptitle=f"Class Prevalence — Standardised Bias ({args.dataset.upper()})",
        output=args.output_bias,
        methods=["expert_only", "dsl", "ppi"],
    )

    # Extra: include LLM-only horizontal reference
    make_figure(
        df, args.dataset,
        metric="sRMSE", se_col="sRMSE_se",
        ylabel="sRMSE",
        suptitle=f"Class Prevalence — sRMSE with LLM baseline ({args.dataset.upper()})",
        output=args.output_debiasing,
        methods=["expert_only", "dsl", "ppi", "llm_only"],
    )

    # Averaged over LLMs — single panel matching paper Figure 3
    make_averaged_figure(
        df, args.dataset,
        metric="sRMSE", _se_col="sRMSE_se",
        ylabel="sRMSE",
        suptitle=f"Class Prevalence — sRMSE averaged over LLMs ({args.dataset.upper()})",
        output=args.output_avg,
        methods=["expert_only", "dsl", "ppi"],
    )