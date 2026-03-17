import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from argparse import ArgumentParser


LLM_ORDER  = ["llama", "deepseek", "gpt54", "mistral"]
LLM_TITLES = {"llama": "Llama", "deepseek": "DeepSeek",
               "gpt54": "GPT-5.4", "mistral": "Mistral"}

DATASET_N: dict[tuple[str, str], int] = {
    ("cuad", "llama"):            1395,
    ("cuad", "deepseek"):         1396,
    ("cuad", "gpt54"):            1396,
    ("cuad", "mistral"):          1391,
    ("misogynistic", "llama"):    997,
    ("misogynistic", "deepseek"): 1000,
    ("misogynistic", "gpt54"):    1000,
    ("misogynistic", "mistral"):  997,
}

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
        path = summaries_dir / f"{dataset}_{llm}_low_variance.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No low_variance CSVs found in {summaries_dir}")
    return pd.concat(frames, ignore_index=True)


def _plot_panel(ax, sub: pd.DataFrame, colors: list, metric: str, se_col: str,
                ylabel: str, title: str, prop_values: np.ndarray,
                methods: list[str] | None = None):
    """Draw one LLM subplot. Falls back gracefully if se_col is absent."""
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
            x   = mdf["prop"].values
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
    ax.set_xlim(left=prop_values[0] * 0.85, right=prop_values[-1] * 1.05)

    ax.set_xlabel("Proportion of expert samples (log)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")


def make_averaged_figure(df: pd.DataFrame, _dataset: str,
                         metric: str, ylabel: str,
                         se_col: str, suptitle: str, output: Path,
                         methods: list[str] | None = None):
    """Single-panel figure averaged over LLMs — analogous to Figure 3 in the paper."""
    active_methods = methods if methods is not None else ["expert_only", "dsl", "ppi"]
    colors         = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map      = {m: colors[i] for i, m in enumerate(METHODS)}

    mean_N = int(np.mean(list(DATASET_N.values())))
    df = df.copy()
    df["prop"] = df["n_expert"].apply(
        lambda n: n / mean_N if pd.notna(n) else np.nan
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ref_method  = next(m for m in active_methods if m != "llm_only")
    prop_values = np.sort(
        df[df["method"] == ref_method]["prop"].dropna().unique()
    )

    n_llms = df["llm"].nunique()

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
    active_methods = methods if methods is not None else METHODS
    llms_present   = [llm for llm in LLM_ORDER if llm in df["llm"].unique()]
    colors         = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map      = {m: colors[i] for i, m in enumerate(METHODS)}

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
    args = parser.parse_args()

    fig_dir = Path("thesis/results/figures")
    ds      = args.dataset

    df = load_summaries(args.summaries_dir, args.dataset)
    print(f"Loaded {len(df)} rows  |  dataset={args.dataset}")

    # Primary metric: sRMSE of β₂ (the feature coefficient)
    make_figure(
        df, ds,
        metric="sRMSE_beta2", se_col="sRMSE_beta2_se",
        ylabel=r"sRMSE ($\beta_2$)",
        suptitle=f"Low-Variance Feature — sRMSE of β₂ ({ds.upper()})",
        output=fig_dir / f"{ds}_low_variance_srmse_beta2.pdf",
        methods=["expert_only", "dsl", "ppi"],
    )

    make_figure(
        df, ds,
        metric="bias_beta2", se_col="bias_beta2_se",
        ylabel=r"Standardised Bias ($\beta_2$)",
        suptitle=f"Low-Variance Feature — Standardised Bias of β₂ ({ds.upper()})",
        output=fig_dir / f"{ds}_low_variance_bias_beta2.pdf",
        methods=["expert_only", "dsl", "ppi"],
    )

    # Include LLM-only horizontal reference
    make_figure(
        df, ds,
        metric="sRMSE_beta2", se_col="sRMSE_beta2_se",
        ylabel=r"sRMSE ($\beta_2$)",
        suptitle=f"Low-Variance Feature — sRMSE β₂ with LLM baseline ({ds.upper()})",
        output=fig_dir / f"{ds}_low_variance_full.pdf",
        methods=["expert_only", "dsl", "ppi", "llm_only"],
    )

    # Euclidean sRMSE (overall θ)
    make_figure(
        df, ds,
        metric="sRMSE_eucl", se_col="sRMSE_eucl_se",
        ylabel="sRMSE (Euclidean)",
        suptitle=f"Low-Variance Feature — Euclidean sRMSE ({ds.upper()})",
        output=fig_dir / f"{ds}_low_variance_srmse_eucl.pdf",
        methods=["expert_only", "dsl", "ppi"],
    )

    # Averaged over LLMs — single panel
    make_averaged_figure(
        df, ds,
        metric="sRMSE_beta2", se_col="sRMSE_beta2_se",
        ylabel=r"sRMSE ($\beta_2$)",
        suptitle=f"Low-Variance Feature — sRMSE β₂ averaged over LLMs ({ds.upper()})",
        output=fig_dir / f"{ds}_low_variance_avg.pdf",
        methods=["expert_only", "dsl", "ppi"],
    )