"""
Correlation analysis between LLM annotation quality (Cohen's kappa)
and per-method sRMSE improvement over expert-only baseline.

For each (dataset, llm, phase, method), compute:
    improvement = mean sRMSE(expert_only) - mean sRMSE(method)
                  averaged over all n values

Then compute Pearson r between kappa and improvement across all
(dataset, llm) pairs, separately per method and phase.
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 1. Kappa values  (dataset, llm) -> kappa
# ------------------------------------------------------------------
KAPPA = {
    ("cuad",         "llama"):    0.8509,
    ("cuad",         "deepseek"): 0.8797,
    ("cuad",         "gpt54"):    0.8897,
    ("cuad",         "mistral"):  0.8418,
    ("cuad",         "claude"):   0.8825,
    ("fomc",         "llama"):    0.4458,
    ("fomc",         "deepseek"): 0.5535,
    ("fomc",         "gpt54"):    0.6431,
    ("fomc",         "mistral"):  0.3638,
    ("fomc",         "claude"):   0.5945,
    ("misogynistic", "llama"):    0.6451,
    ("misogynistic", "deepseek"): 0.6120,
    ("misogynistic", "gpt54"):    0.5760,
    ("misogynistic", "mistral"):  0.6170,
    ("misogynistic", "claude"):   0.6760,
    ("vuamc",        "llama"):    0.6300,
    ("vuamc",        "deepseek"): 0.6852,
    ("vuamc",        "gpt54"):    0.7855,
    ("vuamc",        "mistral"):  0.5801,
    ("vuamc",        "claude"):   0.8029,
}

DATASETS = ["cuad", "fomc", "misogynistic", "vuamc"]
LLMS     = ["llama", "deepseek", "gpt54", "mistral", "claude"]
METHODS  = ["dsl", "ppi", "ppipp"]

PHASE_LABELS = {
    "prevalence":    "Phase 1 (prevalence)",
    "low_variance":  "Phase 2 (low-var)",
    "high_variance": "Phase 3 (high-var)",
    "full_logistic": "Phase 4 (full logistic)",
}

SUMMARIES_DIR = os.path.join(
    os.path.dirname(__file__), "../../results/experiment 1/summaries"
)
FIG_DIR = os.path.join(
    os.path.dirname(__file__), "../../results/experiment 1/figures/kappa_correlation"
)
os.makedirs(FIG_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 2. Load all summary CSVs
# ------------------------------------------------------------------
def load_summaries():
    frames = []
    for dataset in DATASETS:
        folder = os.path.join(SUMMARIES_DIR, dataset)
        for csv_path in glob.glob(os.path.join(folder, "*.csv")):
            df = pd.read_csv(csv_path)
            # Normalise column names: full_logistic uses sRMSE_eucl
            if "sRMSE_eucl" in df.columns:
                df = df.rename(columns={"sRMSE_eucl": "sRMSE",
                                        "sRMSE_eucl_se": "sRMSE_se"})
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------
# 3. Compute improvement = mean_n[sRMSE(expert_only) - sRMSE(method)]
# ------------------------------------------------------------------
def compute_improvements(df):
    records = []
    for dataset in DATASETS:
        for llm in LLMS:
            kappa = KAPPA.get((dataset, llm))
            if kappa is None:
                continue
            sub = df[(df["dataset"] == dataset) & (df["llm"] == llm)]
            if sub.empty:
                continue
            for phase in sub["phase"].unique():
                psub = sub[sub["phase"] == phase]
                expert = psub[psub["method"] == "expert_only"][["n_expert", "sRMSE"]]
                if expert.empty:
                    continue
                expert = expert.set_index("n_expert")["sRMSE"]
                for method in METHODS:
                    msub = psub[psub["method"] == method][["n_expert", "sRMSE"]]
                    if msub.empty:
                        continue
                    msub = msub.set_index("n_expert")["sRMSE"]
                    # Align on shared n values
                    shared = expert.index.intersection(msub.index)
                    if len(shared) == 0:
                        continue
                    improvement = (expert[shared] - msub[shared]).mean()
                    records.append({
                        "dataset":     dataset,
                        "llm":         llm,
                        "phase":       phase,
                        "method":      method,
                        "kappa":       kappa,
                        "improvement": improvement,
                    })
    return pd.DataFrame(records)


# ------------------------------------------------------------------
# 4. Correlation table
# ------------------------------------------------------------------
def correlation_table(imp_df):
    rows = []
    for phase in ["prevalence", "low_variance", "high_variance", "full_logistic"]:
        for method in METHODS:
            sub = imp_df[(imp_df["phase"] == phase) & (imp_df["method"] == method)]
            if len(sub) < 3:
                continue
            r, p = stats.pearsonr(sub["kappa"], sub["improvement"])
            rows.append({
                "phase":  phase,
                "method": method,
                "n":      len(sub),
                "r":      r,
                "p":      p,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 5. Scatter plots: one panel per phase, coloured by method
# ------------------------------------------------------------------
METHOD_COLORS = {"dsl": "#1f77b4", "ppi": "#d62728", "ppipp": "#2ca02c"}
METHOD_LABELS = {"dsl": "DSL", "ppi": "PPI", "ppipp": "PPI++"}
DATASET_MARKERS = {"cuad": "o", "fomc": "s", "misogynistic": "^", "vuamc": "D"}


def scatter_plots(imp_df, corr_df):
    phases = ["prevalence", "low_variance", "high_variance", "full_logistic"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)

    for ax, phase in zip(axes, phases):
        psub = imp_df[imp_df["phase"] == phase]
        for method in METHODS:
            msub = psub[psub["method"] == method]
            for dataset in DATASETS:
                dsub = msub[msub["dataset"] == dataset]
                ax.scatter(dsub["kappa"], dsub["improvement"],
                           color=METHOD_COLORS[method],
                           marker=DATASET_MARKERS[dataset],
                           s=60, alpha=0.8,
                           label=f"{METHOD_LABELS[method]} / {dataset}")
            # Regression line
            all_m = psub[psub["method"] == method]
            if len(all_m) >= 3:
                x = all_m["kappa"].values
                y = all_m["improvement"].values
                m, b = np.polyfit(x, y, 1)
                xr = np.linspace(x.min(), x.max(), 100)
                ax.plot(xr, m * xr + b, color=METHOD_COLORS[method],
                        linewidth=1.5, linestyle="--", alpha=0.7)
        # Annotate with r values
        rc = corr_df[corr_df["phase"] == phase]
        anno = "\n".join(
            f"{METHOD_LABELS[row['method']]}: r={row['r']:.2f}"
            + ("*" if row["p"] < 0.05 else "")
            for _, row in rc.iterrows()
        )
        ax.text(0.03, 0.97, anno, transform=ax.transAxes,
                verticalalignment="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_title(PHASE_LABELS.get(phase, phase), fontsize=10)
        ax.set_xlabel("Cohen's κ", fontsize=9)
        ax.set_ylabel("sRMSE improvement over expert-only", fontsize=9)

    # Combined legend (methods only, deduplicated)
    from matplotlib.lines import Line2D
    method_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8,
               label=METHOD_LABELS[m])
        for m, c in METHOD_COLORS.items()
    ]
    dataset_handles = [
        Line2D([0], [0], marker=mk, color="gray", markersize=8,
               label=ds.capitalize(), linestyle="None")
        for ds, mk in DATASET_MARKERS.items()
    ]
    fig.legend(handles=method_handles + dataset_handles,
               loc="lower center", ncol=7, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "kappa_vs_improvement.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "kappa_vs_improvement.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot to {FIG_DIR}/kappa_vs_improvement.pdf")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
if __name__ == "__main__":
    df = load_summaries()
    print(f"Loaded {len(df)} rows from summaries.")

    imp_df = compute_improvements(df)
    print(f"Computed {len(imp_df)} improvement values.")

    corr_df = correlation_table(imp_df)
    print("\n=== Pearson r between κ and sRMSE improvement ===")
    print("(improvement > 0 means method beats expert-only)")
    print("* = p < 0.05\n")
    for phase in ["prevalence", "low_variance", "high_variance", "full_logistic"]:
        sub = corr_df[corr_df["phase"] == phase]
        print(f"  {PHASE_LABELS.get(phase, phase)}:")
        for _, row in sub.iterrows():
            sig = "*" if row["p"] < 0.05 else ""
            print(f"    {METHOD_LABELS[row['method']]:<8}  r = {row['r']:+.3f}  "
                  f"p = {row['p']:.3f}{sig}  (n={int(row['n'])})")

    # Save table as CSV
    csv_out = os.path.join(FIG_DIR, "kappa_correlation_table.csv")
    corr_df.to_csv(csv_out, index=False)
    print(f"\nSaved table to {csv_out}")

    scatter_plots(imp_df, corr_df)
