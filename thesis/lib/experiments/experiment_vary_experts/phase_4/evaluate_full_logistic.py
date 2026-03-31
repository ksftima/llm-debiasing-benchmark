"""
Evaluation script: Vary Expert Sample Size — Phase 4: Full Logistic Regression

Loads all 300 .npz repetition files and computes metrics for θ = [β₀, β₁, β₂, β₃, β₄, β₅].

Metrics computed:
    sRMSE_eucl = sqrt( E[ ||β - β*||² / ||β*||² ] )   Euclidean-norm sRMSE (main metric)
    bias_eucl  = E[ mean_k( (βk - β*k) / β*k ) ]      Standardized bias averaged over coefficients

Output: a CSV with one row per (method, n) combination.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

N_COEF = 6  # intercept + 5 features


def load_all_reps(results_dir: Path):
    files = sorted(results_dir.glob("rep_*.npz"))
    if not files:
        raise FileNotFoundError(f"No rep_*.npz files found in {results_dir}")

    print(f"Loading {len(files)} repetitions from {results_dir}")

    all_theta_star  = []
    all_theta_llm   = []
    all_thetas_exp  = []
    all_thetas_dsl  = []
    all_thetas_ppi  = []
    all_thetas_ppipp = []

    for f in files:
        d = np.load(f)
        all_theta_star.append(d["theta_star"])
        all_theta_llm.append(d["theta_llm"])
        all_thetas_exp.append(d["thetas_exp"])
        all_thetas_dsl.append(d["thetas_dsl"])
        all_thetas_ppi.append(d["thetas_ppi"])
        all_thetas_ppipp.append(d["thetas_ppipp"])

    n_values = np.load(files[0])["n_values"]

    return {
        "n_values":    n_values,
        "theta_star":  np.stack(all_theta_star),    # (num_reps, 6)
        "theta_llm":   np.stack(all_theta_llm),     # (num_reps, 6)
        "thetas_exp":  np.stack(all_thetas_exp),    # (num_reps, num_n, 6)
        "thetas_dsl":  np.stack(all_thetas_dsl),
        "thetas_ppi":  np.stack(all_thetas_ppi),
        "thetas_ppipp": np.stack(all_thetas_ppipp),
    }


def compute_metrics_euclidean(betas, beta_star):
    """
    Euclidean sRMSE: sqrt( mean( ||β - β*||² / ||β*||² ) ) over reps.
    Rows with any NaN are skipped. Returns n_valid.
    """
    valid     = ~np.isnan(betas).any(axis=1)
    betas_v   = betas[valid]
    bstar_v   = beta_star[valid]
    n         = int(valid.sum())
    sq_error  = np.sum((betas_v - bstar_v) ** 2, axis=1)
    ref_norm  = np.sum(bstar_v ** 2, axis=1)
    ratio     = sq_error / ref_norm
    srmse     = float(np.sqrt(np.mean(ratio)))
    srmse_se  = float(np.std(np.sqrt(ratio)) / np.sqrt(n)) if n > 1 else np.nan
    return srmse, srmse_se, n


def compute_bias_euclidean(betas, beta_star):
    """
    Standardized bias: E[ mean_k( (βk - β*k) / β*k ) ] over reps.
    Matches paper Appendix H formula, averaged over coefficients for a scalar summary.
    Rows with any NaN are skipped.
    """
    valid   = ~np.isnan(betas).any(axis=1)
    betas_v = betas[valid]
    bstar_v = beta_star[valid]
    n       = int(valid.sum())
    # element-wise standardized error, averaged over 6 coefficients per rep
    per_rep = np.mean((betas_v - bstar_v) / bstar_v, axis=1)  # (n_valid,)
    bias    = float(np.mean(per_rep))
    bias_se = float(np.std(per_rep) / np.sqrt(n)) if n > 1 else np.nan
    return bias, bias_se, n


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--dataset",   type=str, required=True)
    parser.add_argument("--llm",       type=str, required=True)
    parser.add_argument("--output",    type=Path, required=True)
    args = parser.parse_args()

    data     = load_all_reps(args.results_dir)
    n_values = data["n_values"]
    num_reps = data["theta_star"].shape[0]
    beta_star = data["theta_star"]  # (num_reps, 6)

    print(f"\nDataset: {args.dataset} | LLM: {args.llm} | Reps: {num_reps}")
    print(f"n values: {n_values.tolist()}\n")

    rows = []

    # --- LLM-only baseline ---
    llm_betas = data["theta_llm"]
    srmse_eucl, srmse_eucl_se, n_valid = compute_metrics_euclidean(llm_betas, beta_star)
    bias_eucl,  bias_eucl_se,  _       = compute_bias_euclidean(llm_betas, beta_star)
    rows.append({
        "dataset":       args.dataset,
        "llm":           args.llm,
        "method":        "llm_only",
        "n_expert":      None,
        "sRMSE_eucl":    round(srmse_eucl,   6),
        "sRMSE_eucl_se": round(srmse_eucl_se,6),
        "bias_eucl":     round(bias_eucl,    6),
        "bias_eucl_se":  round(bias_eucl_se, 6),
        "n_reps":        num_reps,
        "n_valid":       n_valid,
        "phase":         "full_logistic",
    })

    # --- Expert-only, DSL, PPI per n ---
    methods = {
        "expert_only": data["thetas_exp"],
        "dsl":         data["thetas_dsl"],
        "ppi":         data["thetas_ppi"],
        "ppipp":       data["thetas_ppipp"],
    }

    for method_name, all_thetas in methods.items():
        for i, n in enumerate(n_values):
            betas_at_n = all_thetas[:, i, :]  # (num_reps, 6)
            srmse_eucl, srmse_eucl_se, n_valid = compute_metrics_euclidean(
                betas_at_n, beta_star
            )
            bias_eucl, bias_eucl_se, _ = compute_bias_euclidean(
                betas_at_n, beta_star
            )
            rows.append({
                "dataset":       args.dataset,
                "llm":           args.llm,
                "method":        method_name,
                "n_expert":      int(n),
                "sRMSE_eucl":    round(srmse_eucl,   6),
                "sRMSE_eucl_se": round(srmse_eucl_se,6),
                "bias_eucl":     round(bias_eucl,    6),
                "bias_eucl_se":  round(bias_eucl_se, 6),
                "n_reps":        num_reps,
                "n_valid":       n_valid,
                "phase":         "full_logistic",
            })

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved to {args.output}")
