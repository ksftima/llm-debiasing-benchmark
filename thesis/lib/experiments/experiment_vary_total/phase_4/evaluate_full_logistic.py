"""
Evaluation script: Vary Total Dataset Size — Phase 4: Full Logistic Regression

Metric: sRMSE_eucl — Euclidean-norm standardized RMSE over full 6-vector.
        bias_eucl  — mean standardized bias (averaged over coefficients).

Output: CSV with one row per (method, N_total) combination.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


def load_all_reps(results_dir: Path):
    files = sorted(results_dir.glob("rep_*.npz"))
    if not files:
        raise FileNotFoundError(f"No rep_*.npz files found in {results_dir}")

    print(f"Loading {len(files)} repetitions from {results_dir}")

    all_theta_star   = []
    all_theta_llm    = []
    all_thetas_exp   = []
    all_thetas_dsl   = []
    all_thetas_ppi   = []
    all_thetas_ppipp = []

    for f in files:
        d = np.load(f)
        all_theta_star.append(d["theta_star"])
        all_theta_llm.append(d["theta_llm"])
        all_thetas_exp.append(d["thetas_exp"])
        all_thetas_dsl.append(d["thetas_dsl"])
        all_thetas_ppi.append(d["thetas_ppi"])
        all_thetas_ppipp.append(d["thetas_ppipp"])

    d0       = np.load(files[0])
    N_values = d0["N_values"]
    n_expert = int(d0["n_expert"][0])

    return {
        "N_values":    N_values,
        "n_expert":    n_expert,
        "theta_star":  np.stack(all_theta_star),   # (num_reps, 6)
        "theta_llm":   np.stack(all_theta_llm),    # (num_reps, 6)
        "thetas_exp":  np.stack(all_thetas_exp),   # (num_reps, num_N, 6)
        "thetas_dsl":  np.stack(all_thetas_dsl),
        "thetas_ppi":  np.stack(all_thetas_ppi),
        "thetas_ppipp":np.stack(all_thetas_ppipp),
    }


def compute_metrics_euclidean(betas, beta_star):
    """
    sRMSE_eucl = sqrt( mean( ||β - β*||² / ||β*||² ) ) over reps.
    bias_eucl  = mean( mean_coeff( (β - β*) / β* ) ) over reps.
    """
    valid     = ~np.isnan(betas).any(axis=1)
    betas_v   = betas[valid]
    beta_s_v  = beta_star[valid]
    n         = int(valid.sum())

    sq_error  = np.sum((betas_v - beta_s_v) ** 2, axis=1)
    ref_norm  = np.sum(beta_s_v ** 2, axis=1)
    ratio     = sq_error / ref_norm
    srmse     = float(np.sqrt(np.mean(ratio)))
    srmse_se  = float(np.std(np.sqrt(ratio)) / np.sqrt(n)) if n > 1 else np.nan

    # bias: mean over coefficients, then mean over reps
    norm_diff  = (betas_v - beta_s_v) / beta_s_v
    bias_per_rep = np.mean(norm_diff, axis=1)
    bias       = float(np.mean(bias_per_rep))
    bias_se    = float(np.std(bias_per_rep) / np.sqrt(n)) if n > 1 else np.nan

    return srmse, srmse_se, bias, bias_se, n


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--dataset",   type=str, required=True)
    parser.add_argument("--llm",       type=str, required=True)
    parser.add_argument("--output",    type=Path, required=True)
    args = parser.parse_args()

    data     = load_all_reps(args.results_dir)
    N_values = data["N_values"]
    n_expert = data["n_expert"]
    num_reps = data["theta_star"].shape[0]

    beta_star = data["theta_star"]  # (num_reps, 6)

    print(f"\nDataset: {args.dataset} | LLM: {args.llm} | n_expert: {n_expert} | Reps: {num_reps}")
    print(f"N values: {N_values.tolist()}\n")

    rows = []

    # LLM-only baseline
    llm_betas = data["theta_llm"]
    srmse, srmse_se, bias, bias_se, n_valid = compute_metrics_euclidean(llm_betas, beta_star)
    rows.append({
        "dataset":      args.dataset,
        "llm":          args.llm,
        "method":       "llm_only",
        "n_expert":     n_expert,
        "N_total":      None,
        "sRMSE_eucl":   round(srmse,    6),
        "sRMSE_eucl_se":round(srmse_se, 6),
        "bias_eucl":    round(bias,     6),
        "bias_eucl_se": round(bias_se,  6),
        "n_reps":       num_reps,
        "n_valid":      n_valid,
        "phase":        "full_logistic",
    })

    methods = {
        "expert_only": data["thetas_exp"],
        "dsl":         data["thetas_dsl"],
        "ppi":         data["thetas_ppi"],
        "ppipp":       data["thetas_ppipp"],
    }

    for method_name, all_thetas in methods.items():
        for i, N in enumerate(N_values):
            betas_at_N = all_thetas[:, i, :]  # (num_reps, 6)
            srmse, srmse_se, bias, bias_se, n_valid = compute_metrics_euclidean(betas_at_N, beta_star)
            rows.append({
                "dataset":      args.dataset,
                "llm":          args.llm,
                "method":       method_name,
                "n_expert":     n_expert,
                "N_total":      int(N),
                "sRMSE_eucl":   round(srmse,    6),
                "sRMSE_eucl_se":round(srmse_se, 6),
                "bias_eucl":    round(bias,     6),
                "bias_eucl_se": round(bias_se,  6),
                "n_reps":       num_reps,
                "n_valid":      n_valid,
                "phase":        "full_logistic",
            })

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved to {args.output}")
