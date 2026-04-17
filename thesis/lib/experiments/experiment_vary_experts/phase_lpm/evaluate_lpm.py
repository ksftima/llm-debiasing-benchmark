"""
Evaluation script: Vary Expert Sample Size — Linear Probability Model (LPM)

Loads all 300 .npz repetition files and computes metrics for θ = [β₀, β_feature].

Metrics computed:
    sRMSE_beta2  = sqrt( E[ ((β_feature - β*_feature) / β*_feature)² ] )
    bias_beta2   =        E[ (β_feature - β*_feature)  / β*_feature  ]
    sRMSE_eucl   = sqrt( E[ ||β - β*||² / ||β*||² ] )

Output: a CSV with one row per (method, n) combination.

Usage:
    python evaluate_lpm.py <results_dir> --dataset cuad --llm llama --output out.csv --phase low_variance
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

    all_theta_star, all_theta_llm   = [], []
    all_thetas_exp, all_thetas_dsl  = [], []
    all_thetas_ppi, all_thetas_ppipp = [], []

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
        "n_values":     n_values,
        "theta_star":   np.stack(all_theta_star),    # (num_reps, 2)
        "theta_llm":    np.stack(all_theta_llm),     # (num_reps, 2)
        "thetas_exp":   np.stack(all_thetas_exp),    # (num_reps, num_n, 2)
        "thetas_dsl":   np.stack(all_thetas_dsl),
        "thetas_ppi":   np.stack(all_thetas_ppi),
        "thetas_ppipp": np.stack(all_thetas_ppipp),
    }


def compute_metrics_beta2(betas2, beta2_star):
    normalized = (betas2 - beta2_star) / beta2_star
    n          = int(np.sum(~np.isnan(normalized)))
    srmse      = float(np.sqrt(np.nanmean(normalized ** 2)))
    std_bias   = float(np.nanmean(normalized))
    se         = float(np.nanstd(normalized) / np.sqrt(n)) if n > 1 else np.nan
    return srmse, std_bias, se, se, n


def compute_metrics_euclidean(betas, beta_star):
    valid     = ~np.isnan(betas).any(axis=1)
    betas     = betas[valid]
    beta_star = beta_star[valid]
    n         = int(valid.sum())
    sq_error  = np.sum((betas - beta_star) ** 2, axis=1)
    ref_norm  = np.sum(beta_star ** 2, axis=1)
    ratio     = sq_error / ref_norm
    srmse     = float(np.sqrt(np.mean(ratio)))
    srmse_se  = float(np.std(np.sqrt(ratio)) / np.sqrt(n)) if n > 1 else np.nan
    return srmse, srmse_se


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--dataset",  type=str, required=True)
    parser.add_argument("--llm",      type=str, required=True)
    parser.add_argument("--output",   type=Path, required=True)
    parser.add_argument("--phase",    type=str, default="low_variance",
        choices=["low_variance", "high_variance"])
    args = parser.parse_args()

    data     = load_all_reps(args.results_dir)
    n_values = data["n_values"]
    num_reps = data["theta_star"].shape[0]

    beta_star = data["theta_star"]

    print(f"\nDataset: {args.dataset} | LLM: {args.llm} | Reps: {num_reps}")
    print(f"β*₀ mean: {beta_star[:, 0].mean():.4f}  β*_feature mean: {beta_star[:, 1].mean():.4f}")
    print(f"n values: {n_values.tolist()}\n")

    rows = []

    # LLM-only baseline
    llm_betas = data["theta_llm"]
    srmse_b2, bias_b2, srmse_b2_se, bias_b2_se, n_valid = compute_metrics_beta2(
        llm_betas[:, 1], beta_star[:, 1]
    )
    srmse_eucl, srmse_eucl_se = compute_metrics_euclidean(llm_betas, beta_star)
    rows.append({
        "dataset": args.dataset, "llm": args.llm, "method": "llm_only",
        "n_expert": None,
        "sRMSE_beta2": round(srmse_b2, 6),    "sRMSE_beta2_se": round(srmse_b2_se, 6),
        "bias_beta2":  round(bias_b2, 6),      "bias_beta2_se":  round(bias_b2_se, 6),
        "sRMSE_eucl":  round(srmse_eucl, 6),   "sRMSE_eucl_se":  round(srmse_eucl_se, 6),
        "n_reps": num_reps, "n_valid": n_valid, "phase": args.phase,
    })

    methods = {
        "expert_only": data["thetas_exp"],
        "dsl":         data["thetas_dsl"],
        "ppi":         data["thetas_ppi"],
        "ppipp":       data["thetas_ppipp"],
    }

    for method_name, all_thetas in methods.items():
        for i, n in enumerate(n_values):
            betas_at_n = all_thetas[:, i, :]
            srmse_b2, bias_b2, srmse_b2_se, bias_b2_se, n_valid = compute_metrics_beta2(
                betas_at_n[:, 1], beta_star[:, 1]
            )
            srmse_eucl, srmse_eucl_se = compute_metrics_euclidean(betas_at_n, beta_star)
            rows.append({
                "dataset": args.dataset, "llm": args.llm, "method": method_name,
                "n_expert": int(n),
                "sRMSE_beta2": round(srmse_b2, 6),    "sRMSE_beta2_se": round(srmse_b2_se, 6),
                "bias_beta2":  round(bias_b2, 6),      "bias_beta2_se":  round(bias_b2_se, 6),
                "sRMSE_eucl":  round(srmse_eucl, 6),   "sRMSE_eucl_se":  round(srmse_eucl_se, 6),
                "n_reps": num_reps, "n_valid": n_valid, "phase": args.phase,
            })

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved to {args.output}")
