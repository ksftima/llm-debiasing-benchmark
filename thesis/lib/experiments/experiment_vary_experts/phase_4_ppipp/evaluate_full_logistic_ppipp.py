"""
Evaluation script: Vary Expert Sample Size — Phase 4: Full Logistic (PPI++ only)

Loads rep_*.npz files produced by expert_full_logistic_ppipp.py and computes metrics
for the ppipp method only. Output CSV has the same schema as evaluate_full_logistic.py
so it can be merged directly with the originals.
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

    all_theta_star  = []
    all_theta_llm   = []
    all_thetas_ppipp = []

    for f in files:
        d = np.load(f)
        all_theta_star.append(d["theta_star"])
        all_theta_llm.append(d["theta_llm"])
        all_thetas_ppipp.append(d["thetas_ppipp"])

    n_values = np.load(files[0])["n_values"]

    return {
        "n_values":     n_values,
        "theta_star":   np.stack(all_theta_star),    # (num_reps, 6)
        "theta_llm":    np.stack(all_theta_llm),     # (num_reps, 6)
        "thetas_ppipp": np.stack(all_thetas_ppipp),  # (num_reps, num_n, 6)
    }


def compute_metrics_euclidean(betas, beta_star):
    valid    = ~np.isnan(betas).any(axis=1)
    betas_v  = betas[valid]
    bstar_v  = beta_star[valid]
    n        = int(valid.sum())
    ratio    = np.sum((betas_v - bstar_v) ** 2, axis=1) / np.sum(bstar_v ** 2, axis=1)
    srmse    = float(np.sqrt(np.mean(ratio)))
    srmse_se = float(np.std(np.sqrt(ratio)) / np.sqrt(n)) if n > 1 else np.nan
    return srmse, srmse_se, n


def compute_bias_euclidean(betas, beta_star):
    valid   = ~np.isnan(betas).any(axis=1)
    betas_v = betas[valid]
    bstar_v = beta_star[valid]
    n       = int(valid.sum())
    per_rep = np.mean((betas_v - bstar_v) / bstar_v, axis=1)
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

    data      = load_all_reps(args.results_dir)
    n_values  = data["n_values"]
    num_reps  = data["theta_star"].shape[0]
    beta_star = data["theta_star"]

    print(f"\nDataset: {args.dataset} | LLM: {args.llm} | Reps: {num_reps}")
    print(f"n values: {n_values.tolist()}\n")

    rows = []

    for i, n in enumerate(n_values):
        betas_at_n = data["thetas_ppipp"][:, i, :]
        srmse_eucl, srmse_eucl_se, n_valid = compute_metrics_euclidean(betas_at_n, beta_star)
        bias_eucl,  bias_eucl_se,  _       = compute_bias_euclidean(betas_at_n, beta_star)
        rows.append({
            "dataset":       args.dataset,
            "llm":           args.llm,
            "method":        "ppipp",
            "n_expert":      int(n),
            "sRMSE_eucl":    round(srmse_eucl,    6),
            "sRMSE_eucl_se": round(srmse_eucl_se, 6),
            "bias_eucl":     round(bias_eucl,     6),
            "bias_eucl_se":  round(bias_eucl_se,  6),
            "n_reps":        num_reps,
            "n_valid":       n_valid,
            "phase":         "full_logistic",
        })

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved to {args.output}")
