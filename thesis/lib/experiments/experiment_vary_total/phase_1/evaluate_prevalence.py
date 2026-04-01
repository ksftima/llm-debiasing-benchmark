"""
Evaluation script: Vary Total Dataset Size — Phase 1: Class Prevalence

Loads all 300 .npz repetition files and computes sRMSE and bias for β₀.

Output: CSV with one row per (method, N_total) combination.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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

    d0       = np.load(files[0])
    N_values = d0["N_values"]
    n_expert = int(d0["n_expert"][0])

    return {
        "N_values":    N_values,
        "n_expert":    n_expert,
        "theta_star":  np.stack(all_theta_star),   # (num_reps, 1)
        "theta_llm":   np.stack(all_theta_llm),    # (num_reps, 1)
        "thetas_exp":  np.stack(all_thetas_exp),   # (num_reps, num_N)
        "thetas_dsl":  np.stack(all_thetas_dsl),   # (num_reps, num_N)
        "thetas_ppi":  np.stack(all_thetas_ppi),   # (num_reps, num_N)
        "thetas_ppipp":np.stack(all_thetas_ppipp), # (num_reps, num_N)
    }


def compute_metrics(betas, beta_star):
    """
    Standardized sRMSE and bias for a scalar coefficient.
    betas, beta_star : shape (num_reps,)
    """
    normalized = (betas - beta_star) / beta_star
    n          = int(np.sum(~np.isnan(normalized)))
    srmse      = float(np.sqrt(np.nanmean(normalized ** 2)))
    std_bias   = float(np.nanmean(normalized))
    se         = float(np.nanstd(normalized) / np.sqrt(n)) if n > 1 else np.nan
    return srmse, std_bias, se, n


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

    beta_star = sigmoid(data["theta_star"][:, 0])  # convert log-odds → probability

    print(f"\nDataset: {args.dataset} | LLM: {args.llm} | n_expert: {n_expert} | Reps: {num_reps}")
    print(f"p*₀ mean: {beta_star.mean():.4f}")
    print(f"N values: {N_values.tolist()}\n")

    rows = []

    # LLM-only baseline (constant — no N dependence)
    llm_betas = sigmoid(data["theta_llm"][:, 0])
    srmse, bias, se, n_valid = compute_metrics(llm_betas, beta_star)
    rows.append({
        "dataset":   args.dataset,
        "llm":       args.llm,
        "method":    "llm_only",
        "n_expert":  n_expert,
        "N_total":   None,
        "sRMSE":     round(srmse, 6),
        "sRMSE_se":  round(se,    6),
        "bias":      round(bias,  6),
        "bias_se":   round(se,    6),
        "n_reps":    num_reps,
        "n_valid":   n_valid,
        "phase":     "prevalence",
    })

    methods = {
        "expert_only": data["thetas_exp"],
        "dsl":         data["thetas_dsl"],
        "ppi":         data["thetas_ppi"],
        "ppipp":       data["thetas_ppipp"],
    }

    for method_name, all_thetas in methods.items():
        for i, N in enumerate(N_values):
            betas_at_N = sigmoid(all_thetas[:, i])  # convert log-odds → probability
            srmse, bias, se, n_valid = compute_metrics(betas_at_N, beta_star)
            rows.append({
                "dataset":   args.dataset,
                "llm":       args.llm,
                "method":    method_name,
                "n_expert":  n_expert,
                "N_total":   int(N),
                "sRMSE":     round(srmse, 6),
                "sRMSE_se":  round(se,    6),
                "bias":      round(bias,  6),
                "bias_se":   round(se,    6),
                "n_reps":    num_reps,
                "n_valid":   n_valid,
                "phase":     "prevalence",
            })

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved to {args.output}")
