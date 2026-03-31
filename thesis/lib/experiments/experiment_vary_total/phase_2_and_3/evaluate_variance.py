"""
Evaluation script: Vary Total Dataset Size — Phase 2/3: Single Feature

Metrics:
    sRMSE_beta2 = sqrt( E[ ((β₂ - β*₂) / β*₂)² ] )
    bias_beta2  =        E[ (β₂ - β*₂)  / β*₂  ]

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
    phase    = str(d0["phase"][0])

    return {
        "N_values":    N_values,
        "n_expert":    n_expert,
        "phase":       phase,
        "theta_star":  np.stack(all_theta_star),   # (num_reps, 2)
        "theta_llm":   np.stack(all_theta_llm),    # (num_reps, 2)
        "thetas_exp":  np.stack(all_thetas_exp),   # (num_reps, num_N, 2)
        "thetas_dsl":  np.stack(all_thetas_dsl),
        "thetas_ppi":  np.stack(all_thetas_ppi),
        "thetas_ppipp":np.stack(all_thetas_ppipp),
    }


def compute_metrics_beta2(betas2, beta2_star):
    normalized = (betas2 - beta2_star) / beta2_star
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
    phase    = data["phase"]
    num_reps = data["theta_star"].shape[0]

    beta_star = data["theta_star"]  # (num_reps, 2)

    print(f"\nDataset: {args.dataset} | LLM: {args.llm} | Phase: {phase} | n_expert: {n_expert} | Reps: {num_reps}")
    print(f"β*₀ mean: {beta_star[:, 0].mean():.4f}  β*₂ mean: {beta_star[:, 1].mean():.4f}")
    print(f"N values: {N_values.tolist()}\n")

    rows = []

    # LLM-only baseline
    llm_betas = data["theta_llm"]
    srmse, bias, se, n_valid = compute_metrics_beta2(llm_betas[:, 1], beta_star[:, 1])
    rows.append({
        "dataset":       args.dataset,
        "llm":           args.llm,
        "method":        "llm_only",
        "n_expert":      n_expert,
        "N_total":       None,
        "sRMSE_beta2":   round(srmse, 6),
        "sRMSE_beta2_se":round(se,    6),
        "bias_beta2":    round(bias,  6),
        "bias_beta2_se": round(se,    6),
        "n_reps":        num_reps,
        "n_valid":       n_valid,
        "phase":         phase,
    })

    methods = {
        "expert_only": data["thetas_exp"],
        "dsl":         data["thetas_dsl"],
        "ppi":         data["thetas_ppi"],
        "ppipp":       data["thetas_ppipp"],
    }

    for method_name, all_thetas in methods.items():
        for i, N in enumerate(N_values):
            betas_at_N = all_thetas[:, i, :]  # (num_reps, 2)
            srmse, bias, se, n_valid = compute_metrics_beta2(betas_at_N[:, 1], beta_star[:, 1])
            rows.append({
                "dataset":       args.dataset,
                "llm":           args.llm,
                "method":        method_name,
                "n_expert":      n_expert,
                "N_total":       int(N),
                "sRMSE_beta2":   round(srmse, 6),
                "sRMSE_beta2_se":round(se,    6),
                "bias_beta2":    round(bias,  6),
                "bias_beta2_se": round(se,    6),
                "n_reps":        num_reps,
                "n_valid":       n_valid,
                "phase":         phase,
            })

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved to {args.output}")
