"""
Evaluation script: Vary Expert Sample Size — Phase 1: Class Prevalence

Loads all 300 .npz repetition files and computes the two metrics from the
planning report for each method at each n value:

    sRMSE(θ; d)        = sqrt( E[ ((θ - θ*) / θ*)² ] )
    Standardized Bias  =        E[ (θ - θ*)  / θ*    ]

where E[·] is the expectation taken over the 300 repetitions.

Output: a CSV with one row per (method, n) combination.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


def load_all_reps(results_dir: Path):
    """
    Load every rep_*.npz file in results_dir.
    Returns a dict of stacked arrays, shape (num_reps, ...) for each key.
    """
    files = sorted(results_dir.glob("rep_*.npz"))
    if len(files) == 0:
        raise FileNotFoundError(f"No rep_*.npz files found in {results_dir}")

    print(f"Loading {len(files)} repetitions from {results_dir}")

    # Load each file and collect arrays
    all_theta_star  = []
    all_theta_llm   = []
    all_thetas_exp  = []
    all_thetas_dsl  = []
    all_thetas_ppi  = []

    for f in files:
        d = np.load(f)
        all_theta_star.append(d["theta_star"])
        all_theta_llm.append(d["theta_llm"])
        all_thetas_exp.append(d["thetas_exp"])
        all_thetas_dsl.append(d["thetas_dsl"])
        all_thetas_ppi.append(d["thetas_ppi"])

    # n_values is the same across all reps — just take from the first file
    n_values = np.load(files[0])["n_values"]

    return {
        "n_values":    n_values,                          # shape (num_n,)
        "theta_star":  np.stack(all_theta_star),          # shape (num_reps, 1)
        "theta_llm":   np.stack(all_theta_llm),           # shape (num_reps, 1)
        "thetas_exp":  np.stack(all_thetas_exp),          # shape (num_reps, num_n)
        "thetas_dsl":  np.stack(all_thetas_dsl),          # shape (num_reps, num_n)
        "thetas_ppi":  np.stack(all_thetas_ppi),          # shape (num_reps, num_n)
    }


def compute_metrics(thetas, theta_star):
    """
    Compute sRMSE and standardized bias for one method.

    thetas     : shape (num_reps,)  — one estimate per repetition
    theta_star : shape (num_reps,)  — reference, same value in every row but
                                      stacked this way for vectorised division

    sRMSE = sqrt( mean( ((θ - θ*) / θ*)² ) )
    Std Bias = mean( (θ - θ*) / θ* )

    Dividing by θ* standardises the error so results are comparable across
    datasets with very different coefficient magnitudes (per planning report).
    """
    normalized = (thetas - theta_star) / theta_star  # shape (num_reps,)
    srmse      = float(np.sqrt(np.mean(normalized ** 2)))
    std_bias   = float(np.mean(normalized))
    return srmse, std_bias


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("results_dir", type=Path,
        help="Directory containing rep_*.npz files, e.g. thesis/results/vary-expert-prevalence/cuad/llama/")
    parser.add_argument("--dataset", type=str, required=True,
        help="Dataset name, e.g. cuad")
    parser.add_argument("--llm", type=str, required=True,
        help="LLM name, e.g. llama")
    parser.add_argument("--output", type=Path, required=True,
        help="Where to save the summary CSV")
    args = parser.parse_args()

    # --- Load all repetitions ---
    data      = load_all_reps(args.results_dir)
    n_values  = data["n_values"]          # e.g. [20, 26, 33, ..., 200]
    num_reps  = data["theta_star"].shape[0]

    # theta_star is deterministic (uses all expert labels, no randomness)
    # so all 300 values are identical — we just use it as a scalar
    theta_star_scalar = data["theta_star"][:, 0]  # shape (num_reps,)

    print(f"\nDataset: {args.dataset} | LLM: {args.llm} | Reps: {num_reps}")
    print(f"theta_star (log-odds): {theta_star_scalar[0]:.4f}  (same across all reps)")
    print(f"n values: {n_values.tolist()}\n")

    rows = []

    # --- LLM-only baseline ---
    # theta_llm is also deterministic (no random sampling, uses all LLM labels)
    # sRMSE and std_bias will be constants — but computed consistently
    theta_llm_scalar = data["theta_llm"][:, 0]  # shape (num_reps,)
    srmse_llm, bias_llm = compute_metrics(theta_llm_scalar, theta_star_scalar)

    # LLM-only does not vary with n, so we add one row with n=NaN
    rows.append({
        "dataset":          args.dataset,
        "llm":              args.llm,
        "method":           "llm_only",
        "n_expert":         None,
        "sRMSE":            round(srmse_llm, 6),
        "standardized_bias": round(bias_llm, 6),
        "n_reps":           num_reps,
        "phase":            "prevalence",
    })

    # --- Expert-only, DSL, PPI — computed for each n value ---
    methods = {
        "expert_only": data["thetas_exp"],  # shape (num_reps, num_n)
        "dsl":         data["thetas_dsl"],
        "ppi":         data["thetas_ppi"],
    }

    for method_name, all_thetas in methods.items():
        for i, n in enumerate(n_values):
            thetas_at_n = all_thetas[:, i]   # shape (num_reps,) — one value per rep
            srmse, std_bias = compute_metrics(thetas_at_n, theta_star_scalar)

            rows.append({
                "dataset":           args.dataset,
                "llm":               args.llm,
                "method":            method_name,
                "n_expert":          int(n),
                "sRMSE":             round(srmse, 6),
                "standardized_bias": round(std_bias, 6),
                "n_reps":            num_reps,
                "phase":             "prevalence",
            })

    # --- Save ---
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved to {args.output}")
