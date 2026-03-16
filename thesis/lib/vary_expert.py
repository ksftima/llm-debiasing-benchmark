import sys
sys.path.insert(0, "/code/original/lib")  # tells Python where to find fitting.py inside the container

import fitting as fit  # logit_fit, logit_fit_ppi, logit_fit_dsl live here

import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


def compute_one_n(X, Y, Y_hat, n, seed):
    """
    For one expert sample size n, randomly pick n rows as 'expert-labeled'
    and compute coefficients for all three debiasing conditions.

    X       : feature matrix, shape (N, 5)
    Y       : expert labels for ALL rows -- DSL/PPI use this for selected rows only
    Y_hat   : LLM labels for ALL rows   -- DSL/PPI use this for unlabeled rows
    n       : number of expert labels to simulate having (e.g. 50)
    seed    : makes the random sample reproducible per repetition
    """
    rng = np.random.default_rng(seed)
    # 'selected' is the index of the n rows we pretend are expert-labeled
    selected = rng.choice(len(Y), size=n, replace=False)

    # Expert-only: train only on the n selected rows, ignoring all LLM labels
    # High variance when n is small, but statistically unbiased
    coeffs_exp = fit.logit_fit(X[selected], Y[selected])

    # DSL: uses ALL LLM labels + the n expert labels to correct for LLM bias
    # Passes the full X, Y, Y_hat plus which rows are expert-labeled
    coeffs_dsl = fit.logit_fit_dsl(X, Y, Y_hat, selected)

    # PPI: alternative correction framework, same inputs as DSL
    coeffs_ppi = fit.logit_fit_ppi(X, Y, Y_hat, selected)

    return coeffs_exp, coeffs_dsl, coeffs_ppi


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path,
        help="Path to annotated CSV, e.g. cuad_llama_annotated.csv")
    parser.add_argument("results_path", type=Path,
        help="Where to save the .npz result for this repetition")
    # seed = SLURM array task ID, so each of the 300 jobs produces a different random sample
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    print(f"Repetition seed: {args.seed}")
    print(f"Reading data from: {args.annotated_csv}")

    data = pd.read_csv(args.annotated_csv)

    features = ["x1", "x2", "x3", "x4", "x5"]
    X = data[features].to_numpy()
    Y = data["y"].to_numpy().astype(float)         # expert labels
    Y_hat = data["y_hat"].to_numpy().astype(float) # LLM labels

    num_coeffs = X.shape[1] + 1  # 5 features + 1 intercept = 6 coefficients total

    # --- Reference model theta* ---
    # Logistic regression on ALL expert labels = gold standard.
    # Every other method's coefficients are compared against this using sRMSE.
    theta_star = fit.logit_fit(X, Y)

    # --- LLM-only baseline ---
    # Logistic regression on ALL LLM labels, zero expert labels used.
    # This is the biased-but-cheap scenario. Does not vary with n so computed once.
    theta_llm = fit.logit_fit(X, Y_hat)

    # --- Log-spaced expert sample sizes from 20 to 200 ---
    # 10 points gives: ~20, 26, 33, 43, 56, 72, 93, 120, 155, 200
    # Log spacing means more points at the low end where things are most interesting
    n_values = np.round(
        np.logspace(np.log10(20), np.log10(200), num=10)
    ).astype(int)
    # Remove any duplicates that rounding might create
    n_values = np.unique(n_values)

    print(f"Expert sample sizes: {n_values.tolist()}")

    num_n = len(n_values)
    coeffs_exp = np.zeros((num_n, num_coeffs))
    coeffs_dsl = np.zeros((num_n, num_coeffs))
    coeffs_ppi = np.zeros((num_n, num_coeffs))

    for i, n in enumerate(n_values):
        print(f"  Running n={n}...")
        # Derive a unique seed per n so samples are independent across n values
        # but still deterministic given the job seed
        n_seed = args.seed * 10000 + int(n)
        exp, dsl, ppi = compute_one_n(X, Y, Y_hat, n, n_seed)
        coeffs_exp[i] = exp
        coeffs_dsl[i] = dsl
        coeffs_ppi[i] = ppi

    # Save this repetition's results to a .npz file.
    # After all 300 jobs finish, you load all 300 files and stack them
    # to compute sRMSE and standardized bias across repetitions.
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.results_path,
        theta_star=theta_star,       # shape (6,)       -- reference coefficients
        theta_llm=theta_llm,         # shape (6,)       -- LLM-only baseline
        n_values=n_values,           # shape (num_n,)   -- the actual n values used
        coeffs_exp=coeffs_exp,       # shape (num_n, 6) -- expert-only per n
        coeffs_dsl=coeffs_dsl,       # shape (num_n, 6) -- DSL per n
        coeffs_ppi=coeffs_ppi,       # shape (num_n, 6) -- PPI per n
    )
    print(f"Saved to {args.results_path}")