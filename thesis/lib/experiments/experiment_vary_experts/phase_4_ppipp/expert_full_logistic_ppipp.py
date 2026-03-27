"""
Experiment: Vary Expert Sample Size — Phase 4: Full Logistic Regression (PPI++ only)

Computes only the PPI++ estimate alongside reference θ* and θ_llm.
DSL / PPI / expert_only results already exist from prior runs.
Uses identical seeds (n_seed = seed * 10000 + n) for fair comparison.

θ = [β₀, β₁, β₂, β₃, β₄, β₅] — intercept + 5 coefficients (6-vector).
"""

import sys
from pathlib import Path
sys.path.insert(0, "/code/original/lib")
sys.path.insert(0, str(Path(__file__).parent.parent))  # for ppipp.py

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression

from ppipp import fit_ppipp

FEATURES = ["x1", "x2", "x3", "x4", "x5"]
N_COEF   = len(FEATURES) + 1  # 6

LAM_L2 = 0.01  # overridden by --lam


def fit_logistic_full_unregularized(Y, X):
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.concatenate([[clf.intercept_[0]], clf.coef_[0]])


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path)
    parser.add_argument("results_path",  type=Path)
    parser.add_argument("--seed",        type=int,   required=True)
    parser.add_argument("--lam",         type=float, default=0.01)
    parser.add_argument("--n-select",    type=int,   nargs="+", default=None)
    args = parser.parse_args()

    LAM_L2 = args.lam
    print(f"Seed: {args.seed} | CSV: {args.annotated_csv} | lam: {LAM_L2}")

    data  = pd.read_csv(args.annotated_csv).sample(frac=1, random_state=42).reset_index(drop=True)
    Y     = data["y"].to_numpy().astype(float)
    Y_hat = data["y_hat"].to_numpy().astype(float)
    X     = data[FEATURES].to_numpy().astype(float)

    theta_star = fit_logistic_full_unregularized(Y, X)
    theta_llm  = fit_logistic_full_unregularized(Y_hat, X)
    print(f"theta*    {theta_star}")
    print(f"theta_llm {theta_llm}")

    if args.n_select is not None:
        n_values = np.array(sorted(args.n_select))
    else:
        n_values = np.unique(
            np.round(np.logspace(np.log10(20), np.log10(200), num=10)).astype(int)
        )
    print(f"n values: {n_values.tolist()}")

    num_n       = len(n_values)
    thetas_ppipp = np.zeros((num_n, N_COEF))

    for i, n in enumerate(n_values):
        n_seed = args.seed * 10000 + int(n)
        rng    = np.random.default_rng(n_seed)
        selected_mask = np.zeros(len(Y), dtype=bool)
        selected_mask[rng.choice(len(Y), size=n, replace=False)] = True

        print(f"  n={n:3d}")
        ppipp = fit_ppipp(Y, Y_hat, X, selected_mask, LAM_L2)
        thetas_ppipp[i] = ppipp
        print(f"         ppipp={ppipp}")

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.results_path,
        theta_star   = theta_star,    # shape (6,)
        theta_llm    = theta_llm,     # shape (6,)
        n_values     = n_values,      # shape (num_n,)
        thetas_ppipp = thetas_ppipp,  # shape (num_n, 6)
    )
    print(f"Saved to {args.results_path}")
