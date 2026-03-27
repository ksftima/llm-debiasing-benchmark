"""
Experiment: Vary Expert Sample Size — Phases 2 & 3: Single Feature (PPI++ only)

Computes only the PPI++ estimate alongside reference θ* and θ_llm.
DSL / PPI / expert_only results already exist from prior runs.
Uses identical seeds (n_seed = seed * 10000 + n) for fair comparison.

θ = [β₀, β_feature] — 2-vector.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, "/code/original/lib")
sys.path.insert(0, str(Path(__file__).parent.parent))  # for ppipp.py

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression

from ppipp import fit_ppipp

CONFIG_PATH = Path(__file__).parent.parent.parent / "dataset_config.json"

LAM_L2 = 0.01  # overridden by --lam


def get_feature(dataset: str, phase: str) -> str:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    key     = f"{phase}_variance_feature"
    feature = config.get(dataset, {}).get(key)
    if feature is None:
        raise ValueError(f"No {key} configured for dataset '{dataset}'. Update dataset_config.json.")
    return feature


def fit_logistic_x2_unregularized(Y, x2):
    X   = x2.reshape(-1, 1)
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.array([clf.intercept_[0], clf.coef_[0, 0]])


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path)
    parser.add_argument("results_path",  type=Path)
    parser.add_argument("--seed",        type=int,   required=True)
    parser.add_argument("--dataset",     type=str,   required=True)
    parser.add_argument("--phase",       type=str,   choices=["low", "high"], default="low")
    parser.add_argument("--lam",         type=float, default=0.01)
    args = parser.parse_args()

    LAM_L2  = args.lam
    feature = get_feature(args.dataset, args.phase)
    print(f"Seed: {args.seed} | CSV: {args.annotated_csv} | Feature: {feature} | lam: {LAM_L2}")

    data  = pd.read_csv(args.annotated_csv).sample(frac=1, random_state=42).reset_index(drop=True)
    Y     = data["y"].to_numpy().astype(float)
    Y_hat = data["y_hat"].to_numpy().astype(float)
    x2    = data[feature].to_numpy().astype(float)

    theta_star = fit_logistic_x2_unregularized(Y, x2)
    theta_llm  = fit_logistic_x2_unregularized(Y_hat, x2)
    print(f"theta* [β₀, β_feature]: {theta_star}")
    print(f"theta_llm:              {theta_llm}")

    n_values = np.unique(
        np.round(np.logspace(np.log10(20), np.log10(200), num=10)).astype(int)
    )
    print(f"n values: {n_values.tolist()}")

    num_n        = len(n_values)
    thetas_ppipp = np.zeros((num_n, 2))

    X_feat = x2.reshape(-1, 1)  # shape (N, 1) — intercept added inside fit_ppipp

    for i, n in enumerate(n_values):
        n_seed = args.seed * 10000 + int(n)
        rng    = np.random.default_rng(n_seed)
        selected_mask = np.zeros(len(Y), dtype=bool)
        selected_mask[rng.choice(len(Y), size=n, replace=False)] = True

        print(f"  n={n:3d}")
        ppipp = fit_ppipp(Y, Y_hat, X_feat, selected_mask, LAM_L2)
        thetas_ppipp[i] = ppipp
        print(f"         ppipp={ppipp}")

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.results_path,
        theta_star   = theta_star,    # shape (2,)
        theta_llm    = theta_llm,     # shape (2,)
        n_values     = n_values,      # shape (num_n,)
        thetas_ppipp = thetas_ppipp,  # shape (num_n, 2)
    )
    print(f"Saved to {args.results_path}")
