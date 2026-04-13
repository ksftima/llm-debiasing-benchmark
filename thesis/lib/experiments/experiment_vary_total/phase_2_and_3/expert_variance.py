"""
Experiment: Vary Total Dataset Size — Phase 2/3: Single Feature

Phase 2: low-variance feature (x2)
Phase 3: high-variance feature (x1)
θ = [β₀, β_feature] — intercept + one coefficient (2-vector).

Fix n (expert annotations), vary N (total samples) log-spaced from n to 1000.

For each repetition (controlled by --seed = SLURM array task ID):
    - Compute theta* on full dataset (reference)
    - Compute theta_llm on full dataset (LLM-only baseline)
    - For each N in log-spaced [n_expert, 1000]:
        - Subsample N rows from the full dataset
        - Select n_expert of those N rows as expert-labeled
        - Fit expert-only, DSL, PPI
    - Save all results to a .npz file
"""

import sys
sys.path.insert(0, "/code/original/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent / "experiment_vary_experts"))

import json
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from ppipp import fit_ppipp

CONFIG_PATH = Path(__file__).parent.parent.parent / "dataset_config.json"

N_MAX   = 997  # fixed to smallest annotated dataset
LAM_L2  = 0.01  # default — overridden at runtime by --lam argument


def get_feature(dataset: str, phase: str) -> str:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    key     = f"{phase}_variance_feature"
    feature = config.get(dataset, {}).get(key)
    if feature is None:
        raise ValueError(f"No {key} configured for dataset '{dataset}'.")
    return feature


def fit_logistic_x2(Y, x2):
    X   = x2.reshape(-1, 1)
    clf = LogisticRegression(penalty="l2", C=1.0/LAM_L2, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.array([clf.intercept_[0], clf.coef_[0, 0]])


def fit_logistic_x2_unreg(Y, x2):
    X   = x2.reshape(-1, 1)
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.array([clf.intercept_[0], clf.coef_[0, 0]])


def fit_ppi_x2(Y, Y_hat, x2, selected_mask):
    ones = np.ones(len(Y))
    X    = np.column_stack([ones, x2])

    X_lab      = X[selected_mask]
    Y_lab      = Y[selected_mask].astype(float)
    Yhat_lab   = Y_hat[selected_mask].astype(float)
    X_unlab    = X[~selected_mask]
    Yhat_unlab = Y_hat[~selected_mask].astype(float)

    def safe_log1pexp(z):
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

    def objective(theta):
        loss_unlab  = np.mean(-Yhat_unlab  * (X_unlab @ theta) + safe_log1pexp(X_unlab @ theta))
        loss_lab_y  = np.mean(-Y_lab       * (X_lab   @ theta) + safe_log1pexp(X_lab   @ theta))
        loss_lab_yh = np.mean(-Yhat_lab    * (X_lab   @ theta) + safe_log1pexp(X_lab   @ theta))
        l2 = LAM_L2 / 2.0 * np.sum(theta[1:] ** 2)
        return loss_unlab - loss_lab_yh + loss_lab_y + l2

    def gradient(theta):
        grad_unlab  = X_unlab.T @ (expit(X_unlab @ theta) - Yhat_unlab)  / len(Yhat_unlab)
        grad_lab_y  = X_lab.T   @ (expit(X_lab   @ theta) - Y_lab)       / len(Y_lab)
        grad_lab_yh = X_lab.T   @ (expit(X_lab   @ theta) - Yhat_lab)    / len(Y_lab)
        grad_l2     = LAM_L2 * np.concatenate([[0.0], theta[1:]])
        return grad_unlab - grad_lab_yh + grad_lab_y + grad_l2

    try:
        result = minimize(objective, np.zeros(2), jac=gradient, method="L-BFGS-B")
        if not result.success:
            print(f"    PPI optimisation did not converge: {result.message}")
        return result.x
    except Exception as e:
        print(f"    PPI failed: {e}")
        return np.array([np.nan, np.nan])


def fit_dsl_x2(Y, Y_hat, x2, selected_mask, feature: str):
    import rpy2.robjects as ro
    import rpy2.rinterface_lib.callbacks as rcb
    import logging
    rcb.logger.setLevel(logging.ERROR)

    Y_true_sel = Y.copy().astype(object)
    Y_true_sel[~selected_mask] = None

    data = pd.DataFrame({"Y": Y_true_sel, "Y_hat": Y_hat, feature: x2})

    with tempfile.TemporaryDirectory() as tmp:
        data_file  = Path(tmp) / "data.csv"
        coeff_file = Path(tmp) / "coeff.csv"
        data.to_csv(data_file, index=False)

        try:
            ro.r(f"""
                sink("/dev/null")
                suppressWarnings(library("dsl"))
                data <- read.csv("{data_file}")
                out <- suppressWarnings(dsl(
                    model         = "logit",
                    formula       = Y ~ {feature},
                    predicted_var = "Y",
                    prediction    = "Y_hat",
                    data          = data,
                    seed          = Sys.time(),
                    lambda        = {LAM_L2}
                ))
                write.csv(out$coefficients, "{coeff_file}", row.names=FALSE)
                sink()
            """)
            coeffs = np.array(pd.read_csv(coeff_file)).squeeze()
        except Exception as e:
            print(f"    DSL failed: {e}")
            return np.array([np.nan, np.nan])

    return np.atleast_1d(coeffs)


def compute_one_N(Y_full, Y_hat_full, x2_full, N, n_expert, seed, feature):
    """
    Subsample N rows, select n_expert as expert-labeled, fit all methods.
    """
    rng = np.random.default_rng(seed)

    idx   = rng.choice(len(Y_full), size=N, replace=False)
    Y     = Y_full[idx]
    Y_hat = Y_hat_full[idx]
    x2    = x2_full[idx]

    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[rng.choice(N, size=n_expert, replace=False)] = True

    beta_exp   = fit_logistic_x2(Y[selected_mask], x2[selected_mask])
    beta_dsl   = fit_dsl_x2(Y, Y_hat, x2, selected_mask, feature)
    beta_ppi   = fit_ppi_x2(Y, Y_hat, x2, selected_mask)
    beta_ppipp = fit_ppipp(Y, Y_hat, x2.reshape(-1, 1), selected_mask, LAM_L2)

    return beta_exp, beta_dsl, beta_ppi, beta_ppipp


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path)
    parser.add_argument("results_path",  type=Path)
    parser.add_argument("--seed",     type=int, required=True)
    parser.add_argument("--dataset",  type=str, required=True)
    parser.add_argument("--phase",    type=str, choices=["low", "high"], default="low")
    parser.add_argument("--n-expert", type=int, required=True,
        help="Fixed number of expert annotations (e.g. 50, 100, 200)")
    parser.add_argument("--lam", type=float, default=0.01,
        help="L2 regularization strength (default: 0.01)")
    args = parser.parse_args()

    LAM_L2 = args.lam

    feature = get_feature(args.dataset, args.phase)
    print(f"Seed: {args.seed} | CSV: {args.annotated_csv} | Feature: {feature} | n_expert: {args.n_expert}")

    data  = pd.read_csv(args.annotated_csv).sample(frac=1, random_state=42).reset_index(drop=True)
    Y_full     = data["y"].to_numpy().astype(float)
    Y_hat_full = data["y_hat"].to_numpy().astype(float)
    x2_full    = data[feature].to_numpy().astype(float)

    # Cap to N_MAX
    if len(Y_full) > N_MAX:
        rng0 = np.random.default_rng(0)
        idx0 = rng0.choice(len(Y_full), size=N_MAX, replace=False)
        Y_full     = Y_full[idx0]
        Y_hat_full = Y_hat_full[idx0]
        x2_full    = x2_full[idx0]

    # Reference on full (capped) dataset — unregularized to avoid bias in target
    theta_star = fit_logistic_x2_unreg(Y_full, x2_full)
    theta_llm  = fit_logistic_x2_unreg(Y_hat_full, x2_full)

    print(f"theta* [β₀, β_feat]: {theta_star}")
    print(f"theta_llm:           {theta_llm}")

    effective_N_max = min(N_MAX, len(Y_full))
    N_values = np.unique(
        np.round(np.logspace(
            np.log10(args.n_expert),
            np.log10(effective_N_max),
            num=10,
        )).astype(int)
    )
    N_values = N_values[N_values > args.n_expert]
    print(f"N values: {N_values.tolist()}")

    num_N        = len(N_values)
    thetas_exp   = np.zeros((num_N, 2))
    thetas_dsl   = np.zeros((num_N, 2))
    thetas_ppi   = np.zeros((num_N, 2))
    thetas_ppipp = np.zeros((num_N, 2))

    for i, N in enumerate(N_values):
        N_seed = args.seed * 100000 + int(N)
        exp, dsl, ppi, ppipp = compute_one_N(Y_full, Y_hat_full, x2_full, N, args.n_expert, N_seed, feature)
        thetas_exp[i]   = exp
        thetas_dsl[i]   = dsl
        thetas_ppi[i]   = ppi
        thetas_ppipp[i] = ppipp
        print(f"  N={N:4d} | exp={exp} | dsl={dsl} | ppi={ppi} | ppipp={ppipp}")

    phase_label = "low_variance" if args.phase == "low" else "high_variance"

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.results_path,
        theta_star   = theta_star,
        theta_llm    = theta_llm,
        N_values     = N_values,
        n_expert     = np.array([args.n_expert]),
        thetas_exp   = thetas_exp,
        thetas_dsl   = thetas_dsl,
        thetas_ppi   = thetas_ppi,
        thetas_ppipp = thetas_ppipp,
        phase        = np.array([phase_label]),
    )
    print(f"Saved to {args.results_path}")
