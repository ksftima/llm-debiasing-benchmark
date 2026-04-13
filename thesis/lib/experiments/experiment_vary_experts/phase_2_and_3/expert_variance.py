"""
Experiment: Vary Expert Sample Size — Phase 2: Low-Variance Feature

Logistic regression with a single low-variance feature (x2).
θ = [β₀, β₂] — intercept + x2 coefficient (2-vector).

Why a 2-vector?
    Keeping θ as a coefficient array makes all phases consistent.
    Phase 1 had θ = [β₀] (intercept-only).
    Phase 2 adds x2: θ = [β₀, β₂].
    sRMSE is computed per coefficient and via Euclidean norm.

For each repetition (controlled by --seed = SLURM array task ID):
    - Compute theta*   = logistic_fit(ALL y,     x2)  reference
    - Compute theta_llm = logistic_fit(ALL y_hat, x2)  LLM-only biased baseline
    - For each n in log-spaced [20, 200]:
        - Randomly select n rows as "expert-labeled" (boolean mask)
        - theta_exp = logistic_fit(y[selected],  x2[selected])
        - theta_dsl = R dsl(model="logit", formula=Y ~ x2)
        - theta_ppi = ppi_logistic_pointestimate(X=[[1, x2_i]])
    - Save all results to a .npz file
"""

import sys
sys.path.insert(0, "/code/original/lib")  # find fitting.py inside the container
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

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


def get_feature(dataset: str, phase: str) -> str:
    """Load low/high variance feature name from dataset_config.json."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    key = f"{phase}_variance_feature"
    feature = config.get(dataset, {}).get(key)
    if feature is None:
        raise ValueError(f"No {key} configured for dataset '{dataset}'. Update dataset_config.json.")
    return feature


LAM_L2 = 0.01  # default — overridden at runtime by --lam argument


def fit_logistic_x2(Y, x2):
    """
    L2-regularised logistic regression with a single feature.
    Returns [β₀, β₂] as a 2-element array.
    C = 1/LAM_L2 (sklearn convention).
    """
    X   = x2.reshape(-1, 1)
    clf = LogisticRegression(penalty="l2", C=1.0/LAM_L2, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.array([clf.intercept_[0], clf.coef_[0, 0]])


def fit_logistic_x2_unregularized(Y, x2):
    """
    Unregularized logistic regression with a single feature.
    Used only for θ* and θ_llm (population-level reference estimates).
    Returns [β₀, β₂] as a 2-element array.
    """
    X   = x2.reshape(-1, 1)
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.array([clf.intercept_[0], clf.coef_[0, 0]])


def fit_ppi_x2(Y, Y_hat, x2, selected_mask):
    """
    PPI logistic regression with L2 regularization (lam_l2 = LAM_L2).

    Implements the PPI objective directly using scipy:
        L(θ) = mean_unlabeled(ℓ(ŷ, Xθ))
             - mean_labeled(ℓ(ŷ, Xθ))
             + mean_labeled(ℓ(y, Xθ))
             + LAM_L2/2 * ||θ[1:]||²
    where ℓ(y, Xθ) = -y*Xθ + log(1 + exp(Xθ)) is the logistic loss.
    L2 penalty is applied to all coefficients except the intercept (index 0).
    """
    ones = np.ones(len(Y))
    X    = np.column_stack([ones, x2])  # shape (N, 2)

    X_lab     = X[selected_mask]
    Y_lab     = Y[selected_mask].astype(float)
    Yhat_lab  = Y_hat[selected_mask].astype(float)
    X_unlab   = X[~selected_mask]
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
        grad_l2 = LAM_L2 * np.concatenate([[0.0], theta[1:]])
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
    """
    DSL estimate for logistic regression with a single feature.
    formula = Y ~ <feature>
    Expert labels for non-selected rows are set to None (missing).
    """
    import rpy2.robjects as ro
    import rpy2.rinterface_lib.callbacks as rcb
    import logging
    rcb.logger.setLevel(logging.ERROR)

    Y_true_sel = Y.copy().astype(object)
    Y_true_sel[~selected_mask] = None

    data = pd.DataFrame({
        "Y":     Y_true_sel,
        "Y_hat": Y_hat,
        feature: x2,
    })

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
            print(f"    DSL failed (separation): {e}")
            return np.array([np.nan, np.nan])

    return np.atleast_1d(coeffs)


def compute_one_n(Y, Y_hat, x2, n, seed, feature: str):
    """
    For one expert sample size n: select n rows, compute θ for each method.
    Returns three 2-element arrays [β₀, β_feature].
    """
    rng = np.random.default_rng(seed)
    selected_mask = np.zeros(len(Y), dtype=bool)
    selected_mask[rng.choice(len(Y), size=n, replace=False)] = True

    beta_exp   = fit_logistic_x2(Y[selected_mask], x2[selected_mask])
    beta_dsl   = fit_dsl_x2(Y, Y_hat, x2, selected_mask, feature)
    beta_ppi   = fit_ppi_x2(Y, Y_hat, x2, selected_mask)
    beta_ppipp = fit_ppipp(Y, Y_hat, x2.reshape(-1, 1), selected_mask, LAM_L2)

    return beta_exp, beta_dsl, beta_ppi, beta_ppipp


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path,
        help="Path to annotated CSV, e.g. cuad_llama_annotated.csv")
    parser.add_argument("results_path", type=Path,
        help="Where to save the .npz result for this repetition")
    parser.add_argument("--seed",    type=int, required=True,
        help="Random seed = SLURM array task ID")
    parser.add_argument("--dataset", type=str, required=True,
        help="Dataset name, e.g. cuad — used to look up feature from dataset_config.json")
    parser.add_argument("--phase", type=str, choices=["low", "high"], default="low",
        help="'low' for Phase 2 (low-variance feature), 'high' for Phase 3 (high-variance feature)")
    parser.add_argument("--lam", type=float, default=0.01,
        help="L2 regularization strength (default: 0.01)")
    args = parser.parse_args()

    LAM_L2 = args.lam

    feature = get_feature(args.dataset, args.phase)
    print(f"Seed: {args.seed} | CSV: {args.annotated_csv} | Feature: {feature}")

    data  = pd.read_csv(args.annotated_csv)
    if len(data) > 997:
        data = data.sample(n=997, random_state=42).reset_index(drop=True)
    Y     = data["y"].to_numpy().astype(float)
    Y_hat = data["y_hat"].to_numpy().astype(float)
    x2    = data[feature].to_numpy().astype(float)

    # Reference: unregularized fit on ALL expert labels (population-level ground truth)
    theta_star = fit_logistic_x2_unregularized(Y, x2)
    theta_llm  = fit_logistic_x2_unregularized(Y_hat, x2)

    print(f"theta* [β₀, β₂]: {theta_star}")
    print(f"theta_llm:        {theta_llm}")

    # Log-spaced n values: ~20, 26, 33, 43, 56, 72, 93, 120, 155, 200
    n_values = np.unique(
        np.round(np.logspace(np.log10(20), np.log10(200), num=10)).astype(int)
    )
    print(f"n values: {n_values.tolist()}")

    num_n        = len(n_values)
    thetas_exp   = np.zeros((num_n, 2))
    thetas_dsl   = np.zeros((num_n, 2))
    thetas_ppi   = np.zeros((num_n, 2))
    thetas_ppipp = np.zeros((num_n, 2))

    for i, n in enumerate(n_values):
        n_seed = args.seed * 10000 + int(n)
        exp, dsl, ppi, ppipp = compute_one_n(Y, Y_hat, x2, n, n_seed, feature)
        thetas_exp[i]   = exp
        thetas_dsl[i]   = dsl
        thetas_ppi[i]   = ppi
        thetas_ppipp[i] = ppipp
        print(f"  n={n:3d} | exp={exp} | dsl={dsl} | ppi={ppi} | ppipp={ppipp}")

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.results_path,
        theta_star   = theta_star,    # shape (2,)
        theta_llm    = theta_llm,     # shape (2,)
        n_values     = n_values,      # shape (num_n,)
        thetas_exp   = thetas_exp,    # shape (num_n, 2)
        thetas_dsl   = thetas_dsl,    # shape (num_n, 2)
        thetas_ppi   = thetas_ppi,    # shape (num_n, 2)
        thetas_ppipp = thetas_ppipp,  # shape (num_n, 2)
    )
    print(f"Saved to {args.results_path}")