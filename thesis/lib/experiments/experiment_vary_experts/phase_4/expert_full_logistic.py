"""
Experiment: Vary Expert Sample Size — Phase 4: Full Logistic Regression

Logistic regression with all 5 features (x1, x2, x3, x4, x5).
θ = [β₀, β₁, β₂, β₃, β₄, β₅] — intercept + 5 coefficients (6-vector).

For each repetition (controlled by --seed = SLURM array task ID):
    - Compute theta*    = logistic_fit(ALL y,     X)  reference
    - Compute theta_llm = logistic_fit(ALL y_hat, X)  LLM-only biased baseline
    - For each n in log-spaced [20, 200]:
        - Randomly select n rows as "expert-labeled" (boolean mask)
        - theta_exp = logistic_fit(y[selected], X[selected])
        - theta_dsl = R dsl(model="logit", formula=Y ~ x1+x2+x3+x4+x5)
        - theta_ppi = custom PPI with L2 regularization
    - Save all results to a .npz file
"""

import sys
sys.path.insert(0, "/code/original/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from ppipp import fit_ppipp

FEATURES = ["x1", "x2", "x3", "x4", "x5"]
N_COEF   = len(FEATURES) + 1  # intercept + 5 features = 6

LAM_L2 = 0.01  # default — overridden at runtime by --lam argument


def fit_logistic_full(Y, X):
    """
    L2-regularised full logistic regression with all 5 features.
    Returns [β₀, β₁, β₂, β₃, β₄, β₅] as a 6-element array.
    C = 1/LAM_L2 (sklearn convention).
    """
    clf = LogisticRegression(penalty="l2", C=1.0/LAM_L2, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.concatenate([[clf.intercept_[0]], clf.coef_[0]])


def fit_logistic_full_unregularized(Y, X):
    """
    Unregularized full logistic regression with all 5 features.
    Used only for θ* and θ_llm (population-level reference estimates).
    Returns [β₀, β₁, β₂, β₃, β₄, β₅] as a 6-element array.
    """
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.concatenate([[clf.intercept_[0]], clf.coef_[0]])


def fit_ppi_full(Y, Y_hat, X, selected_mask):
    """
    PPI logistic regression with L2 regularization over all features.

    L(θ) = mean_unlabeled(ℓ(ŷ, Xθ))
          - mean_labeled(ℓ(ŷ, Xθ))
          + mean_labeled(ℓ(y, Xθ))
          + LAM_L2/2 * ||θ[1:]||²
    """
    ones = np.ones(len(Y))
    X_aug = np.column_stack([ones, X])  # shape (N, 6)

    X_lab      = X_aug[selected_mask]
    Y_lab      = Y[selected_mask].astype(float)
    Yhat_lab   = Y_hat[selected_mask].astype(float)
    X_unlab    = X_aug[~selected_mask]
    Yhat_unlab = Y_hat[~selected_mask].astype(float)

    def safe_log1pexp(z):
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

    def objective(theta):
        loss_unlab  = np.mean(-Yhat_unlab * (X_unlab @ theta) + safe_log1pexp(X_unlab @ theta))
        loss_lab_y  = np.mean(-Y_lab      * (X_lab   @ theta) + safe_log1pexp(X_lab   @ theta))
        loss_lab_yh = np.mean(-Yhat_lab   * (X_lab   @ theta) + safe_log1pexp(X_lab   @ theta))
        l2 = LAM_L2 / 2.0 * np.sum(theta[1:] ** 2)
        return loss_unlab - loss_lab_yh + loss_lab_y + l2

    def gradient(theta):
        grad_unlab  = X_unlab.T @ (expit(X_unlab @ theta) - Yhat_unlab) / len(Yhat_unlab)
        grad_lab_y  = X_lab.T   @ (expit(X_lab   @ theta) - Y_lab)      / len(Y_lab)
        grad_lab_yh = X_lab.T   @ (expit(X_lab   @ theta) - Yhat_lab)   / len(Y_lab)
        grad_l2 = LAM_L2 * np.concatenate([[0.0], theta[1:]])
        return grad_unlab - grad_lab_yh + grad_lab_y + grad_l2

    try:
        result = minimize(objective, np.zeros(N_COEF), jac=gradient, method="L-BFGS-B")
        if not result.success:
            print(f"    PPI optimisation did not converge: {result.message}")
        return result.x
    except Exception as e:
        print(f"    PPI failed: {e}")
        return np.full(N_COEF, np.nan)


def fit_dsl_full(Y, Y_hat, X_df, selected_mask):
    """
    DSL estimate for full logistic regression with all 5 features.
    formula = Y ~ x1 + x2 + x3 + x4 + x5
    """
    import rpy2.robjects as ro
    import rpy2.rinterface_lib.callbacks as rcb
    import logging
    rcb.logger.setLevel(logging.ERROR)

    Y_true_sel = Y.copy().astype(object)
    Y_true_sel[~selected_mask] = None

    data = X_df.copy()
    data["Y"]     = Y_true_sel
    data["Y_hat"] = Y_hat

    with tempfile.TemporaryDirectory() as tmp:
        data_file  = Path(tmp) / "data.csv"
        coeff_file = Path(tmp) / "coeff.csv"
        data.to_csv(data_file, index=False)

        formula = "Y ~ " + " + ".join(FEATURES)

        try:
            ro.r(f"""
                sink("/dev/null")
                data <- read.csv("{data_file}")
                out <- suppressWarnings(dsl(
                    model         = "logit",
                    formula       = {formula},
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
            return np.full(N_COEF, np.nan)

    return np.atleast_1d(coeffs)


def compute_one_n(packed_args):
    """
    For one expert sample size n: select n rows, compute θ for each method.
    Returns three 6-element arrays.
    """
    global LAM_L2
    Y, Y_hat, X, X_df, n, seed, lam_l2 = packed_args
    LAM_L2 = lam_l2
    rng = np.random.default_rng(seed)
    selected_mask = np.zeros(len(Y), dtype=bool)
    selected_mask[rng.choice(len(Y), size=n, replace=False)] = True

    beta_exp   = fit_logistic_full(Y[selected_mask], X[selected_mask])
    beta_dsl   = fit_dsl_full(Y, Y_hat, X_df, selected_mask)
    beta_ppi   = fit_ppi_full(Y, Y_hat, X, selected_mask)
    beta_ppipp = fit_ppipp(Y, Y_hat, X, selected_mask, LAM_L2)

    return beta_exp, beta_dsl, beta_ppi, beta_ppipp


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path,
        help="Path to annotated CSV, e.g. cuad_llama_annotated.csv")
    parser.add_argument("results_path", type=Path,
        help="Where to save the .npz result for this repetition")
    parser.add_argument("--seed", type=int, required=True,
        help="Random seed = SLURM array task ID")
    parser.add_argument("--lam", type=float, default=0.01,
        help="L2 regularization strength (default: 0.01)")
    parser.add_argument("--n-select", type=int, nargs="+", default=None,
        help="Run only at these n values instead of the full log-spaced grid, e.g. --n-select 26 72 200")
    args = parser.parse_args()

    LAM_L2 = args.lam

    print(f"Seed: {args.seed} | CSV: {args.annotated_csv}")

    data  = pd.read_csv(args.annotated_csv).sample(frac=1, random_state=42).reset_index(drop=True)
    Y     = data["y"].to_numpy().astype(float)
    Y_hat = data["y_hat"].to_numpy().astype(float)
    X     = data[FEATURES].to_numpy().astype(float)
    X_df  = data[FEATURES].copy()

    # Reference: unregularized fit on ALL expert labels (population-level ground truth)
    theta_star = fit_logistic_full_unregularized(Y, X)
    theta_llm  = fit_logistic_full_unregularized(Y_hat, X)

    print(f"theta* {theta_star}")
    print(f"theta_llm {theta_llm}")

    if args.n_select is not None:
        n_values = np.array(sorted(args.n_select))
    else:
        n_values = np.unique(
            np.round(np.logspace(np.log10(20), np.log10(200), num=10)).astype(int)
        )
    print(f"n values: {n_values.tolist()}")

    num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {num_cores} cores")

    num_n        = len(n_values)
    thetas_exp   = np.zeros((num_n, N_COEF))
    thetas_dsl   = np.zeros((num_n, N_COEF))
    thetas_ppi   = np.zeros((num_n, N_COEF))
    thetas_ppipp = np.zeros((num_n, N_COEF))

    worker_args = [
        (Y, Y_hat, X, X_df, int(n), args.seed * 10000 + int(n), LAM_L2)
        for n in n_values
    ]
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        all_results = list(executor.map(compute_one_n, worker_args))

    for i, (exp, dsl, ppi, ppipp) in enumerate(all_results):
        thetas_exp[i]   = exp
        thetas_dsl[i]   = dsl
        thetas_ppi[i]   = ppi
        thetas_ppipp[i] = ppipp
        n = n_values[i]
        print(f"  n={n:3d} | exp={exp} | dsl={dsl} | ppi={ppi} | ppipp={ppipp}")

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.results_path,
        theta_star   = theta_star,    # shape (6,)
        theta_llm    = theta_llm,     # shape (6,)
        n_values     = n_values,      # shape (num_n,)
        thetas_exp   = thetas_exp,    # shape (num_n, 6)
        thetas_dsl   = thetas_dsl,    # shape (num_n, 6)
        thetas_ppi   = thetas_ppi,    # shape (num_n, 6)
        thetas_ppipp = thetas_ppipp,  # shape (num_n, 6)
    )
    print(f"Saved to {args.results_path}")
