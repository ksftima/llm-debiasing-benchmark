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

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from ppi_py import ppi_logistic_pointestimate


def fit_logistic_x2(Y, x2):
    """
    MLE for logistic regression with x2 as the single feature.
    Returns [β₀, β₂] as a 2-element array.
    penalty=None gives the unregularised MLE.
    """
    X   = x2.reshape(-1, 1)
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.array([clf.intercept_[0], clf.coef_[0, 0]])


def fit_ppi_x2(Y, Y_hat, x2, selected_mask):
    """
    PPI estimate for logistic regression with x2 feature.

    X must include the intercept column (ppi_py convention):
        X = [[1, x2_0], [1, x2_1], ...]  shape (N, 2)
    """
    N    = len(Y)
    ones = np.ones(N)
    X    = np.column_stack([ones, x2])  # shape (N, 2)

    return ppi_logistic_pointestimate(
        X              = X[selected_mask],
        Y              = Y[selected_mask],
        Yhat           = Y_hat[selected_mask],
        X_unlabeled    = X[~selected_mask],
        Yhat_unlabeled = Y_hat[~selected_mask],
    )


def fit_dsl_x2(Y, Y_hat, x2, selected_mask):
    """
    DSL estimate for logistic regression with x2 feature.
    formula = Y ~ x2
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
        "x2":    x2,
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
                    formula       = Y ~ x2,
                    predicted_var = "Y",
                    prediction    = "Y_hat",
                    data          = data,
                    seed          = Sys.time()
                ))
                write.csv(out$coefficients, "{coeff_file}", row.names=FALSE)
                sink()
            """)
            coeffs = np.array(pd.read_csv(coeff_file)).squeeze()
        except Exception as e:
            print(f"    DSL failed (separation): {e}")
            return np.array([np.nan, np.nan])

    return np.atleast_1d(coeffs)  # [β₀, β₂]


def compute_one_n(Y, Y_hat, x2, n, seed):
    """
    For one expert sample size n: select n rows, compute θ for each method.
    Returns three 2-element arrays [β₀, β₂].
    """
    rng = np.random.default_rng(seed)
    selected_mask = np.zeros(len(Y), dtype=bool)
    selected_mask[rng.choice(len(Y), size=n, replace=False)] = True

    beta_exp = fit_logistic_x2(Y[selected_mask], x2[selected_mask])
    beta_dsl = fit_dsl_x2(Y, Y_hat, x2, selected_mask)
    beta_ppi = fit_ppi_x2(Y, Y_hat, x2, selected_mask)

    return beta_exp, beta_dsl, beta_ppi


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path,
        help="Path to annotated CSV, e.g. cuad_llama_annotated.csv")
    parser.add_argument("results_path", type=Path,
        help="Where to save the .npz result for this repetition")
    parser.add_argument("--seed", type=int, required=True,
        help="Random seed = SLURM array task ID")
    args = parser.parse_args()

    print(f"Seed: {args.seed} | CSV: {args.annotated_csv}")

    data  = pd.read_csv(args.annotated_csv).sample(frac=1, random_state=42).reset_index(drop=True)
    Y     = data["y"].to_numpy().astype(float)
    Y_hat = data["y_hat"].to_numpy().astype(float)
    x2    = data["x2"].to_numpy().astype(float)

    # Reference: fit on ALL expert labels
    theta_star = fit_logistic_x2(Y, x2)
    theta_llm  = fit_logistic_x2(Y_hat, x2)

    print(f"theta* [β₀, β₂]: {theta_star}")
    print(f"theta_llm:        {theta_llm}")

    # Log-spaced n values: ~20, 26, 33, 43, 56, 72, 93, 120, 155, 200
    n_values = np.unique(
        np.round(np.logspace(np.log10(20), np.log10(200), num=10)).astype(int)
    )
    print(f"n values: {n_values.tolist()}")

    num_n      = len(n_values)
    thetas_exp = np.zeros((num_n, 2))
    thetas_dsl = np.zeros((num_n, 2))
    thetas_ppi = np.zeros((num_n, 2))

    for i, n in enumerate(n_values):
        n_seed = args.seed * 10000 + int(n)
        exp, dsl, ppi = compute_one_n(Y, Y_hat, x2, n, n_seed)
        thetas_exp[i] = exp
        thetas_dsl[i] = dsl
        thetas_ppi[i] = ppi
        print(f"  n={n:3d} | exp={exp} | dsl={dsl} | ppi={ppi}")

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.results_path,
        theta_star  = theta_star,   # shape (2,)  — reference [β₀*, β₂*]
        theta_llm   = theta_llm,    # shape (2,)  — LLM-only [β₀_llm, β₂_llm]
        n_values    = n_values,     # shape (num_n,)
        thetas_exp  = thetas_exp,   # shape (num_n, 2)
        thetas_dsl  = thetas_dsl,   # shape (num_n, 2)
        thetas_ppi  = thetas_ppi,   # shape (num_n, 2)
    )
    print(f"Saved to {args.results_path}")