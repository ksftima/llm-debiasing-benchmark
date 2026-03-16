"""
Experiment: Vary Expert Sample Size — Phase 1: Class Prevalence

No features (X) — intercept-only logistic regression.
The single parameter θ = β₀ is the log-odds of P(Y=1).

Why log-odds and not raw probability?
    Keeping θ in log-odds space makes this phase consistent with phases 2-4,
    where θ is always the coefficient vector of a logistic regression.
    sRMSE is computed on β₀ directly.

For each repetition (controlled by --seed = SLURM array task ID):
    - Compute theta*   = logit(mean(ALL y))       reference, uses all expert labels
    - Compute theta_llm = logit(mean(ALL y_hat))  LLM-only biased baseline
    - For each n in log-spaced [20, 200]:
        - Randomly select n rows as "expert-labeled" (boolean mask)
        - theta_exp = logit(mean(y[selected]))             expert-only
        - theta_dsl = R dsl(model="logit", formula=Y~1)    DSL intercept-only
        - theta_ppi = ppi_logistic_pointestimate(X=ones)   PPI intercept-only
    - Save all results to a .npz file
"""

import sys
sys.path.insert(0, "/code/original/lib")  # find fitting.py inside the container

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from scipy.special import logit  # logit(p) = log(p / (1-p)), the log-odds transform
from ppi_py import ppi_logistic_pointestimate


def fit_logit_intercept_only(Y):
    """
    MLE for intercept-only logistic regression = logit of the sample mean.
    Returns a 1-element array [β₀] for consistency with later phases
    where θ is always an array of coefficients.
    """
    p = np.clip(Y.mean(), 1e-7, 1 - 1e-7)  # clip to avoid log(0) or log(inf)
    return np.array([logit(p)])


def fit_ppi_intercept_only(Y, Y_hat, selected_mask):
    """
    PPI estimate for intercept-only logistic regression using ppi_py.

    ppi_logistic_pointestimate expects X to already include the intercept column,
    which is why the supervisor's logit_fit_ppi prepends a column of ones to X.
    Here X IS the column of ones — that is the only "feature".

    selected_mask : boolean array, True = this row is expert-labeled
    """
    N = len(Y)
    ones = np.ones((N, 1))  # shape (N, 1) — intercept column only

    return ppi_logistic_pointestimate(
        X              = ones[selected_mask],    # labeled feature matrix
        Y              = Y[selected_mask],        # expert labels for labeled rows
        Yhat           = Y_hat[selected_mask],    # LLM labels for labeled rows
        X_unlabeled    = ones[~selected_mask],    # unlabeled feature matrix
        Yhat_unlabeled = Y_hat[~selected_mask],   # LLM labels for unlabeled rows
    )


def fit_dsl_intercept_only(Y, Y_hat, selected_mask):
    """
    DSL estimate for intercept-only logistic regression using R's dsl package.

    formula = Y ~ 1 means intercept-only in R formula syntax.
    Expert labels for non-selected rows are set to None (missing) —
    that is how DSL knows which rows are expert-labeled vs LLM-only.
    """
    import rpy2.robjects as ro

    # DSL needs Y to be missing (None) for rows that are NOT expert-labeled
    Y_true_sel = Y.copy().astype(object)
    Y_true_sel[~selected_mask] = None

    data = pd.DataFrame({
        "Y":     Y_true_sel,
        "Y_hat": Y_hat,
    })

    with tempfile.TemporaryDirectory() as tmp:
        data_file  = Path(tmp) / "data.csv"
        coeff_file = Path(tmp) / "coeff.csv"
        data.to_csv(data_file, index=False)

        ro.r(f"""
            sink("/dev/null")
            library("dsl")
            data <- read.csv("{data_file}")
            out <- dsl(
                model        = "logit",
                formula      = Y ~ 1,
                predicted_var = "Y",
                prediction   = "Y_hat",
                data         = data,
                seed         = Sys.time()
            )
            write.csv(out$coefficients, "{coeff_file}", row.names=FALSE)
            sink()
        """)

        coeffs = np.array(pd.read_csv(coeff_file)).squeeze()

    return np.atleast_1d(coeffs)  # always return 1-element array [β₀]


def compute_one_n(Y, Y_hat, n, seed):
    """
    For one expert sample size n:
      - randomly select n rows as expert-labeled
      - compute θ for expert-only, DSL, PPI
    Returns three 1-element arrays (log-odds).
    """
    rng = np.random.default_rng(seed)

    # Boolean mask is needed for PPI (~selected_mask = unlabeled rows)
    selected_mask = np.zeros(len(Y), dtype=bool)
    selected_mask[rng.choice(len(Y), size=n, replace=False)] = True

    beta_exp = fit_logit_intercept_only(Y[selected_mask])
    beta_dsl = fit_dsl_intercept_only(Y, Y_hat, selected_mask)
    beta_ppi = fit_ppi_intercept_only(Y, Y_hat, selected_mask)

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

    data  = pd.read_csv(args.annotated_csv)
    Y     = data["y"].to_numpy().astype(float)
    Y_hat = data["y_hat"].to_numpy().astype(float)

    # Reference: log-odds from ALL expert labels — the target every method aims for
    theta_star = fit_logit_intercept_only(Y)

    # LLM-only baseline: log-odds from ALL LLM labels — biased but uses full data
    theta_llm = fit_logit_intercept_only(Y_hat)

    # Log-spaced n values: ~20, 26, 33, 43, 56, 72, 93, 120, 155, 200
    n_values = np.unique(
        np.round(np.logspace(np.log10(20), np.log10(200), num=10)).astype(int)
    )
    print(f"n values: {n_values.tolist()}")

    num_n = len(n_values)
    thetas_exp = np.zeros(num_n)
    thetas_dsl = np.zeros(num_n)
    thetas_ppi = np.zeros(num_n)

    for i, n in enumerate(n_values):
        n_seed = args.seed * 10000 + int(n)
        exp, dsl, ppi = compute_one_n(Y, Y_hat, n, n_seed)
        thetas_exp[i] = float(exp)
        thetas_dsl[i] = float(dsl)
        thetas_ppi[i] = float(ppi)
        print(f"  n={n:3d} | exp={float(exp):.4f} | dsl={float(dsl):.4f} | ppi={float(ppi):.4f}")

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.results_path,
        theta_star  = theta_star,   # shape (1,) — reference log-odds
        theta_llm   = theta_llm,    # shape (1,) — LLM-only log-odds
        n_values    = n_values,     # shape (num_n,)
        thetas_exp  = thetas_exp,   # shape (num_n,) — expert-only per n
        thetas_dsl  = thetas_dsl,   # shape (num_n,) — DSL per n
        thetas_ppi  = thetas_ppi,   # shape (num_n,) — PPI per n
    )
    print(f"Saved to {args.results_path}")
