"""
Experiment: Vary Expert Sample Size — Linear Probability Model (LPM)

Replaces logistic regression with OLS (linear probability model) across all methods:
    - expert_only : OLS on labeled data only
    - ppi         : closed-form PPI correction with squared loss
    - ppipp       : PPI++ with λ* estimated via linear-model Hessian
    - dsl         : R dsl() with model="linear"

θ = [β₀, β_feature] — intercept + single feature coefficient (2-vector).
θ* is the OLS fit on ALL expert labels (population-level ground truth).

Supports low- and high-variance features via --phase and dataset_config.json.

No regularization — OLS is the natural unregularized estimator and does not
suffer from complete separation.

Results saved to vary-expert-lpm/<dataset>/<llm>/rep_<seed>.npz.
"""

import sys
sys.path.insert(0, "/code/original/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression
from ppipp_linear import fit_ppipp_linear

CONFIG_PATH = Path(__file__).parent.parent.parent / "dataset_config.json"


def get_feature(dataset: str, phase: str) -> str:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    key     = f"{phase}_variance_feature"
    feature = config.get(dataset, {}).get(key)
    if feature is None:
        raise ValueError(f"No {key} configured for dataset '{dataset}'. Update dataset_config.json.")
    return feature


def fit_ols(Y, x2):
    """
    OLS (linear probability model) with a single feature.
    Returns [β₀, β_feature] as a 2-element array.
    """
    X   = x2.reshape(-1, 1)
    clf = LinearRegression()
    clf.fit(X, Y)
    return np.array([clf.intercept_, clf.coef_[0]])


def fit_ppi_ols(Y, Y_hat, x2, selected_mask):
    """
    PPI for linear regression — closed-form solution.

    θ_PPI = (X^T X)^{-1} [X^T ŷ_all - X_lab^T ŷ_lab + X_lab^T y_lab]

    where X is the full dataset augmented with intercept.
    No optimization needed — exact solution.
    """
    ones  = np.ones(len(Y))
    X     = np.column_stack([ones, x2])
    X_lab = X[selected_mask]
    y_lab = Y[selected_mask].astype(float)
    yhat_lab = Y_hat[selected_mask].astype(float)

    rhs = X.T @ Y_hat - X_lab.T @ yhat_lab + X_lab.T @ y_lab
    try:
        return np.linalg.solve(X.T @ X, rhs)
    except np.linalg.LinAlgError:
        print("    PPI (linear) failed: singular matrix")
        return np.array([np.nan, np.nan])


def fit_dsl_ols(Y, Y_hat, x2, selected_mask, feature: str):
    """
    DSL estimate for linear regression with a single feature (lambda=0).
    Uses model="linear" in R's dsl().
    Returns NaN array on failure.
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
                    model         = "linear",
                    formula       = Y ~ {feature},
                    predicted_var = "Y",
                    prediction    = "Y_hat",
                    data          = data,
                    seed          = Sys.time(),
                    lambda        = 0
                ))
                write.csv(out$coefficients, "{coeff_file}", row.names=FALSE)
                sink()
            """)
            coeffs = np.array(pd.read_csv(coeff_file)).squeeze()
        except Exception as e:
            print(f"    DSL (linear) failed: {e}")
            return np.array([np.nan, np.nan])

    return np.atleast_1d(coeffs)


def compute_one_n(Y, Y_hat, x2, n, seed, feature: str):
    """
    For one expert sample size n: select n rows, compute θ for each method.
    Returns four 2-element arrays [β₀, β_feature] (NaN on per-method failure).
    """
    rng = np.random.default_rng(seed)
    selected_mask = np.zeros(len(Y), dtype=bool)
    selected_mask[rng.choice(len(Y), size=n, replace=False)] = True

    beta_exp   = fit_ols(Y[selected_mask], x2[selected_mask])
    beta_dsl   = fit_dsl_ols(Y, Y_hat, x2, selected_mask, feature)
    beta_ppi   = fit_ppi_ols(Y, Y_hat, x2, selected_mask)
    beta_ppipp = fit_ppipp_linear(Y, Y_hat, x2.reshape(-1, 1), selected_mask)

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
        help="Dataset name — used to look up feature from dataset_config.json")
    parser.add_argument("--phase",   type=str, choices=["low", "high"], default="low",
        help="'low' for low-variance feature, 'high' for high-variance feature")
    args = parser.parse_args()

    feature = get_feature(args.dataset, args.phase)
    print(f"Seed: {args.seed} | CSV: {args.annotated_csv} | Feature: {feature} | mode: LPM (OLS)")

    data  = pd.read_csv(args.annotated_csv)
    if len(data) > 997:
        data = data.sample(n=997, random_state=42).reset_index(drop=True)
    Y     = data["y"].to_numpy().astype(float)
    Y_hat = data["y_hat"].to_numpy().astype(float)
    x2    = data[feature].to_numpy().astype(float)

    # Reference: OLS fit on ALL expert labels (population-level ground truth)
    theta_star = fit_ols(Y, x2)
    theta_llm  = fit_ols(Y_hat, x2)

    print(f"theta* [β₀, β_feature]: {theta_star}")
    print(f"theta_llm:              {theta_llm}")

    n_values = np.unique(
        np.round(np.logspace(np.log10(20), np.log10(200), num=10)).astype(int)
    )
    print(f"n values: {n_values.tolist()}")

    num_n        = len(n_values)
    thetas_exp   = np.full((num_n, 2), np.nan)
    thetas_dsl   = np.full((num_n, 2), np.nan)
    thetas_ppi   = np.full((num_n, 2), np.nan)
    thetas_ppipp = np.full((num_n, 2), np.nan)

    for i, n in enumerate(n_values):
        n_seed = args.seed * 10000 + int(n)
        try:
            exp, dsl, ppi, ppipp = compute_one_n(Y, Y_hat, x2, n, n_seed, feature)
        except Exception as e:
            print(f"  n={n:3d} | UNEXPECTED FAILURE: {e}")
            exp = dsl = ppi = ppipp = np.array([np.nan, np.nan])
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
