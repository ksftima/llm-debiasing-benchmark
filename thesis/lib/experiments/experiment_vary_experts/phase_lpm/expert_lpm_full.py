"""
Experiment: Vary Expert Sample Size — Full Linear Probability Model (LPM)

Replaces full logistic regression (phase 4) with OLS across all methods:
    - expert_only : OLS on labeled data only
    - ppi         : closed-form PPI correction with squared loss
    - ppipp       : PPI++ with λ* estimated via linear-model Hessian
    - dsl         : R dsl() with model="linear", formula = Y ~ x1+x2+x3+x4+x5

θ = [β₀, β₁, β₂, β₃, β₄, β₅] — intercept + 5 feature coefficients (6-vector).
θ* is the OLS fit on ALL expert labels (population-level ground truth).

No regularization — OLS does not suffer from complete separation.

Results saved to vary-expert-lpm-full/<dataset>/<llm>/rep_<seed>.npz.
"""

import sys
sys.path.insert(0, "/code/original/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression
from ppipp_linear import fit_ppipp_linear

FEATURES = ["x1", "x2", "x3", "x4", "x5"]
N_COEF   = len(FEATURES) + 1  # intercept + 5 features = 6


def fit_ols_full(Y, X):
    """
    OLS on all 5 features.
    Returns [β₀, β₁, β₂, β₃, β₄, β₅] as a 6-element array.
    """
    clf = LinearRegression()
    clf.fit(X, Y)
    return np.concatenate([[clf.intercept_], clf.coef_])


def fit_ppi_ols_full(Y, Y_hat, X, selected_mask):
    """
    PPI for linear regression — closed-form solution.

    θ_PPI = (X_aug^T X_aug)^{-1} [X_aug^T ŷ_all - X_lab^T ŷ_lab + X_lab^T y_lab]

    where X_aug is the full dataset augmented with intercept.
    """
    ones  = np.ones(len(Y))
    X_aug = np.column_stack([ones, X])
    X_lab = X_aug[selected_mask]
    y_lab = Y[selected_mask].astype(float)
    yhat_lab = Y_hat[selected_mask].astype(float)

    rhs = X_aug.T @ Y_hat - X_lab.T @ yhat_lab + X_lab.T @ y_lab
    try:
        return np.linalg.solve(X_aug.T @ X_aug, rhs)
    except np.linalg.LinAlgError:
        print("    PPI (linear) failed: singular matrix")
        return np.full(N_COEF, np.nan)


def fit_dsl_ols_full(Y, Y_hat, X_df, selected_mask, ro):
    """
    DSL estimate for full linear regression (lambda=0, model="linear").
    Accepts a pre-initialized rpy2 robjects module to avoid R startup overhead.
    Returns NaN array on failure.
    """
    Y_true_sel = Y.copy().astype(object)
    Y_true_sel[~selected_mask] = None

    data = X_df.copy()
    data["Y"]     = Y_true_sel
    data["Y_hat"] = Y_hat

    formula = "Y ~ " + " + ".join(FEATURES)

    with tempfile.TemporaryDirectory() as tmp:
        data_file  = Path(tmp) / "data.csv"
        coeff_file = Path(tmp) / "coeff.csv"
        data.to_csv(data_file, index=False)

        try:
            ro.r(f"""
                sink("/dev/null")
                data <- read.csv("{data_file}")
                out <- suppressWarnings(dsl(
                    model         = "linear",
                    formula       = {formula},
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
            print(f"    DSL (linear full) failed: {e}")
            return np.full(N_COEF, np.nan)

    return np.atleast_1d(coeffs)


def compute_one_n(Y, Y_hat, X, X_df, n, seed, ro):
    """
    For one expert sample size n: select n rows, compute θ for each method.
    Returns four 6-element arrays (NaN on per-method failure).
    """
    rng = np.random.default_rng(seed)
    selected_mask = np.zeros(len(Y), dtype=bool)
    selected_mask[rng.choice(len(Y), size=n, replace=False)] = True

    beta_exp   = fit_ols_full(Y[selected_mask], X[selected_mask])
    beta_dsl   = fit_dsl_ols_full(Y, Y_hat, X_df, selected_mask, ro)
    beta_ppi   = fit_ppi_ols_full(Y, Y_hat, X, selected_mask)
    beta_ppipp = fit_ppipp_linear(Y, Y_hat, X, selected_mask)

    return beta_exp, beta_dsl, beta_ppi, beta_ppipp


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path,
        help="Path to annotated CSV, e.g. cuad_llama_annotated.csv")
    parser.add_argument("results_path", type=Path,
        help="Where to save the .npz result for this repetition")
    parser.add_argument("--seed", type=int, required=True,
        help="Random seed = SLURM array task ID")
    args = parser.parse_args()

    print(f"Seed: {args.seed} | CSV: {args.annotated_csv} | mode: LPM full (OLS)")

    data  = pd.read_csv(args.annotated_csv)
    if len(data) > 997:
        data = data.sample(n=997, random_state=42).reset_index(drop=True)
    Y     = data["y"].to_numpy().astype(float)
    Y_hat = data["y_hat"].to_numpy().astype(float)
    X     = data[FEATURES].to_numpy().astype(float)
    X_df  = data[FEATURES].copy()

    theta_star = fit_ols_full(Y, X)
    theta_llm  = fit_ols_full(Y_hat, X)

    print(f"theta* {theta_star}")
    print(f"theta_llm {theta_llm}")

    n_values = np.unique(
        np.round(np.logspace(np.log10(20), np.log10(200), num=10)).astype(int)
    )
    print(f"n values: {n_values.tolist()}")

    # Initialize R once
    import rpy2.robjects as ro
    import rpy2.rinterface_lib.callbacks as rcb
    import logging
    rcb.logger.setLevel(logging.ERROR)
    ro.r('suppressWarnings(library("dsl"))')
    print("R session initialized.")

    num_n        = len(n_values)
    thetas_exp   = np.full((num_n, N_COEF), np.nan)
    thetas_dsl   = np.full((num_n, N_COEF), np.nan)
    thetas_ppi   = np.full((num_n, N_COEF), np.nan)
    thetas_ppipp = np.full((num_n, N_COEF), np.nan)

    for i, n in enumerate(n_values):
        n_seed = args.seed * 10000 + int(n)
        try:
            exp, dsl, ppi, ppipp = compute_one_n(Y, Y_hat, X, X_df, n, n_seed, ro)
        except Exception as e:
            print(f"  n={n:3d} | UNEXPECTED FAILURE: {e}")
            exp = dsl = ppi = ppipp = np.full(N_COEF, np.nan)
        thetas_exp[i]   = exp
        thetas_dsl[i]   = dsl
        thetas_ppi[i]   = ppi
        thetas_ppipp[i] = ppipp
        print(f"  n={n:3d} | exp={exp} | dsl={dsl} | ppi={ppi} | ppipp={ppipp}")

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.results_path,
        theta_star   = theta_star,
        theta_llm    = theta_llm,
        n_values     = n_values,
        thetas_exp   = thetas_exp,
        thetas_dsl   = thetas_dsl,
        thetas_ppi   = thetas_ppi,
        thetas_ppipp = thetas_ppipp,
    )
    print(f"Saved to {args.results_path}")
