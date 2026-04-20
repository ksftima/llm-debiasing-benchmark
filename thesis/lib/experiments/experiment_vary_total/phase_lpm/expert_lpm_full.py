"""
Experiment: Vary Total Dataset Size — Full LPM (OLS), 5 Features

θ = [β₀, β₁, β₂, β₃, β₄, β₅] — intercept + 5 feature coefficients (6-vector).

Fix n (expert annotations), vary N (total samples) log-spaced from n to 1000.
No regularization — OLS does not suffer from complete separation.

Results saved to vary-total-lpm-full/<dataset>/<llm>/n_<n>/rep_<seed>.npz.
"""

import sys
sys.path.insert(0, "/code/original/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent / "experiment_vary_experts" / "phase_lpm"))

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression
from ppipp_linear import fit_ppipp_linear

FEATURES = ["x1", "x2", "x3", "x4", "x5"]
N_COEF   = len(FEATURES) + 1
N_MAX    = 997


def fit_ols_full(Y, X):
    clf = LinearRegression()
    clf.fit(X, Y)
    return np.concatenate([[clf.intercept_], clf.coef_])


def fit_ppi_ols_full(Y, Y_hat, X, selected_mask):
    ones  = np.ones(len(Y))
    X_aug = np.column_stack([ones, X])
    X_lab = X_aug[selected_mask]
    y_lab    = Y[selected_mask].astype(float)
    yhat_lab = Y_hat[selected_mask].astype(float)

    N, n = len(Y), int(selected_mask.sum())
    rhs  = X_aug.T @ Y_hat + (N / n) * X_lab.T @ (y_lab - yhat_lab)
    try:
        return np.linalg.solve(X_aug.T @ X_aug, rhs)
    except np.linalg.LinAlgError:
        return np.full(N_COEF, np.nan)


def fit_dsl_ols_full(Y, Y_hat, X_df, selected_mask, ro):
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
                    model         = "lm",
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
            print(f"    DSL (lm full) failed: {e}")
            return np.full(N_COEF, np.nan)

    return np.atleast_1d(coeffs)


def compute_one_N(Y_full, Y_hat_full, X_full, X_df_full, N, n_expert, seed, ro):
    rng = np.random.default_rng(seed)

    idx   = rng.choice(len(Y_full), size=N, replace=False)
    Y     = Y_full[idx]
    Y_hat = Y_hat_full[idx]
    X     = X_full[idx]
    X_df  = X_df_full.iloc[idx].reset_index(drop=True)

    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[rng.choice(N, size=n_expert, replace=False)] = True

    beta_exp   = fit_ols_full(Y[selected_mask], X[selected_mask])
    beta_dsl   = fit_dsl_ols_full(Y, Y_hat, X_df, selected_mask, ro)
    beta_ppi   = fit_ppi_ols_full(Y, Y_hat, X, selected_mask)
    beta_ppipp = fit_ppipp_linear(Y, Y_hat, X, selected_mask)

    return beta_exp, beta_dsl, beta_ppi, beta_ppipp


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path)
    parser.add_argument("results_path",  type=Path)
    parser.add_argument("--seed",     type=int, required=True)
    parser.add_argument("--n-expert", type=int, required=True)
    args = parser.parse_args()

    print(f"Seed: {args.seed} | CSV: {args.annotated_csv} | n_expert: {args.n_expert} | mode: LPM full")

    data = pd.read_csv(args.annotated_csv).sample(frac=1, random_state=42).reset_index(drop=True)
    if len(data) > N_MAX:
        data = data.iloc[:N_MAX]

    Y_full     = data["y"].to_numpy().astype(float)
    Y_hat_full = data["y_hat"].to_numpy().astype(float)
    X_full     = data[FEATURES].to_numpy().astype(float)
    X_df_full  = data[FEATURES].copy()

    theta_star = fit_ols_full(Y_full, X_full)
    theta_llm  = fit_ols_full(Y_hat_full, X_full)

    print(f"theta* {theta_star}")
    print(f"theta_llm {theta_llm}")

    N_values = np.unique(
        np.round(np.logspace(
            np.log10(args.n_expert),
            np.log10(min(N_MAX, len(Y_full))),
            num=10,
        )).astype(int)
    )
    N_values = N_values[N_values > args.n_expert]
    print(f"N values: {N_values.tolist()}")

    import rpy2.robjects as ro
    import rpy2.rinterface_lib.callbacks as rcb
    import logging
    rcb.logger.setLevel(logging.ERROR)
    ro.r('suppressWarnings(library("dsl"))')
    print("R session initialized.")

    num_N        = len(N_values)
    thetas_exp   = np.full((num_N, N_COEF), np.nan)
    thetas_dsl   = np.full((num_N, N_COEF), np.nan)
    thetas_ppi   = np.full((num_N, N_COEF), np.nan)
    thetas_ppipp = np.full((num_N, N_COEF), np.nan)

    for i, N in enumerate(N_values):
        N_seed = args.seed * 100000 + int(N)
        try:
            exp, dsl, ppi, ppipp = compute_one_N(Y_full, Y_hat_full, X_full, X_df_full, N, args.n_expert, N_seed, ro)
        except Exception as e:
            print(f"  N={N:4d} | UNEXPECTED FAILURE: {e}")
            exp = dsl = ppi = ppipp = np.full(N_COEF, np.nan)
        thetas_exp[i]   = exp
        thetas_dsl[i]   = dsl
        thetas_ppi[i]   = ppi
        thetas_ppipp[i] = ppipp
        print(f"  N={N:4d} | exp={exp} | dsl={dsl} | ppi={ppi} | ppipp={ppipp}")

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
    )
    print(f"Saved to {args.results_path}")
