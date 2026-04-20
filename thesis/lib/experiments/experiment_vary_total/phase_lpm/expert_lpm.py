"""
Experiment: Vary Total Dataset Size — LPM (OLS), Single Feature

θ = [β₀, β_feature] — intercept + one coefficient (2-vector).

Fix n (expert annotations), vary N (total samples) log-spaced from n to 1000.
No regularization — OLS is the natural unregularized estimator.

Results saved to vary-total-lpm-{low,high}/<dataset>/<llm>/n_<n>/rep_<seed>.npz.
"""

import sys
sys.path.insert(0, "/code/original/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent / "experiment_vary_experts" / "phase_lpm"))

import json
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression
from ppipp_linear import fit_ppipp_linear

CONFIG_PATH = Path(__file__).parent.parent.parent / "dataset_config.json"

N_MAX = 997


def get_feature(dataset: str, phase: str) -> str:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    key     = f"{phase}_variance_feature"
    feature = config.get(dataset, {}).get(key)
    if feature is None:
        raise ValueError(f"No {key} configured for dataset '{dataset}'.")
    return feature


def fit_ols(Y, x2):
    clf = LinearRegression()
    clf.fit(x2.reshape(-1, 1), Y)
    return np.array([clf.intercept_, clf.coef_[0]])


def fit_ppi_ols(Y, Y_hat, x2, selected_mask):
    ones  = np.ones(len(Y))
    X     = np.column_stack([ones, x2])
    X_lab = X[selected_mask]
    y_lab    = Y[selected_mask].astype(float)
    yhat_lab = Y_hat[selected_mask].astype(float)

    N, n = len(Y), int(selected_mask.sum())
    rhs  = X.T @ Y_hat + (N / n) * X_lab.T @ (y_lab - yhat_lab)
    try:
        return np.linalg.solve(X.T @ X, rhs)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan])


def fit_dsl_ols(Y, Y_hat, x2, selected_mask, feature: str, ro):
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
                data <- read.csv("{data_file}")
                out <- suppressWarnings(dsl(
                    model         = "lm",
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
            print(f"    DSL (lm) failed: {e}")
            return np.array([np.nan, np.nan])

    return np.atleast_1d(coeffs)


def compute_one_N(Y_full, Y_hat_full, x2_full, N, n_expert, seed, feature, ro):
    rng = np.random.default_rng(seed)

    idx   = rng.choice(len(Y_full), size=N, replace=False)
    Y     = Y_full[idx]
    Y_hat = Y_hat_full[idx]
    x2    = x2_full[idx]

    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[rng.choice(N, size=n_expert, replace=False)] = True

    beta_exp   = fit_ols(Y[selected_mask], x2[selected_mask])
    beta_dsl   = fit_dsl_ols(Y, Y_hat, x2, selected_mask, feature, ro)
    beta_ppi   = fit_ppi_ols(Y, Y_hat, x2, selected_mask)
    beta_ppipp = fit_ppipp_linear(Y, Y_hat, x2.reshape(-1, 1), selected_mask)

    return beta_exp, beta_dsl, beta_ppi, beta_ppipp


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path)
    parser.add_argument("results_path",  type=Path)
    parser.add_argument("--seed",     type=int, required=True)
    parser.add_argument("--dataset",  type=str, required=True)
    parser.add_argument("--phase",    type=str, choices=["low", "high"], default="low")
    parser.add_argument("--n-expert", type=int, required=True)
    args = parser.parse_args()

    feature = get_feature(args.dataset, args.phase)
    print(f"Seed: {args.seed} | CSV: {args.annotated_csv} | Feature: {feature} | n_expert: {args.n_expert} | mode: LPM")

    data = pd.read_csv(args.annotated_csv).sample(frac=1, random_state=42).reset_index(drop=True)
    if len(data) > N_MAX:
        data = data.iloc[:N_MAX]

    Y_full     = data["y"].to_numpy().astype(float)
    Y_hat_full = data["y_hat"].to_numpy().astype(float)
    x2_full    = data[feature].to_numpy().astype(float)

    theta_star = fit_ols(Y_full, x2_full)
    theta_llm  = fit_ols(Y_hat_full, x2_full)

    print(f"theta* [β₀, β_feature]: {theta_star}")
    print(f"theta_llm:              {theta_llm}")

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
    thetas_exp   = np.full((num_N, 2), np.nan)
    thetas_dsl   = np.full((num_N, 2), np.nan)
    thetas_ppi   = np.full((num_N, 2), np.nan)
    thetas_ppipp = np.full((num_N, 2), np.nan)

    phase_label = "low_variance" if args.phase == "low" else "high_variance"

    for i, N in enumerate(N_values):
        N_seed = args.seed * 100000 + int(N)
        try:
            exp, dsl, ppi, ppipp = compute_one_N(Y_full, Y_hat_full, x2_full, N, args.n_expert, N_seed, feature, ro)
        except Exception as e:
            print(f"  N={N:4d} | UNEXPECTED FAILURE: {e}")
            exp = dsl = ppi = ppipp = np.array([np.nan, np.nan])
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
        phase        = np.array([phase_label]),
    )
    print(f"Saved to {args.results_path}")
