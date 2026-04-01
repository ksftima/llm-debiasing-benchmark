"""
Experiment: Vary Total Dataset Size — Phase 1: Class Prevalence

No features (X) — intercept-only logistic regression.
θ = [β₀] — log-odds of P(Y=1).

Fix n (expert annotations), vary N (total samples) log-spaced from n to 1000.

For each repetition (controlled by --seed = SLURM array task ID):
    - Compute theta* = logit(mean(ALL y))  reference on full dataset
    - Compute theta_llm = logit(mean(ALL y_hat))  LLM-only on full dataset
    - For each N in log-spaced [n_expert, 1000]:
        - Subsample N rows from the full dataset
        - Select n_expert of those N rows as "expert-labeled"
        - theta_exp = logit(mean(y[selected]))
        - theta_dsl = R dsl(model="logit", formula=Y ~ 1)
        - theta_ppi = ppi_logistic_pointestimate(X=ones)
    - Save all results to a .npz file
"""

import sys
sys.path.insert(0, "/code/original/lib")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent / "experiment_vary_experts"))

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from scipy.special import logit
from ppi_py import ppi_logistic_pointestimate
from ppipp import fit_ppipp

N_MAX = 997  # cap total dataset size — fixed to smallest annotated dataset


def fit_logit_intercept_only(Y):
    p = np.clip(Y.mean(), 1e-7, 1 - 1e-7)
    return np.array([logit(p)])


def fit_ppi_intercept_only(Y, Y_hat, selected_mask):
    N = len(Y)
    ones = np.ones((N, 1))
    return ppi_logistic_pointestimate(
        X              = ones[selected_mask],
        Y              = Y[selected_mask],
        Yhat           = Y_hat[selected_mask],
        X_unlabeled    = ones[~selected_mask],
        Yhat_unlabeled = Y_hat[~selected_mask],
    )


def fit_dsl_intercept_only(Y, Y_hat, selected_mask):
    import rpy2.robjects as ro
    import rpy2.rinterface_lib.callbacks as rcb
    import logging
    rcb.logger.setLevel(logging.ERROR)

    Y_true_sel = Y.copy().astype(object)
    Y_true_sel[~selected_mask] = None

    data = pd.DataFrame({"Y": Y_true_sel, "Y_hat": Y_hat})

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
                    formula       = Y ~ 1,
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
            print(f"    DSL failed: {e}")
            return np.array([np.nan])

    return np.atleast_1d(coeffs)


def compute_one_N(Y_full, Y_hat_full, N, n_expert, seed):
    """
    Subsample N rows from full dataset, select n_expert as expert-labeled.
    Returns four 1-element arrays (log-odds).
    """
    rng = np.random.default_rng(seed)

    # Subsample N rows
    idx = rng.choice(len(Y_full), size=N, replace=False)
    Y     = Y_full[idx]
    Y_hat = Y_hat_full[idx]

    # Select n_expert as expert-labeled
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[rng.choice(N, size=n_expert, replace=False)] = True

    beta_exp   = fit_logit_intercept_only(Y[selected_mask])
    beta_dsl   = fit_dsl_intercept_only(Y, Y_hat, selected_mask)
    beta_ppi   = fit_ppi_intercept_only(Y, Y_hat, selected_mask)
    beta_ppipp = fit_ppipp(Y, Y_hat, np.empty((len(Y), 0)), selected_mask, lam_l2=0.0)

    return beta_exp, beta_dsl, beta_ppi, beta_ppipp


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_csv", type=Path)
    parser.add_argument("results_path",  type=Path)
    parser.add_argument("--seed",     type=int, required=True)
    parser.add_argument("--n-expert", type=int, required=True,
        help="Fixed number of expert annotations (e.g. 50, 100, 200)")
    args = parser.parse_args()

    print(f"Seed: {args.seed} | CSV: {args.annotated_csv} | n_expert: {args.n_expert}")

    data  = pd.read_csv(args.annotated_csv)
    Y_full     = data["y"].to_numpy().astype(float)
    Y_hat_full = data["y_hat"].to_numpy().astype(float)

    # Cap to N_MAX
    if len(Y_full) > N_MAX:
        rng0 = np.random.default_rng(0)
        idx0 = rng0.choice(len(Y_full), size=N_MAX, replace=False)
        Y_full     = Y_full[idx0]
        Y_hat_full = Y_hat_full[idx0]

    # Reference: fit on ALL data (after cap)
    theta_star = fit_logit_intercept_only(Y_full)
    theta_llm  = fit_logit_intercept_only(Y_hat_full)

    # N values: log-spaced from n_expert to N_MAX (capped at actual dataset size as safety)
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
    thetas_exp   = np.zeros(num_N)
    thetas_dsl   = np.zeros(num_N)
    thetas_ppi   = np.zeros(num_N)
    thetas_ppipp = np.zeros(num_N)

    for i, N in enumerate(N_values):
        N_seed = args.seed * 100000 + int(N)
        exp, dsl, ppi, ppipp = compute_one_N(Y_full, Y_hat_full, N, args.n_expert, N_seed)
        thetas_exp[i]   = float(exp)
        thetas_dsl[i]   = float(dsl)
        thetas_ppi[i]   = float(ppi)
        thetas_ppipp[i] = float(ppipp[0])
        print(f"  N={N:4d} | exp={float(exp):.4f} | dsl={float(dsl):.4f} | ppi={float(ppi):.4f} | ppipp={float(ppipp[0]):.4f}")

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
