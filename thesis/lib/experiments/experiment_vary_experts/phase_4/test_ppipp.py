"""
Quick test script: compare PPI vs regularized PPI++ on a single repetition.
PPI++ follows Angelopoulos et al. (2023) with L2 regularization added.

Run on Vera (needs container):
    apptainer exec --bind /cephyr/users/kesaf/Vera/llm-debiasing-benchmark:/code --pwd /code \\
        $HOME/benchmarking_reg.sif \\
        python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_4/test_ppipp.py \\
        /code/thesis/datasets/annotated/misogynistic/misogynistic_deepseek_annotated.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

FEATURES = ["x1", "x2", "x3", "x4", "x5"]
N_COEF   = len(FEATURES) + 1
LAM_L2   = 0.01
SEED     = 42
N_VALUES = [20, 72, 200]


def fit_logistic_unregularized(Y, X):
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.concatenate([[clf.intercept_[0]], clf.coef_[0]])


def safe_log1pexp(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _split(Y, Y_hat, X, selected_mask):
    """Augment X with intercept column and split into labeled/unlabeled."""
    X_aug = np.column_stack([np.ones(len(Y)), X])
    return (
        X_aug[selected_mask],  X_aug[~selected_mask],
        Y[selected_mask].astype(float),    Y[~selected_mask].astype(float),
        Y_hat[selected_mask].astype(float), Y_hat[~selected_mask].astype(float),
    )


def _optimize_ppi(X_lab, X_unlab, Y_lab, Yhat_lab, Yhat_unlab, theta0, lam_star=1.0):
    """PPI objective with optional λ* scaling (lam_star=1.0 → standard PPI)."""
    def objective(theta):
        loss_unlab  = np.mean(-Yhat_unlab * (X_unlab @ theta) + safe_log1pexp(X_unlab @ theta))
        loss_lab_y  = np.mean(-Y_lab      * (X_lab   @ theta) + safe_log1pexp(X_lab   @ theta))
        loss_lab_yh = np.mean(-Yhat_lab   * (X_lab   @ theta) + safe_log1pexp(X_lab   @ theta))
        return lam_star * (loss_unlab - loss_lab_yh) + loss_lab_y + LAM_L2 / 2.0 * np.sum(theta[1:] ** 2)

    def gradient(theta):
        grad_unlab  = X_unlab.T @ (expit(X_unlab @ theta) - Yhat_unlab) / len(Yhat_unlab)
        grad_lab_y  = X_lab.T   @ (expit(X_lab   @ theta) - Y_lab)      / len(Y_lab)
        grad_lab_yh = X_lab.T   @ (expit(X_lab   @ theta) - Yhat_lab)   / len(Y_lab)
        grad_l2     = LAM_L2 * np.concatenate([[0.0], theta[1:]])
        return lam_star * (grad_unlab - grad_lab_yh) + grad_lab_y + grad_l2

    return minimize(objective, theta0, jac=gradient, method="L-BFGS-B").x


def fit_ppi(Y, Y_hat, X, selected_mask):
    X_lab, X_unlab, Y_lab, _, Yhat_lab, Yhat_unlab = _split(Y, Y_hat, X, selected_mask)
    return _optimize_ppi(X_lab, X_unlab, Y_lab, Yhat_lab, Yhat_unlab, np.zeros(N_COEF))


def estimate_lambda_star(Y_lab, Yhat_lab, X_lab, X_unlab, theta):
    """λ* = trace(V Σ_cov V) / (2(1 + n/N) * trace(V Σ_var V)), Angelopoulos et al. (2023)."""
    n, N = len(Y_lab), len(X_unlab)

    p_lab     = expit(X_lab @ theta)
    grads     = X_lab * (p_lab - Y_lab)[:, None]
    grads_hat = X_lab * (p_lab - Yhat_lab)[:, None]

    W = expit(X_unlab @ theta)
    W = W * (1 - W)
    H = (X_unlab.T * W) @ X_unlab / N
    H[1:, 1:] += LAM_L2 * np.eye(N_COEF - 1)

    try:
        V = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 1.0

    Sigma_cov = (grads.T @ grads_hat) / n
    Sigma_var = (grads_hat.T @ grads_hat) / n

    numerator   = np.trace(V @ Sigma_cov @ V)
    denominator = 2.0 * (1 + n / N) * np.trace(V @ Sigma_var @ V)

    if denominator <= 0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def fit_ppipp(Y, Y_hat, X, selected_mask):
    X_lab, X_unlab, Y_lab, _, Yhat_lab, Yhat_unlab = _split(Y, Y_hat, X, selected_mask)

    # Initial estimate using X_lab with fit_intercept=False (consistent with X_aug)
    clf = LogisticRegression(penalty="l2", C=1.0/LAM_L2, solver="lbfgs",
                             max_iter=1000, fit_intercept=False)
    clf.fit(X_lab, Y_lab.astype(int))
    theta_init = clf.coef_.squeeze()

    lam_star = estimate_lambda_star(Y_lab, Yhat_lab, X_lab, X_unlab, theta_init)
    print(f"    λ* = {lam_star:.4f}")

    return _optimize_ppi(X_lab, X_unlab, Y_lab, Yhat_lab, Yhat_unlab, theta_init, lam_star)


if __name__ == "__main__":
    csv_path = Path(sys.argv[1])
    print(f"Loading: {csv_path.name}\n")

    data  = pd.read_csv(csv_path).sample(frac=1, random_state=42).reset_index(drop=True)
    Y     = data["y"].to_numpy().astype(float)
    Y_hat = data["y_hat"].to_numpy().astype(float)
    X     = data[FEATURES].to_numpy().astype(float)

    theta_star = fit_logistic_unregularized(Y, X)
    print(f"θ*       : {np.round(theta_star, 4)}\n")

    rng = np.random.default_rng(SEED)

    for n in N_VALUES:
        selected_mask = np.zeros(len(Y), dtype=bool)
        selected_mask[rng.choice(len(Y), size=n, replace=False)] = True

        ones  = np.ones(len(Y[selected_mask]))
        X_lab = np.column_stack([ones, X[selected_mask]])
        clf   = LogisticRegression(penalty="l2", C=1.0/LAM_L2, solver="lbfgs",
                                   max_iter=1000, fit_intercept=False)
        clf.fit(X_lab, Y[selected_mask].astype(int))
        beta_exp = clf.coef_.squeeze()

        beta_ppi   = fit_ppi(Y, Y_hat, X, selected_mask)
        beta_ppipp = fit_ppipp(Y, Y_hat, X, selected_mask)

        srmse = lambda b: np.linalg.norm(b - theta_star) / np.linalg.norm(theta_star)

        print(f"--- n={n} ---")
        print(f"  expert_only : {np.round(beta_exp,   4)}  sRMSE={srmse(beta_exp):.4f}")
        print(f"  ppi         : {np.round(beta_ppi,   4)}  sRMSE={srmse(beta_ppi):.4f}")
        print(f"  ppi++       : {np.round(beta_ppipp, 4)}  sRMSE={srmse(beta_ppipp):.4f}")
        print()
