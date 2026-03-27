"""
Quick test script: compare PPI vs regularized PPI++ on a single repetition.
PPI++ follows Angelopoulos et al. (2023) with L2 regularization added.
λ* is estimated from labeled data using the gradient covariance formula.

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


def fit_logistic_regularized(Y, X):
    clf = LogisticRegression(penalty="l2", C=1.0/LAM_L2, solver="lbfgs", max_iter=1000)
    clf.fit(X, Y.astype(int))
    return np.concatenate([[clf.intercept_[0]], clf.coef_[0]])


def safe_log1pexp(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def fit_ppi(Y, Y_hat, X, selected_mask):
    ones       = np.ones(len(Y))
    X_aug      = np.column_stack([ones, X])
    X_lab      = X_aug[selected_mask]
    Y_lab      = Y[selected_mask].astype(float)
    Yhat_lab   = Y_hat[selected_mask].astype(float)
    X_unlab    = X_aug[~selected_mask]
    Yhat_unlab = Y_hat[~selected_mask].astype(float)

    def objective(theta):
        loss_unlab  = np.mean(-Yhat_unlab * (X_unlab @ theta) + safe_log1pexp(X_unlab @ theta))
        loss_lab_y  = np.mean(-Y_lab      * (X_lab   @ theta) + safe_log1pexp(X_lab   @ theta))
        loss_lab_yh = np.mean(-Yhat_lab   * (X_lab   @ theta) + safe_log1pexp(X_lab   @ theta))
        return loss_unlab - loss_lab_yh + loss_lab_y + LAM_L2 / 2.0 * np.sum(theta[1:] ** 2)

    def gradient(theta):
        grad_unlab  = X_unlab.T @ (expit(X_unlab @ theta) - Yhat_unlab) / len(Yhat_unlab)
        grad_lab_y  = X_lab.T   @ (expit(X_lab   @ theta) - Y_lab)      / len(Y_lab)
        grad_lab_yh = X_lab.T   @ (expit(X_lab   @ theta) - Yhat_lab)   / len(Y_lab)
        grad_l2     = LAM_L2 * np.concatenate([[0.0], theta[1:]])
        return grad_unlab - grad_lab_yh + grad_lab_y + grad_l2

    return minimize(objective, np.zeros(N_COEF), jac=gradient, method="L-BFGS-B").x


def estimate_lambda_star(Y_lab, Yhat_lab, X_lab, X_unlab, theta):
    """
    Estimate optimal λ* following PPI++ (Angelopoulos et al., 2023).
    λ* = trace(V Σ_cov V) / (2(1 + n/N) * trace(V Σ_var V))
    where V = inverse of regularized Hessian.
    """
    n, N = len(Y_lab), len(X_unlab)

    # Individual gradients on labeled data: shape (n, p+1)
    p_lab     = expit(X_lab @ theta)
    grads     = X_lab * (p_lab - Y_lab)[:, None]
    grads_hat = X_lab * (p_lab - Yhat_lab)[:, None]

    # Regularized Hessian estimated on unlabeled data
    p_unlab = expit(X_unlab @ theta)
    W       = p_unlab * (1 - p_unlab)
    H       = (X_unlab.T * W) @ X_unlab / N
    # Add L2 regularization (skip intercept at index 0)
    H[1:, 1:] += LAM_L2 * np.eye(N_COEF - 1)

    try:
        V = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 1.0  # fallback to standard PPI

    # Covariance and variance of gradients
    Sigma_cov = (grads.T @ grads_hat) / n
    Sigma_var = (grads_hat.T @ grads_hat) / n

    numerator   = np.trace(V @ Sigma_cov @ V)
    denominator = 2.0 * (1 + n / N) * np.trace(V @ Sigma_var @ V)

    if denominator <= 0:
        return 1.0

    lam_star = float(np.clip(numerator / denominator, 0.0, 1.0))
    return lam_star


def fit_ppipp_regularized(Y, Y_hat, X, selected_mask):
    """
    PPI++ with L2 regularization.
    Estimates λ* from labeled data, then optimizes the scaled PPI objective.
    """
    ones       = np.ones(len(Y))
    X_aug      = np.column_stack([ones, X])
    X_lab      = X_aug[selected_mask]
    Y_lab      = Y[selected_mask].astype(float)
    Yhat_lab   = Y_hat[selected_mask].astype(float)
    X_unlab    = X_aug[~selected_mask]
    Yhat_unlab = Y_hat[~selected_mask].astype(float)

    # Step 1: initial estimate from labeled data (regularized)
    theta_init = fit_logistic_regularized(Y_lab, X[selected_mask])

    # Step 2: estimate λ*
    lam_star = estimate_lambda_star(Y_lab, Yhat_lab, X_lab, X_unlab, theta_init)
    print(f"    λ* = {lam_star:.4f}")

    # Step 3: optimize PPI++ objective with λ* and L2 regularization
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

    return minimize(objective, theta_init, jac=gradient, method="L-BFGS-B").x


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

        beta_exp   = fit_logistic_regularized(Y[selected_mask], X[selected_mask])
        beta_ppi   = fit_ppi(Y, Y_hat, X, selected_mask)
        beta_ppipp = fit_ppipp_regularized(Y, Y_hat, X, selected_mask)

        srmse_exp   = np.linalg.norm(beta_exp   - theta_star) / np.linalg.norm(theta_star)
        srmse_ppi   = np.linalg.norm(beta_ppi   - theta_star) / np.linalg.norm(theta_star)
        srmse_ppipp = np.linalg.norm(beta_ppipp - theta_star) / np.linalg.norm(theta_star)

        print(f"--- n={n} ---")
        print(f"  expert_only : {np.round(beta_exp,   4)}  sRMSE={srmse_exp:.4f}")
        print(f"  ppi         : {np.round(beta_ppi,   4)}  sRMSE={srmse_ppi:.4f}")
        print(f"  ppi++       : {np.round(beta_ppipp, 4)}  sRMSE={srmse_ppipp:.4f}")
        print()
