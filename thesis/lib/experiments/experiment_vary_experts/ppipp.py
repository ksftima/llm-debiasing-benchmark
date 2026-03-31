"""
Regularized PPI++ implementation following Angelopoulos et al. (2023).

λ* is estimated from labeled data via the gradient covariance formula,
then used to adaptively scale the unlabeled correction term.
When the LLM is a good annotator λ*→1 (standard PPI).
When the LLM is poor λ*→0 (falls back to expert_only).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def _safe_log1pexp(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _split(Y, Y_hat, X, selected_mask):
    """Augment X with intercept column and split into labeled/unlabeled sets."""
    if X.shape[1] > 0:
        X_aug = np.column_stack([np.ones(len(Y)), X])
    else:
        X_aug = np.ones((len(Y), 1))  # intercept-only (phase 1)
    return (
        X_aug[selected_mask],           X_aug[~selected_mask],
        Y[selected_mask].astype(float), Y[~selected_mask].astype(float),
        Y_hat[selected_mask].astype(float), Y_hat[~selected_mask].astype(float),
    )


def _optimize_ppi(X_lab, X_unlab, Y_lab, Yhat_lab, Yhat_unlab, theta0, lam_l2, lam_star=1.0):
    """
    Optimize PPI objective with optional λ* scaling and L2 regularization.
    lam_star=1.0 → standard PPI; lam_star<1 → PPI++ with down-weighted LLM correction.
    """
    n_coef = len(theta0)

    def objective(theta):
        loss_unlab  = np.mean(-Yhat_unlab * (X_unlab @ theta) + _safe_log1pexp(X_unlab @ theta))
        loss_lab_y  = np.mean(-Y_lab      * (X_lab   @ theta) + _safe_log1pexp(X_lab   @ theta))
        loss_lab_yh = np.mean(-Yhat_lab   * (X_lab   @ theta) + _safe_log1pexp(X_lab   @ theta))
        return lam_star * (loss_unlab - loss_lab_yh) + loss_lab_y + lam_l2 / 2.0 * np.sum(theta[1:] ** 2)

    def gradient(theta):
        grad_unlab  = X_unlab.T @ (expit(X_unlab @ theta) - Yhat_unlab) / len(Yhat_unlab)
        grad_lab_y  = X_lab.T   @ (expit(X_lab   @ theta) - Y_lab)      / len(Y_lab)
        grad_lab_yh = X_lab.T   @ (expit(X_lab   @ theta) - Yhat_lab)   / len(Y_lab)
        grad_l2     = lam_l2 * np.concatenate([[0.0], theta[1:]])
        return lam_star * (grad_unlab - grad_lab_yh) + grad_lab_y + grad_l2

    try:
        result = minimize(objective, theta0, jac=gradient, method="L-BFGS-B")
        if not result.success:
            print(f"    PPI++ optimisation did not converge: {result.message}")
        return result.x
    except Exception as e:
        print(f"    PPI++ failed: {e}")
        return np.full(n_coef, np.nan)


def estimate_lambda_star(Y_lab, Yhat_lab, X_lab, X_unlab, theta, lam_l2):
    """
    Estimate optimal λ* from Angelopoulos et al. (2023):
        λ* = trace(V Σ_cov V) / (2(1 + n/N) * trace(V Σ_var V))
    where V = inverse of regularized Hessian on unlabeled data.
    """
    n, N   = len(Y_lab), len(X_unlab)
    n_coef = len(theta)

    p_lab     = expit(X_lab @ theta)
    grads     = X_lab * (p_lab - Y_lab)[:, None]
    grads_hat = X_lab * (p_lab - Yhat_lab)[:, None]

    W = expit(X_unlab @ theta)
    W = W * (1 - W)
    H = (X_unlab.T * W) @ X_unlab / N
    H[1:, 1:] += lam_l2 * np.eye(n_coef - 1)

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


def fit_ppipp(Y, Y_hat, X, selected_mask, lam_l2):
    """
    PPI++ estimate for logistic regression with L2 regularization.
    X: feature matrix of shape (N, p) — intercept is added internally.
    Returns theta of shape (p+1,): [β₀, β₁, ..., βₚ].
    """
    X_lab, X_unlab, Y_lab, _, Yhat_lab, Yhat_unlab = _split(Y, Y_hat, X, selected_mask)

    clf = LogisticRegression(penalty="l2", C=1.0/lam_l2, solver="lbfgs",
                             max_iter=1000, fit_intercept=False)
    clf.fit(X_lab, Y_lab.astype(int))
    theta_init = clf.coef_.squeeze()

    lam_star = estimate_lambda_star(Y_lab, Yhat_lab, X_lab, X_unlab, theta_init, lam_l2)
    print(f"    λ* = {lam_star:.4f}")

    return _optimize_ppi(X_lab, X_unlab, Y_lab, Yhat_lab, Yhat_unlab, theta_init, lam_l2, lam_star)
