"""
PPI++ for linear regression (Linear Probability Model).

Analogous to ../ppipp.py but uses squared loss instead of logistic loss.
λ* is estimated using the linear-model Hessian (X_unlab^T X_unlab / N).

The PPI++ objective for linear regression:
    L(θ) = λ* * [mean_unlab(||ŷ - Xθ||²/2) - mean_lab(||ŷ_lab - X_lab θ||²/2)]
          + mean_lab(||y_lab - X_lab θ||²/2)

No L2 regularization — OLS is the natural unregularized estimator.
The objective is quadratic, so L-BFGS-B converges in very few iterations.
"""

import numpy as np
from scipy.optimize import minimize


def _split(Y, Y_hat, X, selected_mask):
    """Augment X with intercept column and split into labeled/unlabeled sets."""
    X_aug = np.column_stack([np.ones(len(Y)), X])
    return (
        X_aug[selected_mask],               X_aug[~selected_mask],
        Y[selected_mask].astype(float),     Y[~selected_mask].astype(float),
        Y_hat[selected_mask].astype(float), Y_hat[~selected_mask].astype(float),
    )


def _optimize_ppi_linear(X_lab, X_unlab, Y_lab, Yhat_lab, Yhat_unlab, theta0, lam_star=1.0):
    """
    Optimize PPI++ objective with squared loss.
    lam_star=1.0 → standard PPI; lam_star<1 → PPI++ with down-weighted LLM correction.
    """
    n_coef = len(theta0)

    def objective(theta):
        loss_unlab  = np.mean((Yhat_unlab - X_unlab @ theta) ** 2) / 2
        loss_lab_y  = np.mean((Y_lab      - X_lab   @ theta) ** 2) / 2
        loss_lab_yh = np.mean((Yhat_lab   - X_lab   @ theta) ** 2) / 2
        return lam_star * (loss_unlab - loss_lab_yh) + loss_lab_y

    def gradient(theta):
        grad_unlab  = X_unlab.T @ (X_unlab @ theta - Yhat_unlab) / len(Yhat_unlab)
        grad_lab_y  = X_lab.T   @ (X_lab   @ theta - Y_lab)      / len(Y_lab)
        grad_lab_yh = X_lab.T   @ (X_lab   @ theta - Yhat_lab)   / len(Y_lab)
        return lam_star * (grad_unlab - grad_lab_yh) + grad_lab_y

    try:
        result = minimize(objective, theta0, jac=gradient, method="L-BFGS-B")
        if not result.success:
            print(f"    PPI++ (linear) did not converge: {result.message}")
        return result.x
    except Exception as e:
        print(f"    PPI++ (linear) failed: {e}")
        return np.full(n_coef, np.nan)


def estimate_lambda_star_linear(Y_lab, Yhat_lab, X_lab, X_unlab, theta):
    """
    Estimate optimal λ* for linear regression following Angelopoulos et al. (2023).
        λ* = trace(V Σ_cov V) / (2(1 + n/N) * trace(V Σ_var V))

    For linear regression (no regularization):
        Hessian H = X_unlab^T X_unlab / N
        Per-sample gradient: ∇ℓᵢ(θ) = xᵢ (xᵢ^T θ - yᵢ)
    """
    n, N = len(Y_lab), len(X_unlab)

    resid_y   = X_lab @ theta - Y_lab
    resid_yh  = X_lab @ theta - Yhat_lab
    grads     = X_lab * resid_y[:, None]    # (n, p+1)
    grads_hat = X_lab * resid_yh[:, None]   # (n, p+1)

    H = X_unlab.T @ X_unlab / N

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


def fit_ppipp_linear(Y, Y_hat, X, selected_mask):
    """
    PPI++ estimate for linear regression (LPM).
    X: feature matrix of shape (N, p) — intercept is added internally.
    Returns theta of shape (p+1,): [β₀, β₁, ..., βₚ].
    """
    X_lab, X_unlab, Y_lab, _, Yhat_lab, Yhat_unlab = _split(Y, Y_hat, X, selected_mask)

    # Initialize with OLS on labeled data
    theta_init, _, _, _ = np.linalg.lstsq(X_lab, Y_lab, rcond=None)

    lam_star = estimate_lambda_star_linear(Y_lab, Yhat_lab, X_lab, X_unlab, theta_init)
    print(f"    λ* = {lam_star:.4f}")

    return _optimize_ppi_linear(X_lab, X_unlab, Y_lab, Yhat_lab, Yhat_unlab,
                                theta_init, lam_star)
