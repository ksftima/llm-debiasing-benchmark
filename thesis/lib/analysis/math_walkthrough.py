"""
Step-by-step math walkthrough for one example run.

Dataset : CUAD  (binary legal clause detection)
LLM     : Llama-3.1-8B-Instruct
n       : 50 expert-labelled rows (out of N=997)
Seed    : 1

Shows, in order:
  1. The data (Y, Y_hat, X)
  2. theta_star  — unregularised logistic on all 997 expert labels (the target)
  3. theta_llm   — unregularised logistic on all 997 LLM labels (biased baseline)
  4. theta_expert — L2 logistic on the 50 labelled rows only
  5. theta_ppi   — PPI objective (manually expanded)
  6. theta_ppipp — PPI++ objective
  7. Side-by-side coefficient table  +  sRMSE relative to theta_star

DSL is skipped here (requires an R session).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

SEED      = 1
N_LABELED = 50
LAM_L2    = 0.01
CSV       = "thesis/datasets/annotated/cuad/cuad_llama_annotated.csv"
FEATURES  = ["x1", "x2", "x3", "x4", "x5"]
FEATURE_NAMES = ["intercept", "x1", "x2", "x3", "x4", "x5"]

# ─────────────────────────────────────────────────────────────────────────────
SEP = "─" * 72

def header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def vec_str(v, fmt="+.4f"):
    return "  ".join(f"{x:{fmt}}" for x in v)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
header("1. DATA")
data = pd.read_csv(CSV)
if len(data) > 997:
    data = data.sample(n=997, random_state=42).reset_index(drop=True)

N     = len(data)
Y     = data["y"].to_numpy().astype(float)          # expert labels   ∈ {0,1}
Y_hat = data["y_hat"].to_numpy().astype(float)      # LLM labels      ∈ {0,1}
X_raw = data[FEATURES].to_numpy().astype(float)     # covariates, shape (N, 5)
X_aug = np.column_stack([np.ones(N), X_raw])        # shape (N, 6) with intercept

print(f"  Total rows N = {N}")
print(f"  Class prevalence  y:     {Y.mean():.3f}  (expert)")
print(f"  Class prevalence  y_hat: {Y_hat.mean():.3f}  (LLM)")
print(f"  Feature means:  {X_raw.mean(axis=0)}")
print(f"  Feature stds:   {X_raw.std(axis=0)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. theta_star  — fit on ALL expert labels
# ─────────────────────────────────────────────────────────────────────────────
header("2. θ*  (unregularised logistic on all N=997 expert labels  →  the target)")
clf_star = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
clf_star.fit(X_raw, Y.astype(int))
theta_star = np.concatenate([[clf_star.intercept_[0]], clf_star.coef_[0]])
print(f"  Coefficients [β₀, β₁, β₂, β₃, β₄, β₅]:")
for name, val in zip(FEATURE_NAMES, theta_star):
    print(f"    {name:<12} {val:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. theta_llm  — fit on ALL LLM labels
# ─────────────────────────────────────────────────────────────────────────────
header("3. θ_llm  (unregularised logistic on all N=997 LLM labels  →  biased baseline)")
clf_llm = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
clf_llm.fit(X_raw, Y_hat.astype(int))
theta_llm = np.concatenate([[clf_llm.intercept_[0]], clf_llm.coef_[0]])
print(f"  Coefficients [β₀, β₁, β₂, β₃, β₄, β₅]:")
for name, val, star in zip(FEATURE_NAMES, theta_llm, theta_star):
    diff = val - star
    print(f"    {name:<12} {val:+.4f}   (Δ from θ* = {diff:+.4f})")

bias_llm = np.linalg.norm(theta_llm - theta_star) / np.linalg.norm(theta_star)
print(f"\n  ||θ_llm - θ*|| / ||θ*|| = {bias_llm:.4f}   (relative bias)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Select n=50 labeled rows
# ─────────────────────────────────────────────────────────────────────────────
header(f"4. SAMPLING  (n={N_LABELED} expert-labelled rows, seed={SEED})")
rng = np.random.default_rng(SEED)
sel = np.zeros(N, dtype=bool)
sel[rng.choice(N, size=N_LABELED, replace=False)] = True

X_lab   = X_aug[sel];    Y_lab   = Y[sel];     Yhat_lab   = Y_hat[sel]
X_unlab = X_aug[~sel];   Yhat_unlab = Y_hat[~sel]
X_lab_raw = X_raw[sel]

print(f"  Labelled   rows: {sel.sum()}   class balance: {Y_lab.mean():.2f}")
print(f"  Unlabelled rows: {(~sel).sum()}")
print()
print(f"  First 5 labelled rows:")
print(f"  {'y':>5}  {'y_hat':>5}  {'x1':>7}  {'x2':>7}  {'x3':>7}  {'x4':>7}  {'x5':>7}")
for i in np.where(sel)[0][:5]:
    row = data[FEATURES].iloc[i].values
    print(f"  {int(Y[i]):>5}  {int(Y_hat[i]):>5}  "
          + "  ".join(f"{v:>7.3f}" for v in row))

# ─────────────────────────────────────────────────────────────────────────────
# 5. theta_expert — L2 logistic on labeled only
# ─────────────────────────────────────────────────────────────────────────────
header(f"5. θ_expert  (L2 logistic on n={N_LABELED} labeled rows,  λ={LAM_L2})")
print(f"""
  Objective (sklearn's L2 logistic):
    L(θ) = (1/n) Σᵢ₌₁ⁿ  log(1 + exp(−yᵢ · Xᵢθ))  +  (λ/2) ||β||²
  where β = θ[1:] (we do not regularise the intercept)
""")
clf_exp = LogisticRegression(penalty="l2", C=1.0/LAM_L2, solver="lbfgs", max_iter=1000)
clf_exp.fit(X_lab_raw, Y_lab.astype(int))
theta_exp = np.concatenate([[clf_exp.intercept_[0]], clf_exp.coef_[0]])
print(f"  Coefficients [β₀, β₁, β₂, β₃, β₄, β₅]:")
for name, val, star in zip(FEATURE_NAMES, theta_exp, theta_star):
    diff = val - star
    print(f"    {name:<12} {val:+.4f}   (Δ from θ* = {diff:+.4f})")
srmse_exp = np.linalg.norm(theta_exp - theta_star) / np.linalg.norm(theta_star)
print(f"\n  sRMSE relative to θ* = {srmse_exp:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. theta_ppi — PPI objective
# ─────────────────────────────────────────────────────────────────────────────
header("6. θ_ppi  (Prediction-Powered Inference logistic regression)")
print(f"""
  Objective:
    L_ppi(θ) =  (1/N_unlab) Σ_unlab  ℓ(ŷᵢ, Xᵢθ)   ← LLM labels on ALL unlabelled rows
              − (1/n)       Σ_lab    ℓ(ŷᵢ, Xᵢθ)   ← subtract LLM correction on labeled
              + (1/n)       Σ_lab    ℓ(yᵢ, Xᵢθ)   ← add expert correction on labeled
              + (λ/2)       ||β||²

  where ℓ(y, Xθ) = −y·(Xθ) + log(1+exp(Xθ))  (binary cross-entropy)

  The first term anchors θ to LLM-predicted labels on the full pool.
  The difference of the last two terms corrects for the LLM's systematic bias:
    (1/n) Σ_lab [ ℓ(yᵢ, Xᵢθ) − ℓ(ŷᵢ, Xᵢθ) ]
  is an unbiased estimate of E[ℓ(y, Xθ) − ℓ(ŷ, Xθ)].

  Gradient:
    ∇L_ppi(θ) =  (1/N_unlab) X_unlab.T (σ(X_unlab θ) − ŷ_unlab)
               − (1/n)       X_lab.T   (σ(X_lab θ)   − ŷ_lab)
               + (1/n)       X_lab.T   (σ(X_lab θ)   − y_lab)
               + λ · [0, β₁, β₂, β₃, β₄, β₅]
""")

def safe_log1pexp(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

def ppi_objective(theta):
    loss_unlab  = np.mean(-Yhat_unlab * (X_unlab @ theta) + safe_log1pexp(X_unlab @ theta))
    loss_lab_y  = np.mean(-Y_lab      * (X_lab   @ theta) + safe_log1pexp(X_lab   @ theta))
    loss_lab_yh = np.mean(-Yhat_lab   * (X_lab   @ theta) + safe_log1pexp(X_lab   @ theta))
    l2 = LAM_L2 / 2.0 * np.sum(theta[1:] ** 2)
    return loss_unlab - loss_lab_yh + loss_lab_y + l2

def ppi_gradient(theta):
    g_unlab  = X_unlab.T @ (expit(X_unlab @ theta) - Yhat_unlab) / len(Yhat_unlab)
    g_lab_y  = X_lab.T   @ (expit(X_lab   @ theta) - Y_lab)      / len(Y_lab)
    g_lab_yh = X_lab.T   @ (expit(X_lab   @ theta) - Yhat_lab)   / len(Y_lab)
    g_l2 = LAM_L2 * np.concatenate([[0.0], theta[1:]])
    return g_unlab - g_lab_yh + g_lab_y + g_l2

# Show the three loss components at θ=0 to illustrate the decomposition
theta0 = np.zeros(6)
loss_unlab0  = np.mean(-Yhat_unlab * (X_unlab @ theta0) + safe_log1pexp(X_unlab @ theta0))
loss_lab_y0  = np.mean(-Y_lab      * (X_lab   @ theta0) + safe_log1pexp(X_lab   @ theta0))
loss_lab_yh0 = np.mean(-Yhat_lab   * (X_lab   @ theta0) + safe_log1pexp(X_lab   @ theta0))
print(f"  Objective components at θ=0:")
print(f"    (1) LLM loss on unlabelled:  {loss_unlab0:.4f}")
print(f"    (2) LLM loss on labelled:    {loss_lab_yh0:.4f}  (subtracted)")
print(f"    (3) Expert loss on labelled: {loss_lab_y0:.4f}  (added)")
print(f"    Total = (1)−(2)+(3) = {loss_unlab0 - loss_lab_yh0 + loss_lab_y0:.4f}")

result_ppi = minimize(ppi_objective, np.zeros(6), jac=ppi_gradient, method="L-BFGS-B")
theta_ppi = result_ppi.x
print(f"\n  Optimisation converged: {result_ppi.success}  ({result_ppi.message})")
print(f"  Coefficients [β₀, β₁, β₂, β₃, β₄, β₅]:")
for name, val, star in zip(FEATURE_NAMES, theta_ppi, theta_star):
    diff = val - star
    print(f"    {name:<12} {val:+.4f}   (Δ from θ* = {diff:+.4f})")
srmse_ppi = np.linalg.norm(theta_ppi - theta_star) / np.linalg.norm(theta_star)
print(f"\n  sRMSE relative to θ* = {srmse_ppi:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. theta_ppipp — PPI++
# ─────────────────────────────────────────────────────────────────────────────
header("7. θ_ppi++  (PPI++ — power-tuned version)")
print(f"""
  PPI++ wraps PPI by finding the optimal weight λ_pp ∈ [0,1] that minimises
  the variance of the resulting estimator:

    L_ppi++(θ) =  λ_pp · (1/N_unlab) Σ_unlab  ℓ(ŷᵢ, Xᵢθ)
               −  λ_pp · (1/n)       Σ_lab    ℓ(ŷᵢ, Xᵢθ)
               +         (1/n)       Σ_lab    ℓ(yᵢ, Xᵢθ)
               + (λ/2)  ||β||²

  When λ_pp = 1  →  reduces to PPI.
  When λ_pp = 0  →  reduces to expert-only.
  λ_pp is estimated from the labelled data and balances how much to trust the LLM.
""")

try:
    import sys
    sys.path.insert(0, "thesis/lib/experiments/experiment_vary_experts")
    from ppipp import fit_ppipp
    theta_ppipp = fit_ppipp(Y, Y_hat, X_raw, sel, LAM_L2)
    print(f"  Coefficients [β₀, β₁, β₂, β₃, β₄, β₅]:")
    for name, val, star in zip(FEATURE_NAMES, theta_ppipp, theta_star):
        diff = val - star
        print(f"    {name:<12} {val:+.4f}   (Δ from θ* = {diff:+.4f})")
    srmse_ppipp = np.linalg.norm(theta_ppipp - theta_star) / np.linalg.norm(theta_star)
    print(f"\n  sRMSE relative to θ* = {srmse_ppipp:.4f}")
except Exception as e:
    print(f"  (ppipp module not available locally: {e})")
    theta_ppipp = None
    srmse_ppipp = None

# ─────────────────────────────────────────────────────────────────────────────
# 8. Side-by-side summary
# ─────────────────────────────────────────────────────────────────────────────
header("8. SUMMARY — all coefficients side by side")
methods = ["θ*", "θ_llm", "θ_expert", "θ_ppi", "θ_ppi++"]
thetas  = [theta_star, theta_llm, theta_exp, theta_ppi, theta_ppipp]

col_w = 12
print(f"  {'':12}" + "".join(f"  {m:>{col_w}}" for m in methods))
print(f"  {'':12}" + "  " + "  ".join(["─" * col_w] * len(methods)))
for i, name in enumerate(FEATURE_NAMES):
    row = f"  {name:<12}"
    for t in thetas:
        if t is None:
            row += f"  {'n/a':>{col_w}}"
        else:
            row += f"  {t[i]:>{col_w}.4f}"
    print(row)

print()
print(f"  {'sRMSE':12}" + "".join([
    f"  {'—':>{col_w}}",
    f"  {bias_llm:>{col_w}.4f}",
    f"  {srmse_exp:>{col_w}.4f}",
    f"  {srmse_ppi:>{col_w}.4f}",
    f"  {srmse_ppipp:>{col_w}.4f}" if srmse_ppipp is not None else f"  {'n/a':>{col_w}}",
]))

print(f"""
  Interpretation:
    sRMSE = ||θ_method − θ*|| / ||θ*||

    Lower is better. θ* is the oracle estimate (all 997 expert labels).
    θ_expert uses only {N_LABELED}/{N} of those labels — much noisier.
    PPI and PPI++ leverage the LLM labels on the remaining {N-N_LABELED} rows
    to improve the estimate beyond what the {N_LABELED} expert labels alone give.
""")
