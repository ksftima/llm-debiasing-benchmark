"""
Test script for expert_class_prevalence.py

Generates fake binary data with known properties, runs the prevalence
experiment, and checks that the outputs make sense.

Run inside the Apptainer container on Vera:

    apptainer exec \\
        --bind /cephyr/users/kesaf/Vera/llm-debiasing-benchmark:/code \\
        --pwd /code \\
        $HOME/benchmarking.sif \\
        python3 /code/thesis/lib/experiments/experiment_vary_experts/test_prevalence.py
"""

import sys
sys.path.insert(0, "/code/original/lib")

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from scipy.special import expit  # sigmoid: converts log-odds back to probability

# Import the functions we want to test
from expert_class_prevalence import (
    fit_logit_intercept_only,
    fit_ppi_intercept_only,
    fit_dsl_intercept_only,
    compute_one_n,
)


def generate_fake_data(N, true_prevalence, llm_accuracy, seed=42):
    """
    Generate fake binary Y and Y_hat.

    N               : total number of rows
    true_prevalence : P(Y=1), e.g. 0.6
    llm_accuracy    : P(Y_hat == Y), e.g. 0.75
    """
    rng = np.random.default_rng(seed)

    # Expert labels: Bernoulli with true_prevalence
    Y = rng.binomial(1, true_prevalence, size=N).astype(float)

    # LLM labels: flip each label with probability (1 - llm_accuracy)
    flip = rng.binomial(1, 1 - llm_accuracy, size=N)
    Y_hat = np.where(flip, 1 - Y, Y).astype(float)

    return Y, Y_hat


# ─── Settings ─────────────────────────────────────────────────────────────────

N               = 1000     # total rows — similar to CUAD size
TRUE_PREVALENCE = 0.6      # P(Y=1) = 60% — known ground truth
LLM_ACCURACY    = 0.75     # LLM gets 75% of labels right
SEED            = 42
TEST_N_VALUES   = [20, 50, 100, 200]  # n values to test

# ─── Generate data ────────────────────────────────────────────────────────────

print("=" * 60)
print("Generating fake data...")
print(f"  N={N}, true prevalence={TRUE_PREVALENCE}, LLM accuracy={LLM_ACCURACY}")

Y, Y_hat = generate_fake_data(N, TRUE_PREVALENCE, LLM_ACCURACY, seed=SEED)

print(f"  Actual Y mean:     {Y.mean():.4f}  (should be ~{TRUE_PREVALENCE})")
print(f"  Actual Y_hat mean: {Y_hat.mean():.4f}")
print(f"  LLM accuracy:      {(Y == Y_hat).mean():.4f}  (should be ~{LLM_ACCURACY})")

# ─── Write to temporary CSV (mimics what the real script reads) ───────────────

tmp_csv = Path(tempfile.mktemp(suffix=".csv"))
pd.DataFrame({"y": Y, "y_hat": Y_hat}).to_csv(tmp_csv, index=False)
print(f"\nWrote fake data to: {tmp_csv}")

# ─── Test reference and LLM-only ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Testing theta_star and theta_llm...")

theta_star = fit_logit_intercept_only(Y)
theta_llm  = fit_logit_intercept_only(Y_hat)

print(f"  theta_star (log-odds): {float(theta_star):.4f}")
print(f"  theta_star (prob):     {expit(float(theta_star)):.4f}  (should be ~{TRUE_PREVALENCE})")
print(f"  theta_llm  (log-odds): {float(theta_llm):.4f}")
print(f"  theta_llm  (prob):     {expit(float(theta_llm)):.4f}")

# ─── Test compute_one_n for each n ────────────────────────────────────────────

print("\n" + "=" * 60)
print("Testing compute_one_n for each n value...")
print(f"{'n':>5}  {'exp (prob)':>12}  {'dsl (prob)':>12}  {'ppi (prob)':>12}  {'target':>8}")
print("-" * 60)

for n in TEST_N_VALUES:
    n_seed = SEED * 10000 + n
    beta_exp, beta_dsl, beta_ppi = compute_one_n(Y, Y_hat, n, n_seed)

    prob_exp = expit(float(beta_exp))
    prob_dsl = expit(float(beta_dsl))
    prob_ppi = expit(float(beta_ppi))

    print(f"{n:>5}  {prob_exp:>12.4f}  {prob_dsl:>12.4f}  {prob_ppi:>12.4f}  {TRUE_PREVALENCE:>8.4f}")

# ─── Sanity checks ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Sanity checks...")

# theta_star should be close to logit(true_prevalence)
import scipy.special
expected_log_odds = scipy.special.logit(TRUE_PREVALENCE)
assert abs(float(theta_star) - expected_log_odds) < 0.2, \
    f"theta_star {float(theta_star):.4f} too far from expected {expected_log_odds:.4f}"
print("  PASS: theta_star is close to logit(true_prevalence)")

# theta_star and theta_llm should be scalars stored as shape (1,)
assert theta_star.shape == (1,), f"Expected shape (1,), got {theta_star.shape}"
assert theta_llm.shape == (1,),  f"Expected shape (1,), got {theta_llm.shape}"
print("  PASS: theta_star and theta_llm have correct shape (1,)")

# compute_one_n should return 1-element arrays
beta_exp, beta_dsl, beta_ppi = compute_one_n(Y, Y_hat, 100, 999)
assert np.atleast_1d(beta_exp).shape == (1,), "beta_exp wrong shape"
assert np.atleast_1d(beta_dsl).shape == (1,), "beta_dsl wrong shape"
assert np.atleast_1d(beta_ppi).shape == (1,), "beta_ppi wrong shape"
print("  PASS: compute_one_n returns correct shapes")

# All estimates should be reasonable probabilities (between 0.2 and 0.9)
for beta, name in [(beta_exp, "exp"), (beta_dsl, "dsl"), (beta_ppi, "ppi")]:
    p = expit(float(beta))
    assert 0.2 < p < 0.9, f"{name} probability {p:.4f} is unreasonably extreme"
print("  PASS: all estimates are reasonable probabilities")

# Clean up
tmp_csv.unlink(missing_ok=True)

print("\nAll tests passed!")
