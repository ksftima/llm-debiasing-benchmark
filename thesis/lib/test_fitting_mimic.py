import sys
from pathlib import Path

# Add original/lib to path to access fitting.py
lib_path = Path(__file__).parent.parent.parent / "original" / "lib"
sys.path.insert(0, str(lib_path))

import numpy as np
import scipy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import fitting


# ─── Data generation ────────────────────────────────────────────────────────

def generate(num_samples, prediction_accuracy, rng):
    M = 10
    means = np.zeros(M)
    covs = 0.3 * np.ones((M, M))
    covs = covs + 0.7 * np.eye(M)
    X = rng.multivariate_normal(means, covs, size=num_samples)
    X[:,1] = (X[:,1] > scipy.stats.norm.ppf(0.8)).astype(np.float32)
    X1, X2, X3, X4, X6 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,5]
    a = 0.1 / (1.0 + np.exp(0.5 * X3 - 0.5 * X2))
    b = 1.3 * X4 / (1.0 + np.exp(-0.1 * X2))
    c = 1.5 * X4 * X6
    d = 0.5 * X1 * X2
    e = 1.3 * X1
    f = X2
    W = a + b + c + d + e + f
    Y_true = rng.binomial(1, scipy.special.expit(W), size=num_samples).astype(float)
    X_train = np.stack([X[:,0], X[:,0] ** 2, X[:,1], X[:,3]], axis=1)
    p = rng.binomial(1, prediction_accuracy, size=num_samples)
    Y_pred = p * Y_true + (1 - p) * (1 - Y_true)
    return X_train, Y_true, Y_pred


# ─── Stratified expert selection (balanced 0s and 1s) ───────────────────────

def stratified_select(Y_true, n, rng):
    indices_ones = np.flatnonzero(Y_true)
    indices_zeros = np.flatnonzero(1 - Y_true)

    num_ones = len(indices_ones)
    num_zeros = len(indices_zeros)

    if num_ones == 0 or num_zeros == 0:
        raise ValueError("Data only has one class. Rerun.")

    if num_ones <= n // 2:
        n_ones = num_ones
        n_zeros = n - num_ones
    elif num_zeros <= n // 2:
        n_zeros = num_zeros
        n_ones = n - num_zeros
    else:
        n_ones = n // 2
        n_zeros = n // 2 + n % 2

    sel_ones = rng.choice(len(indices_ones), size=n_ones, replace=False)
    sel_zeros = rng.choice(len(indices_zeros), size=n_zeros, replace=False)
    selected_idx = np.concatenate((indices_ones[sel_ones], indices_zeros[sel_zeros]))

    selected = np.zeros(len(Y_true), dtype=bool)
    selected[selected_idx] = True

    return selected, selected_idx


# ─── Single repetition ───────────────────────────────────────────────────────

def run_one(args):
    N, n_expert, prediction_accuracy, seed = args
    rng = np.random.default_rng(seed)
    X, Y_true, Y_pred = generate(N, prediction_accuracy, rng)

    selected, selected_idx = stratified_select(Y_true, n_expert, rng)

    coeffs_all = fitting.logit_fit(X, Y_true)
    coeffs_exp = fitting.logit_fit(X[selected], Y_true[selected])
    coeffs_ppi = fitting.logit_fit_ppi(X, Y_true, Y_pred, selected)
    coeffs_dsl = fitting.logit_fit_dsl(X, Y_true, Y_pred, selected_idx)

    return coeffs_all, coeffs_exp, coeffs_ppi, coeffs_dsl


# ─── RMSE (computed per rep, then averaged across reps) ─────────────────────

def compute_rmse(coeffs_all_reps, coeffs_method_reps):
    """
    coeffs_all_reps:    (NUM_REPS, NUM_COEFFS) - upper bound per rep
    coeffs_method_reps: (NUM_REPS, NUM_COEFFS) - method per rep

    Returns mean RMSE and 95% CI across reps.
    """
    # RMSE per rep: scalar per repetition
    rmse_per_rep = np.sqrt(np.mean((coeffs_all_reps - coeffs_method_reps) ** 2, axis=1))

    mean_rmse = np.mean(rmse_per_rep)
    se = np.std(rmse_per_rep) / np.sqrt(len(rmse_per_rep))

    return mean_rmse, mean_rmse - 2 * se, mean_rmse + 2 * se


# ─── Simulation ──────────────────────────────────────────────────────────────

N = 10000
NUM_REPS = 100
PREDICTION_ACCURACY = 0.9
NUM_EXPERT_SAMPLES = np.array([200, 500, 1000, 2000, 3000])
NUM_WORKERS = 32

rmse_exp, lower_exp, upper_exp = [], [], []
rmse_ppi, lower_ppi, upper_ppi = [], [], []
rmse_dsl, lower_dsl, upper_dsl = [], [], []

for n_expert in NUM_EXPERT_SAMPLES:
    print(f"\nRunning n_expert={n_expert} ({NUM_REPS} reps, {NUM_WORKERS} workers)...")
    args = [(N, n_expert, PREDICTION_ACCURACY, 42 * 1000 + rep) for rep in range(NUM_REPS)]

    with Pool(NUM_WORKERS) as pool:
        results = pool.map(run_one, args)

    rep_all = np.array([r[0] for r in results])  # (NUM_REPS, NUM_COEFFS)
    rep_exp = np.array([r[1] for r in results])
    rep_ppi = np.array([r[2] for r in results])
    rep_dsl = np.array([r[3] for r in results])

    m, lo, hi = compute_rmse(rep_all, rep_exp)
    rmse_exp.append(m); lower_exp.append(lo); upper_exp.append(hi)

    m, lo, hi = compute_rmse(rep_all, rep_ppi)
    rmse_ppi.append(m); lower_ppi.append(lo); upper_ppi.append(hi)

    m, lo, hi = compute_rmse(rep_all, rep_dsl)
    rmse_dsl.append(m); lower_dsl.append(lo); upper_dsl.append(hi)

    print(f"  Done. RMSE — Expert: {rmse_exp[-1]:.4f}, PPI: {rmse_ppi[-1]:.4f}, DSL: {rmse_dsl[-1]:.4f}")


# ─── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
X_ticks = np.arange(len(NUM_EXPERT_SAMPLES))

for label, rmse, lower, upper, color in [
    ("Expert-only", rmse_exp, lower_exp, upper_exp, colors[0]),
    ("PPI",         rmse_ppi, lower_ppi, upper_ppi, colors[1]),
    ("DSL",         rmse_dsl, lower_dsl, upper_dsl, colors[2]),
]:
    ax.fill_between(X_ticks, np.array(lower), np.array(upper), color=color, alpha=0.2, linewidth=0)
    ax.plot(X_ticks, np.array(rmse), "o-", color=color, label=label)

ax.set_xticks(X_ticks)
ax.set_xticklabels([str(n) for n in NUM_EXPERT_SAMPLES], rotation=45)
ax.set_xlabel("Number of Expert Samples")
ax.set_ylabel("RMSE (averaged across repetitions)")
ax.set_title("Simulation: RMSE vs Number of Expert Samples")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

output_dir = Path(__file__).parent.parent / "results"
output_dir.mkdir(exist_ok=True, parents=True)
plot_path = output_dir / "test_fitting_mimic_rmse.png"
fig.savefig(str(plot_path), dpi=300)
plt.close()

print(f"\nPlot saved to: {plot_path}")
