import sys
from pathlib import Path

# Add original/lib directory to Python path
lib_path = Path(__file__).parent.parent.parent / "original" / "lib"
sys.path.insert(0, str(lib_path))

import numpy as np
import scipy
from multiprocessing import Pool, cpu_count
from fitting import logit_fit, logit_fit_ppi, logit_fit_dsl


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


def run_one(args):
    N, n_expert, prediction_accuracy, seed = args
    rng = np.random.default_rng(seed)
    X, Y_true, Y_pred = generate(N, prediction_accuracy, rng)

    selected_idx = rng.choice(N, size=n_expert, replace=False)
    selected = np.zeros(N, dtype=bool)
    selected[selected_idx] = True

    coeffs_all = logit_fit(X, Y_true)
    coeffs_exp = logit_fit(X[selected], Y_true[selected])
    coeffs_ppi = logit_fit_ppi(X, Y_true, Y_pred, selected)
    coeffs_dsl = logit_fit_dsl(X, Y_true, Y_pred, selected_idx)

    return coeffs_all, coeffs_exp, coeffs_ppi, coeffs_dsl


N = 10000
NUM_REPS = 100
PREDICTION_ACCURACY = 0.9
NUM_EXPERT_SAMPLES = np.array([200, 500, 1000, 2000, 3000])
NUM_WORKERS = 32

all_coeffs_all = []
all_coeffs_exp = []
all_coeffs_ppi = []
all_coeffs_dsl = []

for n_expert in NUM_EXPERT_SAMPLES:
    print(f"\nRunning n_expert={n_expert} ({NUM_REPS} reps, {NUM_WORKERS} workers)...")
    args = [(N, n_expert, PREDICTION_ACCURACY, 42 * 1000 + rep) for rep in range(NUM_REPS)]

    with Pool(NUM_WORKERS) as pool:
        results = pool.map(run_one, args)

    rep_all = [r[0] for r in results]
    rep_exp = [r[1] for r in results]
    rep_ppi = [r[2] for r in results]
    rep_dsl = [r[3] for r in results]
    print(f"  {NUM_REPS}/{NUM_REPS} done")

    all_coeffs_all.append(np.mean(rep_all, axis=0))
    all_coeffs_exp.append(np.mean(rep_exp, axis=0))
    all_coeffs_ppi.append(np.mean(rep_ppi, axis=0))
    all_coeffs_dsl.append(np.mean(rep_dsl, axis=0))

output_path = Path(__file__).parent.parent / "results" / "test_fitting_results.npz"
output_path.parent.mkdir(exist_ok=True, parents=True)

np.savez(
    output_path,
    coeffs_all=np.array(all_coeffs_all),
    coeffs_exp=np.array(all_coeffs_exp),
    coeffs_ppi=np.array(all_coeffs_ppi),
    coeffs_dsl=np.array(all_coeffs_dsl),
    num_expert_samples=NUM_EXPERT_SAMPLES,
    N=np.array(N),
)

print(f"\nResults saved to: {output_path}")
