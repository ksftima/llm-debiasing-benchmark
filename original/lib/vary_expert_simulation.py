from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os
import random
import scipy
import time

from fitting import fit, fit_dsl, fit_ppi

def generate(num_samples, prediction_accuracy):
    """
    Generate the simulated data (following Appendix E)
    """
    # number of covariates
    M = 10

    # Generate the covariates
    means = np.zeros(M)
    covs = 0.3 * np.ones((M, M))
    covs = covs + 0.7 * np.eye(M)
    X = np.random.multivariate_normal(means, covs, size=num_samples)

    # Binarize X_i2
    X[:,1] = (X[:,1] > scipy.stats.norm.ppf(0.8)).astype(np.float32)

    # Generate the true outcome Y as a function of the covariates
    X1, X2, X3, X4, X6 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,5]
    a = 0.1 / (1.0 + np.exp(0.5 * X3 - 0.5 * X2))
    b = 1.3 * X4 / (1.0 + np.exp(-0.1 * X2))
    c = 1.5 * X4 * X6
    d = 0.5 * X1 * X2
    e = 1.3 * X1
    f = X2
    W = a + b + c + d + e + f
    Y_true = np.random.binomial(1, scipy.special.expit(W), size=num_samples)
    Y_true = Y_true.astype(float)

    # Make training data from true covariates
    X_train = np.stack([
        X[:,0],
        X[:,0] ** 2,
        X[:,1],
        X[:,3],
    ], axis=1)

    # Simulate the LLM annotations
    p = np.random.binomial(1, prediction_accuracy, size=num_samples)
    Y_pred = p * Y_true + (1 - p) * (1 - Y_true)

    return X_train, Y_true, Y_pred

@dataclass
class SampleParams:
    N: int # total number of samples
    n: int # number of expert-annotated amples
    pq: float # prediction accuracy
    seed: int # random state


def compute_coeffs(params: SampleParams):

    selected = np.random.choice(params.N, size=params.n, replace=False)

    X, Y, Y_hat = generate(params.N, params.pq)

    coeffs_all = fit(X, Y, params.seed)
    coeffs_exp = fit(X[selected], Y[selected], params.seed)
    coeffs_dsl = fit_dsl(X, Y, Y_hat, selected, params.seed)
    coeffs_ppi = fit_ppi(X, Y, Y_hat, selected)

    return coeffs_all, coeffs_exp, coeffs_dsl, coeffs_ppi


def simulate(
        num_total_samples: int,
        num_data_points: int,
        min_expert_samples: int,
        prediction_accuracy: float,
        num_coefficients: int,
        num_cores: int,
        seed: int,
    ):

    size = (num_data_points, num_coefficients)
    results = {
        "coeffs_all": np.zeros(size),
        "coeffs_exp": np.zeros(size),
        "coeffs_dsl": np.zeros(size),
        "coeffs_ppi": np.zeros(size),
    }

    num_expert_samples = np.logspace(
        start = np.log10(min_expert_samples), # too low = convergence issues
        stop = np.log10(num_total_samples),
        num = num_data_points,
        base = 10.0,
    )
    results["num_expert_samples"] = num_expert_samples

    params = []
    for n in np.round(num_expert_samples).astype(int):
        params.append(SampleParams(
            N = num_total_samples,
            n = n,
            pq = prediction_accuracy,
            seed = seed,
        ))

    with ProcessPoolExecutor(max_workers = num_cores) as executor:
        for i, coeffs in enumerate(executor.map(compute_coeffs, params)):
            print(f"Computed data point ({i})")
            results["coeffs_all"][i,:] = coeffs[0]
            results["coeffs_exp"][i,:] = coeffs[1]
            results["coeffs_dsl"][i,:] = coeffs[2]
            results["coeffs_ppi"][i,:] = coeffs[3]

    return results


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("results_path", type = Path)
    parser.add_argument("--seed", type = int)
    args = parser.parse_args()

    if "SLURM_CPUS_ON_NODE" in os.environ:
        num_cores = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        num_cores = 11
    print(f"Using {num_cores} cores")

    if args.seed is None:
        print("The seed was not provided, using current system time")
        seed = time.time()
    else:
        print(f"Using seed = {args.seed}")
        seed = args.seed

    random.seed(args.seed)
    np.random.seed(args.seed)

    results = simulate(
        num_total_samples = 10000,
        num_data_points = 10,
        min_expert_samples = 200,
        prediction_accuracy = 0.9,
        num_coefficients = 5, # 4 Xs + intercept
        num_cores = num_cores,
        seed = seed,
    )

    print(f"Saving results to {args.results_path}")
    np.savez(args.results_path, **results)
