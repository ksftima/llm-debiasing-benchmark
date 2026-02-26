import numpy as np
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from argparse import ArgumentParser
import itertools

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from fitting import logit_fit, logit_fit_dsl, logit_fit_ppi

@dataclass
class SampleParams:
    data: pd.DataFrame # (x1, x2, x3, y, y_hat)
    features: set[str]
    N: int
    n: int

def compute_coeffs(params: SampleParams):

    X = params.data[list(params.features)].to_numpy()
    Y = params.data["y"].to_numpy().astype(float)
    Y_hat = params.data["y_hat"].to_numpy()
    coeffs_all = logit_fit(X, Y)

    samples = np.random.choice(len(params.data), size=params.N, replace=False)
    selected = np.random.choice(params.N, size=params.n, replace=False)
    X = X[samples]
    Y = Y[samples]
    Y_hat = Y_hat[samples]

    coeffs_exp = logit_fit(X[selected], Y[selected])
    coeffs_dsl = logit_fit_dsl(X, Y, Y_hat, selected)
    coeffs_ppi = logit_fit_ppi(X, Y, Y_hat, selected)

    return coeffs_all, coeffs_exp, coeffs_dsl, coeffs_ppi

def simulate(
        data: pd.DataFrame,
        features: set[str],
        num_expert_samples: int,
        num_data_points: int,
        max_total_samples: int,
        num_cores: int,
    ):

    num_coefficients = len(features) + 1 # features + intercept
    size = (num_data_points, num_coefficients)
    results = {
        "coeffs_all": np.zeros(size),
        "coeffs_exp": np.zeros(size),
        "coeffs_dsl": np.zeros(size),
        "coeffs_ppi": np.zeros(size),
    }

    # Generate the compute arguments
    num_total_samples = np.logspace(
        start = np.log10(num_expert_samples), # too low = convergence issues
        stop = np.log10(min(max_total_samples, len(data))),
        num = num_data_points,
        base = 10.0,
    )
    results["num_total_samples"] = num_total_samples

    params = []
    for N in np.round(num_total_samples).astype(int):
        params.append(SampleParams(
            data = data,
            features = features,
            N = N,
            n = num_expert_samples,
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
    parser.add_argument("num_expert", type = int)
    parser.add_argument("annotated_path", type = Path)
    parser.add_argument("results_path", type = Path)
    parser.add_argument("--seed", type = int)
    parser.add_argument("--collinear-threshold", type = float)
    args = parser.parse_args()

    if "SLURM_CPUS_ON_NODE" in os.environ:
        num_cores = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        num_cores = 11
    print(f"Using {num_cores} cores")

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using seed = {args.seed}")
    else:
        print("The seed was not provided, using current system time")

    print("Reading the data")
    data = pd.read_json(args.annotated_path)

    # Features assumed to be in the data table
    features = set(["x1", "x2", "x3", "x4"])

    if args.collinear_threshold:
        print("Removing collinear features above threhsold")
        to_remove = set()
        for x, y in itertools.combinations(features, 2):
            r = data[x].corr(data[y], method = "pearson")
            if args.collinear_threshold < r**2:
                to_remove.add(y)
        data = data.drop(columns=list(to_remove))
        features = features - to_remove

    print("Running the experiment")
    results = simulate(
        data = data,
        features = features,
        num_expert_samples = args.num_expert,
        num_data_points = 10,
        max_total_samples = 10000,
        num_cores = num_cores,
    )

    print(f"Saving results to {args.results_path}")
    np.savez(args.results_path, **results)
