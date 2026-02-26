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

import fitting as fit

@dataclass
class SampleParams:
    data: pd.DataFrame
    model: str
    features: set[str]
    n: int

def compute_coeffs(params: SampleParams):

    selected = np.random.choice(len(params.data), size=params.n, replace=False)

    X = params.data[list(params.features)].to_numpy()
    Y = params.data["y"].to_numpy().astype(float)
    Y_hat = params.data["y_hat"].to_numpy()

    if params.model == "linear":
        coeffs_all = fit.linear_fit(X, Y)
        coeffs_exp = fit.linear_fit(X[selected], Y[selected])
        coeffs_dsl = fit.linear_fit_dsl(X, Y, Y_hat, selected)
        coeffs_ppi = fit.linear_fit_ppi(X, Y, Y_hat, selected)
    elif params.model == "logistic":
        coeffs_all = fit.logit_fit(X, Y)
        coeffs_exp = fit.logit_fit(X[selected], Y[selected])
        coeffs_dsl = fit.logit_fit_dsl(X, Y, Y_hat, selected)
        coeffs_ppi = fit.logit_fit_ppi(X, Y, Y_hat, selected)
    else:
        raise Exception(f"model {params.model} is not supported")

    return coeffs_all, coeffs_exp, coeffs_dsl, coeffs_ppi

def simulate(
        data: pd.DataFrame,
        model: str,
        features: set[str],
        num_data_points: int,
        min_expert_samples: int,
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

    num_expert_samples = np.logspace(
        start = np.log10(min_expert_samples), # too low = convergence issues
        stop = np.log10(len(data)),
        num = num_data_points,
        base = 10.0,
    )
    results["num_expert_samples"] = num_expert_samples

    params = []
    for n in np.round(num_expert_samples).astype(int):
        params.append(SampleParams(
            data = data,
            model = model,
            features = features,
            n = n,
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
    parser.add_argument("model", type = str)
    parser.add_argument("annotated_path", type = Path)
    parser.add_argument("results_path", type = Path)
    parser.add_argument("--seed", type = int)
    parser.add_argument("--collinear-threshold", type = float)
    parser.add_argument("--centered", action = "store_true")
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
            print(" -", x, y, r**2)
            if args.collinear_threshold < r**2:
                to_remove.add(y)
        print("REMOVED:", list(to_remove))
        data = data.drop(columns=list(to_remove))
        features = features - to_remove

    if args.centered:
        print("Centering the data")
        for feature in features:
            data[feature] = data[feature] - data[feature].mean()


    print("Running the experiment")
    results = simulate(
        data = data,
        model = args.model,
        features = features,
        num_data_points = 10,
        min_expert_samples = 200,
        num_cores = num_cores,
    )

    print(f"Saving results to {args.results_path}")
    np.savez(args.results_path, **results)
