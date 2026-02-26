import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from argparse import ArgumentParser

###############
# Gather data #
###############

def gather(dataset, annotation):

    print(f" - {dataset}/{annotation}")

    base_path = Path("/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments/vary-num-expert")
    data_path = base_path / dataset / "data" / annotation

    if not data_path.exists():
        raise Exception(f"Error: {data_path} not found")

    try:
        next(data_path.iterdir())
    except StopIteration:
        raise Exception(f"Error: {data_path} is empty")

    coeffs_all = []
    coeffs_exp = []
    coeffs_dsl = []
    coeffs_ppi = []

    for file in data_path.iterdir():
        data = np.load(file)
        coeffs_all.append(data["coeffs_all"])
        coeffs_exp.append(data["coeffs_exp"])
        coeffs_dsl.append(data["coeffs_dsl"])
        coeffs_ppi.append(data["coeffs_ppi"])

    return {
        "num_expert_samples": data["num_expert_samples"],
        "all": np.stack(coeffs_all, axis=0),
        "exp": np.stack(coeffs_exp, axis=0),
        "dsl": np.stack(coeffs_dsl, axis=0),
        "ppi": np.stack(coeffs_ppi, axis=0),
    }

if __name__ == "__main__":

    datasets = ["amazon", "misinfo", "biobias", "germeval"]
    annotations = ["bert", "deepseek", "phi4"]

    print("")
    print("Gathering the data")
    data = {d: {a: gather(d,a) for a in annotations} for d in datasets}

    R = min(data[d][a]["all"].shape[0] for d in datasets for a in annotations)
    print(f" - minimum number of repetitions: {R}")

    D = len(datasets)
    A = len(annotations)
    num_exp = np.zeros((D, A, 10))
    size = (D, A, R, 10, 5)
    coeffs_all = np.zeros(size)
    coeffs_exp = np.zeros(size)
    coeffs_dsl = np.zeros(size)
    coeffs_ppi = np.zeros(size)

    for i, d in enumerate(datasets):
        for j, a in enumerate(annotations):
            N = data[d][a]["all"].shape[0]
            subsample = np.random.choice(N, R, replace = False)
            coeffs_all[i,j,:,:,:] = data[d][a]["all"][subsample,:,:]
            coeffs_exp[i,j,:,:,:] = data[d][a]["exp"][subsample,:,:]
            coeffs_dsl[i,j,:,:,:] = data[d][a]["dsl"][subsample,:,:]
            coeffs_ppi[i,j,:,:,:] = data[d][a]["ppi"][subsample,:,:]
            num_exp[i,j,:] = data[d][a]["num_expert_samples"]

    print("")
    print("Coefficient statistics")
    print(f" - means: {coeffs_all.mean(axis=(0,1,2,3))}")
    print(f" - stds: {coeffs_all.std(axis=(0,1,2,3))}")

    for d, dataset in enumerate(datasets):
        print("")
        print(f"Coefficient statistics for {dataset}")
        print(f" - means: {coeffs_all[d,:,:,:,:].mean(axis=(0,1,2))}")
        print(f" - stds: {coeffs_all[d,:,:,:,:].std(axis=(0,1,2))}")

    print("")
