import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from argparse import ArgumentParser


###############
# Gather data #
###############


def gather(base_dir: Path, dataset: str, annotation: str, num_exp: int):

    print(f" - {dataset}/{annotation}/{num_exp}")
    data_path = base_dir / "data" / dataset / annotation / f"n{num_exp}"

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
        "num_total_samples": data["num_total_samples"],
        "all": np.stack(coeffs_all, axis=0),
        "exp": np.stack(coeffs_exp, axis=0),
        "dsl": np.stack(coeffs_dsl, axis=0),
        "ppi": np.stack(coeffs_ppi, axis=0),
    }

###########
# Metrics #
###########

def compute_rmse(coeffs_true, coeffs_pred):

    assert len(coeffs_true.shape) == 3
    assert coeffs_true.shape == coeffs_pred.shape

    R = coeffs_true.shape[0]

    # standardsize per coeff
    error = (coeffs_true - coeffs_pred) / coeffs_true

    rmse = np.sqrt(np.mean(error ** 2, axis=(0,2)))

    sd = np.sqrt(np.mean(error ** 2, axis=2)).std(axis=0)
    std_err = sd / np.sqrt(R)
    upper = rmse + 2 * std_err
    lower = rmse - 2 * std_err

    return {
        "rmse": rmse,
        "upper": upper,
        "lower": lower,
    }

#############
# Plot RMSE #
#############

def forward(x, N, n):
    """
    Transformation from linspace(0,1) to logspace(log(200),log(N))/N
    """
    return N ** (x-1) * n ** (1-x)

def inverse(x, N, n):
    """
    Transformation from logspace(log(200),log(N))/N to linspace(0,1) to 
    """
    return 1 + np.log10(x) / np.log10(N / n)

def plot_rmse(ax, exp, dsl, ppi, n):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    X = np.linspace(0, 1, 10)

    ax.fill_between(
        X,
        exp["lower"],
        exp["upper"],
        color = colors[0],
        alpha = 0.2,
        linewidth = 0,
    )
    ax.plot(
        X,
        exp["rmse"],
        "o-",
        color = colors[0],
        label = r"$\theta_\dagger$",
    )

    ax.fill_between(
        X,
        dsl["lower"],
        dsl["upper"],
        color = colors[1],
        alpha = 0.2,
        linewidth = 0,
    )
    ax.plot(
        X,
        dsl["rmse"],
        "o-",
        color = colors[1],
        label = "DSL",
    )

    ax.fill_between(
        X,
        ppi["lower"],
        ppi["upper"],
        color = colors[2],
        alpha = 0.2,
        linewidth = 0,
    )
    ax.plot(
        X,
        ppi["rmse"],
        "o-",
        color = colors[2],
        label = "PPI",
    )

    xticklabels = [f"{x:.2f}" for x in forward(X, 10000, n)]
    for i in [1,2,4,5,7,8]:
        xticklabels[i] = ""
    ax.set_xticks(ticks = X, labels = xticklabels)


def plot_all(ax, data, n, num_reps):

    R_max = min(data[d][a]["all"].shape[0] for d in datasets for a in annotations)
    if num_reps is not None and R_max < num_reps:
        print(" - WARNING: the specified number of reps is too big. Using largest available")
        R = R_max
    elif num_reps is not None:
        R = num_reps
    else:
        R = R_max
    print(f"- number of repetitions: {R}")

    coeffs_all = []
    coeffs_exp = []
    coeffs_dsl = []
    coeffs_ppi = []

    for d in datasets:
        for a in annotations:
            N = data[d][a]["all"].shape[0]
            subsample = np.random.choice(N, R, replace = False)
            coeffs_all.append(data[d][a]["all"][subsample,:,:])
            coeffs_exp.append(data[d][a]["exp"][subsample,:,:])
            coeffs_dsl.append(data[d][a]["dsl"][subsample,:,:])
            coeffs_ppi.append(data[d][a]["ppi"][subsample,:,:])

    coeffs_all = np.concatenate(coeffs_all, axis=-1)
    coeffs_exp = np.concatenate(coeffs_exp, axis=-1)
    coeffs_dsl = np.concatenate(coeffs_dsl, axis=-1)
    coeffs_ppi = np.concatenate(coeffs_ppi, axis=-1)

    print("- computing and plotting the RMSE")
    rmse_exp = compute_rmse(coeffs_all, coeffs_exp)
    rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl)
    rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi)
    plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi, n)


def plot_dataset(ax, data, n, num_reps):

    R_max = min(data[a]["all"].shape[0] for a in annotations)
    if num_reps is not None and R_max < num_reps:
        print(" - WARNING: the specified number of reps is too big. Using largest available")
        R = R_max
    elif num_reps is not None:
        R = num_reps
    else:
        R = R_max
    print(f"- number of repetitions: {R}")

    coeffs_all = []
    coeffs_exp = []
    coeffs_dsl = []
    coeffs_ppi = []

    for a in annotations:
        N = data[a]["all"].shape[0]
        subsample = np.random.choice(N, R, replace = False)
        coeffs_all.append(data[a]["all"][subsample,:,:])
        coeffs_exp.append(data[a]["exp"][subsample,:,:])
        coeffs_dsl.append(data[a]["dsl"][subsample,:,:])
        coeffs_ppi.append(data[a]["ppi"][subsample,:,:])

    coeffs_all = np.concatenate(coeffs_all, axis=-1)
    coeffs_exp = np.concatenate(coeffs_exp, axis=-1)
    coeffs_dsl = np.concatenate(coeffs_dsl, axis=-1)
    coeffs_ppi = np.concatenate(coeffs_ppi, axis=-1)

    print("- computing and plotting the RMSE")
    ax.set_title(dataset)
    rmse_exp = compute_rmse(coeffs_all, coeffs_exp)
    rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl)
    rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi)
    plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi, n)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("base_dir", type = Path)
    parser.add_argument("--num_reps", type = int, default = None)
    args = parser.parse_args()

    plot_dir = args.base_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["amazon", "misinfo", "biobias", "germeval"]
    annotations = ["bert", "deepseek", "phi4", "claude"]
    num_exps = [200, 1000, 5000]

    xlabel = "Proportion of total samples (log)"
    ylabel = "sRMSE"

    rowsize = 3
    colsize = 4

    ##################
    # Gathering Data #
    ##################

    print("")
    print("Gathering the data")
    data = {
        n: {
            d: {
                a: gather(args.base_dir, d, a, n)
                for a in annotations
            }
            for d in datasets
        }
        for n in num_exps
    }

    #########################
    # Plot for all datasets #
    #########################

    print("")
    print("Plot for all datasets:")
    rows = 1
    cols = 3
    figsize=(cols * colsize, rows * rowsize)
    fig, ax = plt.subplots(rows, cols, figsize = figsize)
    for i, n in enumerate(num_exps):
        ax[i].set_title(f"n = {n}")
        plot_all(ax[i], data[n], n, args.num_reps)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc = "upper center",
        bbox_to_anchor = (0.5, 1),
        ncol = 3,
    )
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout(rect = [0.02, 0, 1, 0.93])
    fig.savefig(plot_dir / "rmse_all.png")
    fig.savefig(plot_dir / "rmse_all.pdf")

    ##########################
    # Plot for each datasets #
    ##########################

    for n in num_exps:
        rows = 2
        cols = 2
        figsize = (cols * colsize, rows * rowsize)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        for i, dataset in enumerate(datasets):
            print("")
            print(f"Plot for dataset {dataset}")
            plot_dataset(
                axs[i // rows, i % cols],
                data[n][dataset],
                n,
                args.num_reps,
            )
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        fig.tight_layout()
        fig.savefig(plot_dir / f"rmse_datasets_n{n}.png")
        fig.savefig(plot_dir / f"rmse_datasets_n{n}.pdf")

        print("")
