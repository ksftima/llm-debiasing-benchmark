import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from argparse import ArgumentParser


###############
# Gather data #
###############


def gather(base_dir: Path, dataset: str, annotation: str):

    print(f" - {dataset}/{annotation}")
    data_path = base_dir / "data" / dataset / annotation

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


###########
# Metrics #
###########


def compute_bias(coeffs_true, coeffs_pred):
    assert len(coeffs_true.shape) == 3
    assert coeffs_true.shape == coeffs_pred.shape
    error = (coeffs_true - coeffs_pred) / coeffs_true
    bias = np.mean(error, axis = (0,2))
    num_repetitions = coeffs_true.shape[0]
    std_err = np.std(error, axis = (0,2)) / np.sqrt(num_repetitions)
    upper = bias + 2 * std_err
    lower = bias - 2 * std_err
    return bias, upper, lower


def compute_rmse(coeffs_true, coeffs_pred):
    assert len(coeffs_true.shape) == 3
    assert coeffs_true.shape == coeffs_pred.shape
    error = (coeffs_true - coeffs_pred) / coeffs_true # standardsize per coeff
    rmse = np.sqrt(np.mean(error ** 2, axis=(0,2)))
    num_repetitions = coeffs_true.shape[0]
    sd = np.sqrt(np.mean(error ** 2, axis=2)).std(axis=0)
    std_err = sd / np.sqrt(num_repetitions)
    upper = rmse + 2 * std_err
    lower = rmse - 2 * std_err
    return {
        "rmse": rmse,
        "upper": upper,
        "lower": lower,
    }


########################
# Axis Transformations #
########################


def forward(x, N):
    """
    Transformation from linspace(0,1) to logspace(log(200),log(N))/N
    """
    return N**(x-1) * 200**(1-x)


def inverse(x, N):
    """
    Transformation from logspace(log(200),log(N))/N to linspace(0,1) to 
    """
    return 1 + np.log10(x) / np.log10(N / 200)


#############
# Plot RMSE #
#############


def plot_rmse(ax, exp, dsl, ppi):

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

    xticklabels = [f"{x:.2f}" for x in forward(X, 10000)]
    for i in [1,2,4,5,7,8]:
        xticklabels[i] = ""
    ax.set_xticks(ticks = X, labels = xticklabels)
    ax.legend()


def plot_all(ax, data, num_reps):

    R_max = min(data[d][a]["all"].shape[0] for d in datasets for a in annotations)
    if num_reps is not None and R_max < num_reps:
        print(f" - WARNING: not enough repetitions, using max available")
        R = R_max
    elif num_reps is not None:
        R = num_reps
    else:
        R = R_max
    
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

    rmse_exp = compute_rmse(coeffs_all, coeffs_exp)
    rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl)
    rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi)
    plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi)

    return R


def plot_dataset(ax, data, num_reps):

    R_max = min(data[a]["all"].shape[0] for a in annotations)
    if num_reps is not None and R_max < num_reps:
        print(f" - WARNING: not enough repetitions, using max available")
        R = R_max
    elif num_reps is not None:
        R = num_reps
    else:
        R = R_max

    coeffs_all = []
    coeffs_exp = []
    coeffs_dsl = []
    coeffs_ppi = []

    for i, a in enumerate(annotations):
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

    rmse_exp = compute_rmse(coeffs_all, coeffs_exp)
    rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl)
    rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi)
    plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi)

    return R


def plot_annotation(ax, data, title):

    R = data["all"].shape[0]

    coeffs_all = data["all"]
    coeffs_exp = data["exp"]
    coeffs_dsl = data["dsl"]
    coeffs_ppi = data["ppi"]

    ax.set_title(title)
    rmse_exp = compute_rmse(coeffs_all, coeffs_exp)
    rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl)
    rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi)
    plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi)

    return R


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("base_dir", type = Path)
    parser.add_argument("--num_reps", type = int, default = None)
    args = parser.parse_args()

    plot_dir = args.base_dir / "plot"
    plot_dir.mkdir(exist_ok=True, parents=True)

    datasets = ["amazon", "misinfo", "biobias", "germeval"]
    annotations = ["bert", "deepseek", "phi4", "claude"]

    xlabel = "Proportion of expert samples (log)"
    ylabel = "sRMSE"

    rowsize = 3
    colsize = 5

    ###############
    # Gather data #
    ###############

    print("")
    print("Gathering the data")
    data = {
        d: {a: gather(args.base_dir, d, a) for a in annotations}
        for d in datasets
    }

    #########################
    # Plot for all datasets #
    #########################

    fig, ax = plt.subplots(figsize=(colsize, rowsize))
    R = plot_all(ax, data, args.num_reps)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(plot_dir / "rmse_all.png")
    fig.savefig(plot_dir / "rmse_all.pdf")
    print("")
    print(f"Plot for all datasets (repetitions: {R})")

    ##########################
    # Plot for each datasets #
    ##########################

    print("")
    print("Plots for dataset:")
    rows = 2
    cols = 2
    titles = {
        "amazon": "Multi-domain Sentiment",
        "misinfo": "Misinfo-general",
        "biobias": "Bias in Biographies",
        "germeval": "Germeval18",
    }
    figsize = (cols * colsize, rows * rowsize)
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i, dataset in enumerate(datasets):
        ax = axs[i // rows, i % cols]
        ax.set_title(titles[dataset])
        R = plot_dataset(
            ax,
            data[dataset],
            args.num_reps,
        )
        print(f" - {dataset} (repetitions: {R})")
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(plot_dir / f"rmse_datasets.png")
    fig.savefig(plot_dir / f"rmse_datasets.pdf")

    ########################################
    # Plot for each dataset and annotation #
    ########################################

    print("")
    print("Plots for dataset/annotation:")
    rows = len(dataset)
    cols = len(annotations)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * colsize, rows * rowsize))
    for i, dataset in enumerate(datasets):
        for j, annotation in enumerate(annotations):
            R = plot_annotation(
                axs[i, j],
                data[dataset][annotation],
                f"{dataset}/{annotation}",
            )
            print(f" - {dataset}/{annotation} (repetitions: {R})")
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(plot_dir / f"rmse_annotations.png")
    fig.savefig(plot_dir / f"rmse_annotations.pdf")

    print("")
