#!/usr/bin/env python3
"""
Plot results from test_fitting.py simulation
Shows RMSE comparison between Expert-only, DSL, and PPI methods
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser


def compute_rmse(coeffs_true, coeffs_pred):
    """
    Compute Root Mean Squared Error between true and predicted coefficients
    """
    assert len(coeffs_true.shape) == 2  # (num_samples, num_coeffs)
    assert coeffs_true.shape == coeffs_pred.shape

    # Relative error per coefficient
    error = (coeffs_true - coeffs_pred) / coeffs_true

    # RMSE across all coefficients
    rmse = np.sqrt(np.mean(error ** 2, axis=1))

    # Standard error for confidence intervals
    std_err = np.std(error, axis=1) / np.sqrt(coeffs_true.shape[1])
    upper = rmse + 2 * std_err
    lower = rmse - 2 * std_err

    return {
        "rmse": rmse,
        "upper": upper,
        "lower": lower,
    }


def forward(x, N):
    """
    Transform from linear [0,1] to log space
    """
    return N**(x-1) * 200**(1-x)


def plot_rmse(ax, results, num_expert_samples, N=10000):
    """
    Plot RMSE comparison with confidence intervals
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # X axis: proportion of expert samples (log scale)
    X = num_expert_samples / N
    X_normalized = np.log10(num_expert_samples / 200) / np.log10(N / 200)

    # Compute RMSE for each method
    rmse_exp = compute_rmse(results["coeffs_all"], results["coeffs_exp"])
    rmse_dsl = compute_rmse(results["coeffs_all"], results["coeffs_dsl"])
    rmse_ppi = compute_rmse(results["coeffs_all"], results["coeffs_ppi"])

    # Plot Expert-only
    ax.fill_between(
        X_normalized,
        rmse_exp["lower"],
        rmse_exp["upper"],
        color=colors[0],
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(
        X_normalized,
        rmse_exp["rmse"],
        "o-",
        color=colors[0],
        label="Expert-only",
    )

    # Plot DSL
    ax.fill_between(
        X_normalized,
        rmse_dsl["lower"],
        rmse_dsl["upper"],
        color=colors[1],
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(
        X_normalized,
        rmse_dsl["rmse"],
        "o-",
        color=colors[1],
        label="DSL",
    )

    # Plot PPI
    ax.fill_between(
        X_normalized,
        rmse_ppi["lower"],
        rmse_ppi["upper"],
        color=colors[2],
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(
        X_normalized,
        rmse_ppi["rmse"],
        "o-",
        color=colors[2],
        label="PPI",
    )

    # Format x-axis with actual sample counts
    xticklabels = [f"{int(x)}" for x in num_expert_samples]
    ax.set_xticks(ticks=X_normalized, labels=xticklabels, rotation=45)
    ax.set_xlabel("Number of Expert Samples")
    ax.set_ylabel("Standardized RMSE")
    ax.set_title("Simulation Test: Expert-only vs DSL vs PPI")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    parser = ArgumentParser(description="Plot test_fitting.py simulation results")
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to .npz results file from test_fitting.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as results_file)",
    )
    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = np.load(args.results_file)

    print("\nResults contain:")
    for key in results.files:
        print(f"  - {key}: {results[key].shape}")

    # Set output directory
    if args.output is None:
        output_dir = args.results_file.parent
    else:
        output_dir = args.output
        output_dir.mkdir(exist_ok=True, parents=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_rmse(
        ax,
        results,
        results["num_expert_samples"],
        N=results["coeffs_all"].shape[0],  # Total samples
    )

    fig.tight_layout()

    # Save plot
    plot_path_png = output_dir / "test_fitting_rmse.png"
    plot_path_pdf = output_dir / "test_fitting_rmse.pdf"

    fig.savefig(plot_path_png, dpi=300)
    fig.savefig(plot_path_pdf)

    print(f"\nPlots saved:")
    print(f"  - {plot_path_png}")
    print(f"  - {plot_path_pdf}")

    plt.close()


if __name__ == "__main__":
    main()
