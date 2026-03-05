#!/bin/bash

#SBATCH --job-name=plot-test-fitting
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:05:00

#SBATCH --output=plot_test_fitting_%j.log
#SBATCH --error=plot_test_fitting_%j.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=FAIL

# ====================================
# Plot test_fitting.py simulation results
# ====================================

# Paths
CONTAINER_PATH="$HOME/benchmarking.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

# You need to specify the results file
# Usage: sbatch plot-test-fitting.sh path/to/results.npz

# Get results file from command line argument or use default
if [ -z "$1" ]; then
    # Default: look for results saved by test_fitting.py
    RESULTS_FILE="${CODE_DIR}/thesis/results/test_fitting_results.npz"

    if [ ! -f "$RESULTS_FILE" ]; then
        echo "ERROR: No results file found at: $RESULTS_FILE"
        echo "Run test-fitting.sh first, or provide path as argument:"
        echo "  sbatch plot-test-fitting.sh /path/to/results.npz"
        exit 1
    fi

    echo "Using results file: $RESULTS_FILE"
else
    RESULTS_FILE="$1"
    echo "Using specified results file: $RESULTS_FILE"
fi

# Check if file exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "ERROR: Results file not found: $RESULTS_FILE"
    exit 1
fi

echo "================================"
echo "Plotting test_fitting results"
echo "Job ID: $SLURM_JOB_ID"
echo "Results file: $RESULTS_FILE"
echo "Started at: $(date)"
echo "================================"

# Output plots alongside the results file
OUTPUT_DIR=$(dirname "$RESULTS_FILE")

# Rewrite RESULTS_FILE as container path (everything under CODE_DIR maps to /code)
CONTAINER_RESULTS="/code/${RESULTS_FILE#${CODE_DIR}/}"

# Run plotting with Apptainer
apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/plot_test_fitting.py \
        "${CONTAINER_RESULTS}" \
        --output "$(dirname "${CONTAINER_RESULTS}")"

EXIT_CODE=$?

echo "================================"
echo "Plotting completed with exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Plots saved to: $OUTPUT_DIR"
    echo "  - test_fitting_rmse.png"
    echo "  - test_fitting_rmse.pdf"
else
    echo "✗ Plotting failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
