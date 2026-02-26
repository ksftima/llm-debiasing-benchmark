#!/bin/bash

#SBATCH --job-name=test-fitting
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:05:00

#SBATCH --output=test_fitting_%j.log
#SBATCH --error=test_fitting_%j.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=FAIL

# Paths for your setup
CONTAINER_PATH="$HOME/benchmarking.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

echo "================================"
echo "Testing test_fitting.py methods"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "================================"

# Run test with Apptainer
apptainer exec \
    --bind ${CODE_DIR}:/repo \
    --pwd /repo \
    ${CONTAINER_PATH} \
    python3 /repo/thesis/test_fitting.py

EXIT_CODE=$?

echo "================================"
echo "Test completed with exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "✗ Tests failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
