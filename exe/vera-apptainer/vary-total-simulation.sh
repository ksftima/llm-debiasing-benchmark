#!/bin/bash

#SBATCH --job-name=vary-total-simulation-apptainer
#SBATCH --account=C3SE2025-1-14
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-500
#SBATCH --time=0-00:15:00

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=all
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Paths for your setup
CONTAINER_PATH="$HOME/benchmarking.sif"
CODE_DIR="/cephyr/users/kesaf/Vera/llm-debiasing-benchmark"
BASE_DATA_DIR="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments/vary-num-total/simulation/data"

mkdir -p $BASE_DATA_DIR

echo "Experiment: vary number of total samples (simulation)"
num_expert=(200 1000 5000)
for n in "${num_expert[@]}"; do
    DATA_DIR_N="$BASE_DATA_DIR/n$n"
    mkdir -p $DATA_DIR_N

    # Run with Apptainer
    apptainer exec \
        --bind ${CODE_DIR}:/code \
        --bind ${BASE_DATA_DIR}:/data \
        --pwd /code \
        ${CONTAINER_PATH} \
        python /code/lib/vary_total_simulation.py \
            "/data/n$n/data_simulation_${SLURM_ARRAY_TASK_ID}.npz" \
            "$n"
done
