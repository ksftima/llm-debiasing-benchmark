#!/bin/bash

#SBATCH --job-name=vary-expert-simulation-apptainer
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
DATA_DIR="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments/vary-num-expert/simulation/data"

mkdir -p $DATA_DIR

# Run with Apptainer
apptainer exec \
    --bind ${CODE_DIR}:/code \
    --bind ${DATA_DIR}:/data \
    --pwd /code \
    ${CONTAINER_PATH} \
    python /code/lib/vary_expert_simulation.py \
        "/data/data_simulation_${SLURM_ARRAY_TASK_ID}.npz"
