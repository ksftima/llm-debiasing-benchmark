#!/bin/bash

#SBATCH --job-name=vary-expert-apptainer
#SBATCH --account=C3SE2025-1-14
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-300
#SBATCH --time=0-00:15:00

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/logs/no-collinear-90/output_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/logs/no-collinear-90/error_%A_%a.log

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=all

set -eo pipefail

ANNOTATION=$1
DATASET=$2

# Paths for your setup
CONTAINER_PATH="$HOME/benchmarking.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"
MIMER_PATH="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use"
DATA_DIR="$MIMER_PATH/experiments/no-collinear-90/vary-num-expert/data/$DATASET/$ANNOTATION"

mkdir -p $DATA_DIR

# Run with Apptainer
apptainer exec \
    --bind ${CODE_DIR}:/code \
    --bind ${MIMER_PATH}:/mimer \
    --pwd /code \
    ${CONTAINER_PATH} \
    python /code/lib/vary_expert_realworld.py \
        "logistic" \
        "/mimer/annotations/$DATASET/annotated_$ANNOTATION.json" \
        "/mimer/experiments/no-collinear-90/vary-num-expert/data/$DATASET/$ANNOTATION/data_${SLURM_ARRAY_TASK_ID}.npz" \
        --collinear-threshold 0.90 \
        --seed "${SLURM_ARRAY_TASK_ID}"
