#!/bin/bash

#SBATCH --job-name=vary-expert-thesis
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        # no GPU needed, 4 CPUs is enough for DSL's R process
#SBATCH --array=1-300            # 300 repetitions, each with a different random seed
#SBATCH --time=0-01:00:00        # 1 hour per job should be plenty for 10 n-values

#SBATCH --output=/cephyr/users/kesaf/Vera/llm-debiasing-benchmark/thesis/logs/vary-expert/%x_%A_%a.log
#SBATCH --error=/cephyr/users/kesaf/Vera/llm-debiasing-benchmark/thesis/logs/vary-expert/%x_%A_%a.err

#SBATCH --mail-user=gusfatike@student.gu.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# --- Arguments ---
# $1 = dataset name, e.g. "cuad"
# $2 = LLM name, e.g. "llama", "deepseek", "gpt54", "mistral"
DATASET=$1
LLM=$2

# --- Paths ---
CONTAINER_PATH="$HOME/benchmarking.sif"
CODE_DIR="/cephyr/users/kesaf/Vera/llm-debiasing-benchmark"

# Input: the annotated CSV for this dataset + LLM combination
ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"

# Output: one .npz file per repetition, stored under results/vary-expert/dataset/llm/
OUTPUT_DIR="/code/thesis/results/vary-expert/${DATASET}/${LLM}"

# Create the log and output directories if they don't exist
mkdir -p "${CODE_DIR}/thesis/logs/vary-expert"
mkdir -p "${CODE_DIR}/thesis/results/vary-expert/${DATASET}/${LLM}"

# --- Run inside the container ---
# --bind mounts CODE_DIR as /code inside the container
# --pwd sets the working directory inside the container
apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/vary_expert.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed "${SLURM_ARRAY_TASK_ID}"
