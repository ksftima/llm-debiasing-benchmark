#!/bin/bash

#SBATCH --job-name=vary-expert-high-variance
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        # sklearn + R DSL — 4 CPUs
#SBATCH --array=1-300            # 300 repetitions, seed = SLURM_ARRAY_TASK_ID
#SBATCH --time=0-00:45:00

#SBATCH --output=/cephyr/users/kesaf/Vera/llm-debiasing-benchmark/thesis/logs/vary-expert-high-variance/%x_%A_%a.log
#SBATCH --error=/cephyr/users/kesaf/Vera/llm-debiasing-benchmark/thesis/logs/vary-expert-high-variance/%x_%A_%a.err

#SBATCH --mail-user=gusfatike@student.gu.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-expert-high-variance.sh <dataset> <llm>
# Example: sbatch vary-expert-high-variance.sh cuad llama
DATASET=$1
LLM=$2

CONTAINER_PATH="$HOME/benchmarking.sif"
CODE_DIR="/cephyr/users/kesaf/Vera/llm-debiasing-benchmark"

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-expert-high-variance/${DATASET}/${LLM}"

mkdir -p "${CODE_DIR}/thesis/logs/vary-expert-high-variance"
mkdir -p "${CODE_DIR}/thesis/results/vary-expert-high-variance/${DATASET}/${LLM}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | Rep: ${SLURM_ARRAY_TASK_ID}"

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_2_and_3/expert_low_variance.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed    "${SLURM_ARRAY_TASK_ID}" \
        --dataset "${DATASET}" \
        --phase   "high"
