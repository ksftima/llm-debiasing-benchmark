#!/bin/bash

#SBATCH --job-name=vary-expert-high-variance-ppipp
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-300
#SBATCH --time=0-00:10:00        # PPI++ only — no DSL/R overhead

#SBATCH --output=/cephyr/users/%u/Vera/llm-debiasing-benchmark/thesis/logs/vary-expert-high-variance-ppipp/%x_%A_%a.log
#SBATCH --error=/cephyr/users/%u/Vera/llm-debiasing-benchmark/thesis/logs/vary-expert-high-variance-ppipp/%x_%A_%a.err

#SBATCH --mail-user=kesaf@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-expert-high-variance-ppipp.sh <dataset> <llm> [lam]
# Example: sbatch vary-expert-high-variance-ppipp.sh cuad llama
DATASET=$1
LLM=$2
LAM=${3:-0.01}

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-expert-high-variance-ppipp/${DATASET}/${LLM}"

mkdir -p "${CODE_DIR}/thesis/logs/vary-expert-high-variance-ppipp"
mkdir -p "${CODE_DIR}/thesis/results/vary-expert-high-variance-ppipp/${DATASET}/${LLM}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | lam: ${LAM} | Rep: ${SLURM_ARRAY_TASK_ID}"

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_2_and_3_ppipp/expert_variance_ppipp.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed    "${SLURM_ARRAY_TASK_ID}" \
        --dataset "${DATASET}" \
        --phase   "high" \
        --lam     "${LAM}"
