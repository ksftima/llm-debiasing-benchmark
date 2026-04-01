#!/bin/bash

#SBATCH --job-name=vary-expert-full-logistic-ppipp
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-300
#SBATCH --time=0-00:20:00        # PPI++ only — no DSL/R overhead

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-full-logistic-ppipp/%x_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-full-logistic-ppipp/%x_%A_%a.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-expert-full-logistic-ppipp.sh <dataset> <llm> [lam]
# Example: sbatch vary-expert-full-logistic-ppipp.sh cuad llama
DATASET=$1
LLM=$2
LAM=${3:-0.01}

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-expert-full-logistic-ppipp/${DATASET}/${LLM}"

mkdir -p /mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-full-logistic-ppipp
mkdir -p "${CODE_DIR}/thesis/results/vary-expert-full-logistic-ppipp/${DATASET}/${LLM}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | lam: ${LAM} | Rep: ${SLURM_ARRAY_TASK_ID}"

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_4_ppipp/expert_full_logistic_ppipp.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed "${SLURM_ARRAY_TASK_ID}" \
        --lam  "${LAM}"
