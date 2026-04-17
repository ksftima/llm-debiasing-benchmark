#!/bin/bash

#SBATCH --job-name=vary-expert-lpm-high
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-300
#SBATCH --time=0-00:45:00

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-lpm-high/%x_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-lpm-high/%x_%A_%a.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-expert-lpm-high.sh <dataset> <llm>
# Example: sbatch vary-expert-lpm-high.sh cuad llama
DATASET=$1
LLM=$2

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-expert-lpm-high/${DATASET}/${LLM}"

mkdir -p /mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-lpm-high
mkdir -p "${CODE_DIR}/thesis/results/vary-expert-lpm-high/${DATASET}/${LLM}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | mode: LPM | phase: high | Rep: ${SLURM_ARRAY_TASK_ID}"

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_lpm/expert_lpm.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed    "${SLURM_ARRAY_TASK_ID}" \
        --dataset "${DATASET}" \
        --phase   "high"
