#!/bin/bash

#SBATCH --job-name=vary-expert-lpm-full
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-300
#SBATCH --time=0-01:00:00

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-lpm-full/%x_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-lpm-full/%x_%A_%a.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-expert-lpm-full.sh <dataset> <llm>
# Example: sbatch vary-expert-lpm-full.sh vuamc llama
DATASET=$1
LLM=$2

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-expert-lpm-full/${DATASET}/${LLM}"

mkdir -p /mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-lpm-full
mkdir -p "${CODE_DIR}/thesis/results/vary-expert-lpm-full/${DATASET}/${LLM}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | mode: LPM full | Rep: ${SLURM_ARRAY_TASK_ID}"

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_lpm/expert_lpm_full.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed "${SLURM_ARRAY_TASK_ID}"
