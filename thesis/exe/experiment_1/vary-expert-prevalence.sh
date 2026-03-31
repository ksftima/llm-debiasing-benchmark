#!/bin/bash

#SBATCH --job-name=vary-expert-prevalence
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2        # only numpy, no GPU or R needed — 2 CPUs is enough
#SBATCH --array=1-300            # 300 repetitions, seed = SLURM_ARRAY_TASK_ID
#SBATCH --time=0-00:15:00        # prevalence is just means, each job finishes in seconds

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-prevalence/%x_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-prevalence/%x_%A_%a.err

#SBATCH --mail-user=${USER}@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# --- Arguments ---
# Usage: sbatch vary-expert-prevalence.sh <dataset> <llm>
# Example: sbatch vary-expert-prevalence.sh cuad llama
DATASET=$1
LLM=$2

# --- Paths ---
CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

# Input CSV: the annotated file for this dataset + LLM combination
ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"

# Output: one .npz per repetition, organised by dataset and LLM
OUTPUT_DIR="/code/thesis/results/vary-expert-prevalence/${DATASET}/${LLM}"

# Create log and output directories (outside container, using real path)
mkdir -p /mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-prevalence
mkdir -p "${CODE_DIR}/thesis/results/vary-expert-prevalence/${DATASET}/${LLM}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | Rep: ${SLURM_ARRAY_TASK_ID}"

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_1/expert_class_prevalence.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed "${SLURM_ARRAY_TASK_ID}"
