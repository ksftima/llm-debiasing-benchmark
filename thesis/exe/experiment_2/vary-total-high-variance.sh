#!/bin/bash

#SBATCH --job-name=vary-total-high-variance
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-300
#SBATCH --time=0-00:45:00

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-total-high-variance/%x_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-total-high-variance/%x_%A_%a.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-total-high-variance.sh <dataset> <llm> <n_expert>
# Example: sbatch vary-total-high-variance.sh cuad llama 50
DATASET=$1
LLM=$2
N_EXPERT=$3

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-total-high-variance/${DATASET}/${LLM}/n_${N_EXPERT}"

mkdir -p /mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-total-high-variance
mkdir -p "${CODE_DIR}/thesis/results/vary-total-high-variance/${DATASET}/${LLM}/n_${N_EXPERT}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | n_expert: ${N_EXPERT} | Rep: ${SLURM_ARRAY_TASK_ID}"

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_total/phase_2_and_3/expert_variance.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed     "${SLURM_ARRAY_TASK_ID}" \
        --dataset  "${DATASET}" \
        --phase    "high" \
        --n-expert "${N_EXPERT}"
