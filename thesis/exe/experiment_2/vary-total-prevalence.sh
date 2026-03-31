#!/bin/bash

#SBATCH --job-name=vary-total-prevalence
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-300
#SBATCH --time=0-00:30:00

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-total-prevalence/%x_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-total-prevalence/%x_%A_%a.err

#SBATCH --mail-user=${USER}@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-total-prevalence.sh <dataset> <llm> <n_expert>
# Example: sbatch vary-total-prevalence.sh cuad llama 50
DATASET=$1
LLM=$2
N_EXPERT=$3

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-total-prevalence/${DATASET}/${LLM}/n_${N_EXPERT}"

mkdir -p /mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-total-prevalence
mkdir -p "${CODE_DIR}/thesis/results/vary-total-prevalence/${DATASET}/${LLM}/n_${N_EXPERT}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | n_expert: ${N_EXPERT} | Rep: ${SLURM_ARRAY_TASK_ID}"

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_total/phase_1/expert_prevalence.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed     "${SLURM_ARRAY_TASK_ID}" \
        --n-expert "${N_EXPERT}"
