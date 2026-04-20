#!/bin/bash

#SBATCH --job-name=vary-total-lpm-full
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-300
#SBATCH --time=0-01:00:00

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-total-lpm-full/%x_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-total-lpm-full/%x_%A_%a.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-total-lpm-full.sh <dataset> <llm> <n_expert>
# Example: sbatch vary-total-lpm-full.sh vuamc llama 50
DATASET=$1
LLM=$2
N_EXPERT=$3

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-total-lpm-full/${DATASET}/${LLM}/n_${N_EXPERT}"

mkdir -p /mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-total-lpm-full
mkdir -p "${CODE_DIR}/thesis/results/vary-total-lpm-full/${DATASET}/${LLM}/n_${N_EXPERT}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | n_expert: ${N_EXPERT} | mode: LPM full | Rep: ${SLURM_ARRAY_TASK_ID}"

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_total/phase_lpm/expert_lpm_full.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed     "${SLURM_ARRAY_TASK_ID}" \
        --n-expert "${N_EXPERT}"
