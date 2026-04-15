#!/bin/bash

#SBATCH --job-name=vary-expert-full-logistic-noreg
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-300
#SBATCH --time=0-01:30:00        # full logistic with 5 features is slower

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-full-logistic-noreg/%x_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-full-logistic-noreg/%x_%A_%a.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-expert-full-logistic-noreg.sh <dataset> <llm> [n_select]
# Example: sbatch vary-expert-full-logistic-noreg.sh cuad llama
#          sbatch vary-expert-full-logistic-noreg.sh misogynistic llama "26 72 200"
DATASET=$1
LLM=$2
N_SELECT=${3:-""}

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-expert-full-logistic-noreg/${DATASET}/${LLM}"

mkdir -p /mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-full-logistic-noreg
mkdir -p "${CODE_DIR}/thesis/results/vary-expert-full-logistic-noreg/${DATASET}/${LLM}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | mode: unregularized | n_select: ${N_SELECT:-full} | Rep: ${SLURM_ARRAY_TASK_ID}"

N_SELECT_ARG=$([ -z "$N_SELECT" ] && echo "" || echo "--n-select ${N_SELECT}")

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_4/expert_full_logistic_noreg.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed "${SLURM_ARRAY_TASK_ID}" \
        ${N_SELECT_ARG}
