#!/bin/bash

#SBATCH --job-name=vary-expert-full-logistic
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-300
#SBATCH --time=0-01:30:00        # full logistic with 5 features is slower

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-full-logistic/%x_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-full-logistic/%x_%A_%a.err

#SBATCH --mail-user=${USER}@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-expert-full-logistic.sh <dataset> <llm> [lam] [n_select]
# Example: sbatch vary-expert-full-logistic.sh cuad llama
#          sbatch vary-expert-full-logistic.sh misogynistic llama 0.1 "26 72 200"
DATASET=$1
LLM=$2
LAM=${3:-0.01}
N_SELECT=${4:-""}

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

LAM_SUFFIX=$([ "$LAM" = "0.01" ] && echo "" || echo "_lam$(echo $LAM | tr -d '.')")

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-expert-full-logistic${LAM_SUFFIX}/${DATASET}/${LLM}"

mkdir -p /mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-full-logistic
mkdir -p "${CODE_DIR}/thesis/results/vary-expert-full-logistic${LAM_SUFFIX}/${DATASET}/${LLM}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | lam: ${LAM} | n_select: ${N_SELECT:-full} | Rep: ${SLURM_ARRAY_TASK_ID}"

N_SELECT_ARG=$([ -z "$N_SELECT" ] && echo "" || echo "--n-select ${N_SELECT}")

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_4/expert_full_logistic.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed "${SLURM_ARRAY_TASK_ID}" \
        --lam  "${LAM}" \
        ${N_SELECT_ARG}
