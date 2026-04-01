#!/bin/bash

#SBATCH --job-name=vary-expert-high-variance
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-300            # 300 repetitions, seed = SLURM_ARRAY_TASK_ID
#SBATCH --time=0-00:45:00

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-high-variance/%x_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-high-variance/%x_%A_%a.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Usage: sbatch vary-expert-high-variance.sh <dataset> <llm> [lam]
# Example: sbatch vary-expert-high-variance.sh cuad llama
#          sbatch vary-expert-high-variance.sh misogynistic llama 0.1
DATASET=$1
LLM=$2
LAM=${3:-0.01}

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

LAM_SUFFIX=$([ "$LAM" = "0.01" ] && echo "" || echo "_lam$(echo $LAM | tr -d '.')")

ANNOTATED_CSV="/code/thesis/datasets/annotated/${DATASET}/${DATASET}_${LLM}_annotated.csv"
OUTPUT_DIR="/code/thesis/results/vary-expert-high-variance${LAM_SUFFIX}/${DATASET}/${LLM}"

mkdir -p /mimer/NOBACKUP/groups/ci-nlp-alvis/logs/vary-expert-high-variance
mkdir -p "${CODE_DIR}/thesis/results/vary-expert-high-variance${LAM_SUFFIX}/${DATASET}/${LLM}"

echo "Dataset: ${DATASET} | LLM: ${LLM} | lam: ${LAM} | Rep: ${SLURM_ARRAY_TASK_ID}"

apptainer exec \
    --bind ${CODE_DIR}:/code \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_2_and_3/expert_variance.py \
        "${ANNOTATED_CSV}" \
        "${OUTPUT_DIR}/rep_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed    "${SLURM_ARRAY_TASK_ID}" \
        --dataset "${DATASET}" \
        --phase   "high" \
        --lam     "${LAM}"
