#!/bin/bash
#
# Evaluate full logistic regression (Phase 4) results for all LLMs.
# Loops over the 4 LLMs, loads all 300 rep_*.npz files, and writes one summary CSV each.
#
# Usage: bash evaluate-full-logistic.sh <dataset>
# Example: bash evaluate-full-logistic.sh cuad

set -eo pipefail

DATASET=${1:?"Usage: $0 <dataset>  (e.g. cuad)"}

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/kesaf/Vera/llm-debiasing-benchmark"

for LLM in llama deepseek gpt54 mistral claude; do
    RESULTS_DIR="/code/thesis/results/vary-expert-full-logistic/${DATASET}/${LLM}"
    OUTPUT_CSV="/code/thesis/results/summaries/${DATASET}_${LLM}_full_logistic.csv"

    echo "=== Evaluating ${DATASET} / ${LLM} ==="

    apptainer exec \
        --bind ${CODE_DIR}:/code \
        --pwd /code \
        ${CONTAINER_PATH} \
        python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_4/evaluate_full_logistic.py \
            "${RESULTS_DIR}" \
            --dataset "${DATASET}" \
            --llm     "${LLM}" \
            --output  "${OUTPUT_CSV}"

    echo "Done → ${OUTPUT_CSV}"
    echo ""
done

echo "All LLMs evaluated."
