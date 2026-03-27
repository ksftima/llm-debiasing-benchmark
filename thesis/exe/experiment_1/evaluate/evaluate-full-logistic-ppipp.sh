#!/bin/bash
#
# Evaluate full logistic PPI++ (Phase 4) results for all LLMs.
# Outputs one ppipp-only CSV per LLM — merge with originals locally.
#
# Usage: bash evaluate-full-logistic-ppipp.sh <dataset>
# Example: bash evaluate-full-logistic-ppipp.sh cuad

set -eo pipefail

DATASET=${1:?"Usage: $0 <dataset>  (e.g. cuad)"}

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

for LLM in llama deepseek gpt54 mistral claude; do
    RESULTS_DIR="/code/thesis/results/vary-expert-full-logistic-ppipp/${DATASET}/${LLM}"
    OUTPUT_CSV="/code/thesis/results/summaries-ppipp/${DATASET}_${LLM}_full_logistic_ppipp.csv"

    echo "=== Evaluating ${DATASET} / ${LLM} ==="

    apptainer exec \
        --bind ${CODE_DIR}:/code \
        --pwd /code \
        ${CONTAINER_PATH} \
        python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_4_ppipp/evaluate_full_logistic_ppipp.py \
            "${RESULTS_DIR}" \
            --dataset "${DATASET}" \
            --llm     "${LLM}" \
            --output  "${OUTPUT_CSV}"

    echo "Done → ${OUTPUT_CSV}"
    echo ""
done

echo "All LLMs evaluated."
