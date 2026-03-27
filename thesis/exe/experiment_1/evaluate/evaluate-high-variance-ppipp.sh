#!/bin/bash
#
# Evaluate high-variance PPI++ (Phase 3) results for all LLMs.
# Outputs one ppipp-only CSV per LLM — merge with originals locally.
#
# Usage: bash evaluate-high-variance-ppipp.sh <dataset>
# Example: bash evaluate-high-variance-ppipp.sh cuad

set -eo pipefail

DATASET=${1:?"Usage: $0 <dataset>  (e.g. cuad)"}

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

for LLM in llama deepseek gpt54 mistral claude; do
    RESULTS_DIR="/code/thesis/results/vary-expert-high-variance-ppipp/${DATASET}/${LLM}"
    OUTPUT_CSV="/code/thesis/results/summaries-ppipp/${DATASET}_${LLM}_high_variance_ppipp.csv"

    echo "=== Evaluating ${DATASET} / ${LLM} ==="

    apptainer exec \
        --bind ${CODE_DIR}:/code \
        --pwd /code \
        ${CONTAINER_PATH} \
        python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_2_and_3_ppipp/evaluate_variance_ppipp.py \
            "${RESULTS_DIR}" \
            --dataset "${DATASET}" \
            --llm     "${LLM}" \
            --output  "${OUTPUT_CSV}" \
            --phase   "high_variance"

    echo "Done → ${OUTPUT_CSV}"
    echo ""
done

echo "All LLMs evaluated."
