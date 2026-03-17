#!/bin/bash
#
# Evaluate high-variance (Phase 3) results for all LLMs.
# Loops over the 4 LLMs, loads all 300 rep_*.npz files, and writes one summary CSV each.
#
# Usage: bash evaluate-high-variance.sh <dataset>
# Example: bash evaluate-high-variance.sh cuad

set -eo pipefail

DATASET=${1:?"Usage: $0 <dataset>  (e.g. cuad)"}

CONTAINER_PATH="$HOME/benchmarking.sif"
CODE_DIR="/cephyr/users/kesaf/Vera/llm-debiasing-benchmark"

for LLM in llama deepseek gpt54 mistral; do
    RESULTS_DIR="/code/thesis/results/vary-expert-high-variance/${DATASET}/${LLM}"
    OUTPUT_CSV="/code/thesis/results/summaries/${DATASET}_${LLM}_high_variance.csv"

    echo "=== Evaluating ${DATASET} / ${LLM} ==="

    apptainer exec \
        --bind ${CODE_DIR}:/code \
        --pwd /code \
        ${CONTAINER_PATH} \
        python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_2_and_3/evaluate_low_variance.py \
            "${RESULTS_DIR}" \
            --dataset "${DATASET}" \
            --llm    "${LLM}" \
            --output "${OUTPUT_CSV}"

    echo "Done → ${OUTPUT_CSV}"
    echo ""
done

echo "All LLMs evaluated."
