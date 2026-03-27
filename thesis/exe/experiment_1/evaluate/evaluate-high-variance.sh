#!/bin/bash
#
# Evaluate high-variance (Phase 3) results for all LLMs.
# Loops over the 4 LLMs, loads all 300 rep_*.npz files, and writes one summary CSV each.
#
# Usage: bash evaluate-high-variance.sh <dataset> [lam]
# Example: bash evaluate-high-variance.sh cuad
#          bash evaluate-high-variance.sh cuad 0.1

set -eo pipefail

DATASET=${1:?"Usage: $0 <dataset>  (e.g. cuad)"}
LAM=${2:-0.01}

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

LAM_SUFFIX=$([ "$LAM" = "0.01" ] && echo "" || echo "_lam$(echo $LAM | tr -d '.')")

for LLM in llama deepseek gpt54 mistral claude; do
    RESULTS_DIR="/code/thesis/results/vary-expert-high-variance${LAM_SUFFIX}/${DATASET}/${LLM}"
    OUTPUT_CSV="/code/thesis/results/summaries/${DATASET}_${LLM}_high_variance${LAM_SUFFIX}.csv"

    echo "=== Evaluating ${DATASET} / ${LLM} ==="

    apptainer exec \
        --bind ${CODE_DIR}:/code \
        --pwd /code \
        ${CONTAINER_PATH} \
        python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_2_and_3/evaluate_variance.py \
            "${RESULTS_DIR}" \
            --dataset "${DATASET}" \
            --llm    "${LLM}" \
            --output "${OUTPUT_CSV}" \
            --phase  "high_variance"

    echo "Done → ${OUTPUT_CSV}"
    echo ""
done

echo "All LLMs evaluated."
