#!/bin/bash
#
# Evaluate full logistic regression (Phase 4) results for all LLMs.
# Loops over the 4 LLMs, loads all 300 rep_*.npz files, and writes one summary CSV each.
#
# Usage: bash evaluate-full-logistic.sh <dataset> [lam]
# Example: bash evaluate-full-logistic.sh cuad
#          bash evaluate-full-logistic.sh misogynistic 0.1

set -eo pipefail

DATASET=${1:?"Usage: $0 <dataset>  (e.g. cuad)"}
LAM=${2:-0.01}

CONTAINER_PATH="$HOME/benchmarking_reg.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"

LAM_SUFFIX=$([ "$LAM" = "0.01" ] && echo "" || echo "_lam$(echo $LAM | tr -d '.')")

for LLM in llama deepseek gpt54 mistral claude; do
    RESULTS_DIR="/code/thesis/results/vary-expert-full-logistic${LAM_SUFFIX}/${DATASET}/${LLM}"
    OUTPUT_CSV="/code/thesis/results/summaries/${DATASET}_${LLM}_full_logistic${LAM_SUFFIX}.csv"

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
