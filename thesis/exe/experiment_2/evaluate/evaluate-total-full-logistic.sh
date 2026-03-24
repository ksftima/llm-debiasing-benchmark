#!/bin/bash
# Usage: bash evaluate-total-full-logistic.sh <dataset>
# Example: bash evaluate-total-full-logistic.sh cuad

DATASET=$1
CODE_DIR="/cephyr/users/kesaf/Vera/llm-debiasing-benchmark"
CONTAINER_PATH="$HOME/benchmarking_reg.sif"

for LLM in llama deepseek gpt54 mistral claude; do
    for N_EXPERT in 50 100 200; do
        RESULTS_DIR="/code/thesis/results/vary-total-full-logistic/${DATASET}/${LLM}/n_${N_EXPERT}"
        OUTPUT="/code/thesis/results/summaries/${DATASET}_${LLM}_n${N_EXPERT}_full_logistic_total.csv"

        echo "Evaluating: dataset=${DATASET} llm=${LLM} n_expert=${N_EXPERT}"

        apptainer exec \
            --bind ${CODE_DIR}:/code \
            --pwd /code \
            ${CONTAINER_PATH} \
            python3 /code/thesis/lib/experiments/experiment_vary_total/phase_4/evaluate_full_logistic.py \
                "${RESULTS_DIR}" \
                --dataset "${DATASET}" \
                --llm     "${LLM}" \
                --output  "${OUTPUT}"
    done
done
