#!/bin/bash
# Usage: bash evaluate-total-variance.sh <dataset>
# Example: bash evaluate-total-variance.sh cuad
# Evaluates both low-variance (phase 2) and high-variance (phase 3)

DATASET=$1
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"
CONTAINER_PATH="$HOME/benchmarking_reg.sif"

for PHASE in low high; do
    for LLM in llama deepseek gpt54 mistral claude; do
        for N_EXPERT in 50 100 200; do
            RESULTS_DIR="/code/thesis/results/vary-total-${PHASE}-variance/${DATASET}/${LLM}/n_${N_EXPERT}"
            OUTPUT="/code/thesis/results/summaries/${DATASET}_${LLM}_n${N_EXPERT}_${PHASE}_variance_total.csv"

            echo "Evaluating: dataset=${DATASET} phase=${PHASE} llm=${LLM} n_expert=${N_EXPERT}"

            apptainer exec \
                --bind ${CODE_DIR}:/code \
                --pwd /code \
                ${CONTAINER_PATH} \
                python3 /code/thesis/lib/experiments/experiment_vary_total/phase_2_and_3/evaluate_variance.py \
                    "${RESULTS_DIR}" \
                    --dataset "${DATASET}" \
                    --llm     "${LLM}" \
                    --output  "${OUTPUT}"
        done
    done
done
