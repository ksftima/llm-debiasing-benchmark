#!/bin/bash

#SBATCH --job-name=annotate-local-model
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-06:00:00

#SBATCH --output=annotate_local_%j.log
#SBATCH --error=annotate_local_%j.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=FAIL

# Usage:
#   sbatch annotate-local-model.sh <dataset> <model_path> [num_examples]
#
# Arguments:
#   dataset      fomc | pubmedqa | cuad | misogynistic
#   model_path   HuggingFace model ID or absolute path to local model
#   num_examples Number of few-shot examples (default: 0)
#
# Example:
#   sbatch annotate-local-model.sh fomc /path/to/llama3 0

DATASET=$1
MODEL=$2
NUM_EXAMPLES=${3:-0}

if [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
    echo "ERROR: Usage: sbatch annotate-local-model.sh <dataset> <model_path> [num_examples]"
    exit 1
fi

CONTAINER_PATH="$HOME/benchmarking.sif"
CODE_DIR="/cephyr/users/$USER/Vera/llm-debiasing-benchmark"
MODELS_DIR="/mimer/NOBACKUP/groups/ci-nlp-alvis/theat/models"

MODEL_NAME=$(basename "$MODEL")
ANN_DIR="${CODE_DIR}/thesis/datasets/annotated/${DATASET}/local/${MODEL_NAME}"

echo "================================"
echo "Annotating with local model"
echo "Job ID:      $SLURM_JOB_ID"
echo "Dataset:     $DATASET"
echo "Model:       $MODEL"
echo "Examples:    $NUM_EXAMPLES"
echo "Output dir:  $ANN_DIR"
echo "Started at:  $(date)"
echo "================================"

mkdir -p "$ANN_DIR"

apptainer exec --nv \
    --bind ${CODE_DIR}:/code \
    --bind ${MODELS_DIR}:${MODELS_DIR} \
    --pwd /code \
    ${CONTAINER_PATH} \
    python3 /code/thesis/lib/annotation/annotate_local_model.py \
        "$DATASET" \
        "/code/thesis/datasets/parsed/parsed_scaled_datasets/${DATASET}.csv" \
        "/code/thesis/datasets/annotated/${DATASET}/local/${MODEL_NAME}" \
        --model "$MODEL" \
        --num_examples "$NUM_EXAMPLES" \
        --batchsize 8

EXIT_CODE=$?

echo "================================"
echo "Completed with exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Annotation complete"
else
    echo "✗ Annotation failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
