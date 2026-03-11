#!/bin/bash

#SBATCH --job-name=download-model
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00

#SBATCH --output=download_model_%j.log
#SBATCH --error=download_model_%j.err

#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=FAIL

# Usage:
#   sbatch download-model.sh <hf_model_id>
#
# Arguments:
#   hf_model_id   HuggingFace model ID, e.g. mistralai/Mistral-7B-Instruct-v0.3
#
# The model will be downloaded to:
#   /mimer/NOBACKUP/groups/ci-nlp-alvis/theat/models/<model_name>
#
# Example:
#   sbatch download-model.sh mistralai/Mistral-7B-Instruct-v0.3
#   sbatch download-model.sh meta-llama/Llama-3.1-8B-Instruct

MODEL_ID=$1

if [ -z "$MODEL_ID" ]; then
    echo "ERROR: Usage: sbatch download-model.sh <hf_model_id>"
    exit 1
fi

MODELS_DIR="/mimer/NOBACKUP/groups/ci-nlp-alvis/theat/models"
CONTAINER_PATH="$HOME/benchmarking.sif"

echo "================================"
echo "Downloading HuggingFace model"
echo "Job ID:      $SLURM_JOB_ID"
echo "Model:       $MODEL_ID"
echo "Output dir:  $MODELS_DIR"
echo "Started at:  $(date)"
echo "================================"

mkdir -p "$MODELS_DIR"

apptainer exec \
    --bind ${MODELS_DIR}:/models \
    ${CONTAINER_PATH} \
    python3 -c "
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = '${MODEL_ID}'
models_dir = Path('/models')
model_path = models_dir / model_id.replace('/', '--')

if model_path.exists():
    print(f'Model already exists at {model_path}')
else:
    print(f'Downloading {model_id} to {model_path}')
    snapshot_download(
        repo_id=model_id,
        local_dir=model_path,
        local_dir_use_symlinks=False,
    )
    print(f'Downloaded {model_id}')

print(f'Model path: {model_path}')
"

EXIT_CODE=$?

echo "================================"
echo "Completed with exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Download complete"
    echo ""
    echo "To annotate with this model, run:"
    MODEL_NAME=$(echo "$MODEL_ID" | sed 's|/|--|g')
    echo "  sbatch annotate-local-model.sh <dataset> /mimer/NOBACKUP/groups/ci-nlp-alvis/theat/models/${MODEL_NAME}"
else
    echo "✗ Download failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
