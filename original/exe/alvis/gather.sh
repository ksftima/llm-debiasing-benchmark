#!/bin/bash

DATASET=$1
ANNOTATION=$2

module purge
module load Python/3.12.3-GCCcore-13.3.0

source /mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/venv_alvis/bin/activate

BASE_DIR="/cephyr/users/audinet/Alvis/dsl-use"
MIMER_PATH="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/"
ANN_DIR="$MIMER_PATH/annotations/$DATASET/$ANNOTATION"

mkdir -p "$ANN_DIR"

python3 "$BASE_DIR/lib/gather_responses.py" \
	"$DATASET" \
	"$MIMER_PATH/annotations/$DATASET/parsed.json" \
	"$ANN_DIR" \
	"$MIMER_PATH/annotations/$DATASET/annotated_$ANNOTATION.json"

