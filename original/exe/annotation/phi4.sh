#!/bin/bash

set -eo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0

source /mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/venv_alvis/bin/activate

echo ""
echo "##########"
echo "# Gather #"
echo "##########"
echo ""

MIMER="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/annotations"

function gather () {

    local DATASET=$1
    local ANN_DIR="$MIMER/$DATASET"

    mkdir -p "$ANN_DIR/phi4_10ex"

    echo ""
    echo "Gathering responses for $DATASET"
    echo ""

    python3 lib/gather_responses.py \
        "$DATASET" \
        "$ANN_DIR/parsed.json" \
        "$ANN_DIR/phi4_10ex" \
        "$ANN_DIR/annotated_phi4_10ex.json"
}

gather "amazon"
gather "misinfo"
gather "biobias"
gather "germeval"
