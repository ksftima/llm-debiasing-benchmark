#!/bin/bash

set -eo pipefail

DATASET=$1

ANN_DIR="results/annotations/$DATASET"
RESPONSE_DIR="$ANN_DIR/claude"

mkdir -p $RESPONSE_DIR

python3 lib/annotate_api.py \
    "claude" \
    "$DATASET" \
    "$ANN_DIR/parsed.json" \
    "$RESPONSE_DIR" \
    --num 10000 \
    --start 0 \
    --num_examples 0
