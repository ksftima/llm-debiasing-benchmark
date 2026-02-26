#!/bin/bash

set -eo pipefail

DATASET=$1

ANN_DIR="results/annotations/$DATASET"
RESPONSE_DIR="$ANN_DIR/deepseek_test"
mkdir -p "$RESPONSE_DIR"

python3 lib/annotate_api.py \
    "deepseek" \
    "$DATASET" \
    "$ANN_DIR/parsed.json" \
    "$RESPONSE_DIR" \
    --num 10 \
    --start 10 \
    --num_examples 0
