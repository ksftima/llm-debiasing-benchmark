#!/bin/bash

DATASET=$1
ANNOTATION=$2

python3 "lib/gather_responses.py" \
	"$DATASET" \
	"results/annotations/$DATASET/parsed.json" \
	"results/annotations/$DATASET/$ANNOTATION" \
	"results/annotations/$DATASET/annotated_$ANNOTATION.json"

