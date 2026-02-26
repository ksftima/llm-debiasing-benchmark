#!/bin/bash

datasets=("amazon" "misinfo" "biobias" "germeval")
annotations=("bert" "deepseek" "phi4" "claude")

for dataset in "${datasets[@]}"; do
    for annotation in "${annotations[@]}"; do
        sbatch exe/vera/vary-total.sh $annotation $dataset
    done
done
