#! /bin/bash

BASE_DIR="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments/no-collinear-90"

#########################
# No collinearity plots #
#########################

EXPERT_PLOTS="plots/no-collinear-90/vary-expert"
TOTAL_PLOTS="plots/no-collinear-90/vary-total"

mkdir -p $EXPERT_PLOTS
mkdir -p $TOTAL_PLOTS

scp -r vera:"$BASE_DIR/vary-num-expert/plot/*" "$EXPERT_PLOTS"
scp -r vera:"$BASE_DIR/vary-num-total/plot/*" "$TOTAL_PLOTS"
