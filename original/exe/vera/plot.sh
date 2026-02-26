#! /bin/bash

set -eo pipefail

HOME_DIR="/cephyr/users/audinet/Vera/dsl-use"
MIMER_DIR="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use"
BASE_DIR="$MIMER_DIR/experiments/no-collinear-90"

module purge
module load matplotlib/3.7.2-gfbf-2023a

python "$HOME_DIR/lib/vary_expert_plot.py" \
    "$BASE_DIR/vary-num-expert" \
    --num_reps 225

python "$HOME_DIR/lib/vary_total_plot.py" \
    "$BASE_DIR/vary-num-total" \
    --num_reps 225
