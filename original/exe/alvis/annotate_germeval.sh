#!/bin/bash

#SBATCH --job-name=germeval-annotate
#SBATCH --account=NAISS2025-22-180
#SBATCH --partition=alvis

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=NOGPU
#SBATCH --time=0-01:00:00

#SBATCH --output=/dev/null
#SBATCH --error=results/logs/error_%A_%a.log

#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a
module load scikit-learn/1.3.1-gfbf-2023a

BASE_DIR="/cephyr/users/audinet/Vera"

python3 \
	"$BASE_DIR/dsl-use/lib/annotate_germeval.py" \
	"$BASE_DIR/datasets/germeval/" \
	"$BASE_DIR/datasets/germeval/annotated.pkl"
