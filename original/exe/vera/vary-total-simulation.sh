#!/bin/bash

#SBATCH --job-name=vary-total-simulation
#SBATCH --account=C3SE2025-1-14
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-500
#SBATCH --time=0-00:15:00

#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module purge
module load rpy2
module load scikit-learn/1.4.2-gfbf-2023a

source venv/bin/activate

DATA_DIR="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments/vary-num-total/simulation/data"

mkdir -p $DATA_DIR

echo "Experiment: vary number of total samples (simulation)"
num_expert=(200 1000 5000)
for n in "${num_expert[@]}"; do
    DATA_DIR_N="$DATA_DIR/n$n"
    mkdir -p $DATA_DIR_N
    python "/cephyr/users/audinet/Vera/dsl-use/lib/vary_total_simulation.py" \
        "$DATA_DIR_N/data_simulation_${SLURM_ARRAY_TASK_ID}.npz" \
        "$n"
done
