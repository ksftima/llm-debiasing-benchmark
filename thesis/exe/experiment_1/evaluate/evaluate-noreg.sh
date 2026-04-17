#!/bin/bash
#SBATCH --job-name=eval-noreg
#SBATCH --account=C3SE2026-1-12
#SBATCH --partition=vera
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-02:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/eval-noreg.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/logs/eval-noreg.err
#SBATCH --mail-user=theat@chalmers.se
#SBATCH --mail-type=END,FAIL

# Usage: sbatch evaluate-noreg.sh <dataset1> [dataset2] ...
# Example: sbatch evaluate-noreg.sh cuad fomc
# If no datasets given, runs all 5

DATASETS=${@:-"cuad fomc pubmedqa misogynistic vuamc"}

cd /cephyr/users/theat/Vera/llm-debiasing-benchmark

for dataset in $DATASETS; do
  for llm in llama deepseek gpt54 mistral claude; do
    apptainer exec --bind $(pwd):/code --pwd /code $HOME/benchmarking_reg.sif python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_4/evaluate_full_logistic.py /code/thesis/results/vary-expert-full-logistic-noreg/${dataset}/${llm} --dataset ${dataset} --llm ${llm} --output /code/thesis/results/summaries/${dataset}_${llm}_full_logistic_noreg.csv
    for phase in low_variance high_variance; do
      phase_dir=$(echo $phase | tr _ -)
      apptainer exec --bind $(pwd):/code --pwd /code $HOME/benchmarking_reg.sif python3 /code/thesis/lib/experiments/experiment_vary_experts/phase_2_and_3/evaluate_variance.py /code/thesis/results/vary-expert-${phase_dir}-noreg/${dataset}/${llm} --dataset ${dataset} --llm ${llm} --phase ${phase} --output /code/thesis/results/summaries/${dataset}_${llm}_${phase}_noreg.csv
    done
  done
done
