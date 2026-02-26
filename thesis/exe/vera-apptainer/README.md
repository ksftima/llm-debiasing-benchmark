# Vera Apptainer Scripts

This folder contains Apptainer-compatible versions of the SLURM scripts for running experiments on Vera.

## Setup on Vera

### 1. Container Location
These scripts assume your container is at: `~/benchmarking.sif`

If it's elsewhere, update the `CONTAINER_PATH` variable in each script.

### 2. Code Location
Scripts assume code is at: `/cephyr/users/$USER/Vera/llm-debiasing-benchmark` (automatically uses your username)

Update `CODE_DIR` if your path is different.

### 3. Download Datasets

You need to download the annotated datasets from HuggingFace:
https://huggingface.co/datasets/nicaudinet/llm-debiasing-benchmark

The scripts expect annotation files at:
```
/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/annotations/$DATASET/annotated_$ANNOTATION.json
```

For datasets: `amazon`, `misinfo`, `biobias`, `germeval`
With annotations: `bert`, `deepseek`, `phi4`, `claude`

**Download command on Vera:**
```bash
# You may need to install huggingface-hub first
pip install --user huggingface-hub

# Download the dataset
huggingface-cli download nicaudinet/llm-debiasing-benchmark \
    --repo-type dataset \
    --local-dir /mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/annotations/
```

### 4. Update Your Email
Change `theat@chalmers.se` to your email in all scripts if you want job notifications.

## Available Scripts

### Simulation Experiments (No Datasets Needed)
- `vary-expert-simulation.sh` - Vary number of expert samples (simulation)
- `vary-total-simulation.sh` - Vary number of total samples (simulation)

### Real-World Experiments (Need Datasets)
- `vary-expert.sh` - Vary number of expert samples (real-world data)
- `vary-total.sh` - Vary number of total samples (real-world data)
- `run-expert.sh` - Submit all vary-expert jobs for all dataset/annotation combos
- `run-total.sh` - Submit all vary-total jobs for all dataset/annotation combos

## How to Run

### Test with Simulation (No datasets needed)
```bash
sbatch exe/vera-apptainer/vary-expert-simulation.sh
```

### Run Real-World Experiments (After downloading datasets)
```bash
# Single job
sbatch exe/vera-apptainer/vary-expert.sh bert amazon

# All combinations
bash exe/vera-apptainer/run-expert.sh
```

## Key Differences from Original Scripts

1. **No module loading** - Everything runs inside the container
2. **No virtual environment** - Container has all dependencies
3. **Bind mounts** - Code and data are bind-mounted into the container:
   - Code: `/code`
   - Data: `/mimer` or `/data` depending on the script

## Troubleshooting

If jobs fail, check:
1. Container exists at `~/benchmarking.sif`
2. Code directory exists and has correct permissions
3. Datasets are downloaded to correct location
4. Log directories exist (scripts try to create them automatically)
