# Plotting Guide for Test Fitting Results

## Overview

After running `test-fitting.sh` on Vera, you'll have results saved as `.npz` files. These scripts help you visualize those results.

## Files Created

1. **`thesis/lib/plot_test_fitting.py`** - Python plotting script
2. **`thesis/exe/vera-apptainer/plot-test-fitting.sh`** - SLURM script to run on Vera

## What Gets Plotted

The plot shows **Standardized RMSE** (Root Mean Squared Error) comparing three methods:
- **Expert-only** (baseline) - Uses only expert labels
- **DSL** - Your method using expert + predicted labels
- **PPI** - Alternative method for comparison

**X-axis:** Number of expert samples (log scale)
**Y-axis:** RMSE (lower is better)

**Goal:** DSL should have lower RMSE than Expert-only, showing it leverages predicted labels effectively!

---

## Usage on Vera

### Method 1: Automatic (Finds Most Recent Results)

```bash
# On Vera
cd ~/llm-debiasing-benchmark
sbatch thesis/exe/vera-apptainer/plot-test-fitting.sh
```

The script will automatically find your most recent test_fitting results.

### Method 2: Specify Results File

```bash
# On Vera
sbatch thesis/exe/vera-apptainer/plot-test-fitting.sh /path/to/your/results.npz
```

### Method 3: Local Testing (Without SLURM)

```bash
# On Vera (interactive)
apptainer exec \
    --bind ~/llm-debiasing-benchmark:/code \
    ~/benchmarking.sif \
    python /code/thesis/lib/plot_test_fitting.py /path/to/results.npz
```

---

## Output

The script creates two plot files in the same directory as your results:

```
test_fitting_rmse.png  # High-res image for viewing
test_fitting_rmse.pdf  # Vector format for papers
```

---

## Example Workflow

```bash
# Step 1: Run test (already done)
sbatch thesis/exe/vera-apptainer/test-fitting.sh

# Step 2: Check job completed
squeue -u $USER

# Step 3: Plot results
sbatch thesis/exe/vera-apptainer/plot-test-fitting.sh

# Step 4: Download plots
scp theat@vera.c3se.chalmers.se:/path/to/test_fitting_rmse.png .
```

---

## Troubleshooting

### "No results file found"
```bash
# Find your results manually:
find /mimer/NOBACKUP -name "*.npz" 2>/dev/null

# Then specify it:
sbatch plot-test-fitting.sh /full/path/to/results.npz
```

### "Module matplotlib not found"
Check that matplotlib is in your container's `requirements_container.txt`:
```
matplotlib
seaborn
```

### Check logs
```bash
cat plot_test_fitting_*.log
cat plot_test_fitting_*.err
```

---

## Customizing the Plot

Edit `thesis/lib/plot_test_fitting.py` to:
- Change figure size: `figsize=(10, 6)`
- Add more methods
- Change colors/styles
- Add annotations

Then re-run the plotting script!
