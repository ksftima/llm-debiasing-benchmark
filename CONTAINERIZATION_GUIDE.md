# Containerization Guide for LLM Debiasing Benchmark

This document provides a checklist and guideline for containerizing this codebase for use on HPC clusters (Vera/Minerva at C3SE) with Apptainer/Singularity.

## Overview

This project requires both Python and R environments due to:
- Python-based ML/LLM code (PyTorch, transformers, scikit-learn)
- R-based DSL (Design-based Supervised Learning) package
- Integration via rpy2

## Pre-Containerization Checklist

### 1. Project Scope
**Which components need to be containerized?** (Check all that apply)

- [ ] Core fitting/experiment code (`lib/fitting.py`, `lib/vary_*.py`)
- [ ] LLM annotation code (`lib/annotate_*.py`)
- [ ] Statistical analysis (`stats/`)
- [ ] Data parsing (`lib/parse_*.py`)
- [ ] All of the above

### 2. Compute Requirements

**GPU Support:**
- [ ] Yes - Need CUDA for PyTorch/transformers (LLM annotation)
- [ ] No - CPU-only (fitting/experiments only)
- [ ] Not sure - (Assistant: check if running local LLMs or using API-only)

**Target Cluster:**
- [ ] Vera (CPU-focused)
- [ ] Alvis (GPU-focused)
- [ ] Minerva
- [ ] Both/All
- [ ] Other: _______________

### 3. Software Dependencies

**Python Version:**
- Current: Python 3.12+ (from codebase inspection)
- Required: _______________ (if specific version needed)

**Python Packages:**
- Location: `requirements.txt` and/or `requirements_cluster.txt`
- Any package version constraints? _______________

**R Requirements:**
- R version: _______________ (or latest stable)
- R packages needed: `dsl` (confirmed), others: _______________
- CRAN or custom repository? _______________

**System Dependencies:**
- [ ] Build tools (gcc, g++, make) - for compiling packages
- [ ] HDF5, NetCDF, or other scientific libraries
- [ ] Other: _______________

### 4. Data and File Access

**Where is data stored?**
- [ ] In repository (will be copied into container)
- [ ] On cluster filesystem (e.g., `/cephyr`, project directory)
- [ ] Will be mounted as volume
- Path: _______________

**Working directory on cluster:**
- Path: _______________

**Output directory:**
- Path: _______________

### 5. External Services

**API Keys Required:** (will be passed as environment variables)
- [ ] OpenAI API (for `annotate_api.py`)
- [ ] Anthropic/Claude API (for `annotate_api.py`)
- [ ] DeepSeek API (for `annotate_api.py`)
- [ ] Other: _______________

**How are secrets managed?**
- [ ] Environment variables in SLURM script
- [ ] Loaded from file (specify path: _______________)
- [ ] Other: _______________

### 6. SLURM Integration

**Current SLURM scripts location:**
- `exe/vera/` (confirmed)
- `exe/alvis/` (confirmed)
- Others: _______________

**Typical job structure:**
- [ ] Single-node jobs
- [ ] Multi-node MPI jobs (needs special MPI configuration)
- [ ] Array jobs (confirmed - 300 jobs in vary-expert.sh)
- [ ] GPU jobs

**Resource requirements per job:**
- CPUs: _______________
- Memory: _______________
- Time limit: _______________
- GPU (if applicable): _______________

---

## What the Assistant Will Create

### 1. Apptainer Definition File (`llm-debiasing.def`)

**Structure:**
```
Bootstrap: docker
From: <base-image>

%files
    requirements.txt /opt/
    [other necessary files]

%post
    # System packages
    # Python packages
    # R installation
    # R packages (dsl)
    # Cleanup

%environment
    # PATH, PYTHONPATH, R_LIBS
    # Any default environment variables

%runscript
    # Default command when using `apptainer run`

%labels
    Author <your-name>
    Version <version>
```

### 2. Modified SLURM Scripts (Optional)

**Before:**
```bash
python lib/vary_expert_realworld.py --args
```

**After:**
```bash
apptainer exec llm-debiasing.sif python lib/vary_expert_realworld.py --args
```

### 3. Build Instructions

```bash
# On Vera login node
apptainer build llm-debiasing.sif llm-debiasing.def

# Test the container
apptainer exec llm-debiasing.sif python --version
apptainer exec llm-debiasing.sif R --version
apptainer shell llm-debiasing.sif  # Interactive testing
```

### 4. Usage Documentation

- How to rebuild the container
- How to update dependencies
- How to modify SLURM scripts
- Troubleshooting common issues

---

## Cluster-Specific Considerations

### Vera/C3SE Best Practices

**From C3SE Documentation:**

1. **Don't mix modules and containers**
   - Container should be self-contained
   - Avoid `module load` in SLURM scripts when using containers

2. **Automatic bind mounts**
   - Home directory: automatically mounted
   - Current working directory: automatically mounted
   - System paths: `/cephyr`, `/apps` available

3. **GPU support**
   - Use `--nv` flag (auto-configured on GPU nodes)
   - Example: `apptainer exec --nv llm-debiasing.sif python script.py`

4. **Performance considerations**
   - C3SE-optimized NumPy can be 9x faster than generic pip version
   - Consider using C3SE's base containers from `/apps/containers/`
   - Build on top of their optimized images when possible

5. **Container storage**
   - `.sif` files can be large (GB)
   - Store in project directory, not home (quota limits)
   - Recommended location: _______________

---

## Testing Checklist

Before using in production, test:

- [ ] Container builds successfully
- [ ] Python imports work (`import numpy, pandas, sklearn, torch, transformers`)
- [ ] R works and can load `dsl` package
- [ ] rpy2 integration works (test `lib/fitting.py` DSL functions)
- [ ] Can read/write files in expected locations
- [ ] Environment variables (API keys) are accessible
- [ ] SLURM job submission works
- [ ] Results match non-containerized runs (sanity check)

**Test commands:**
```bash
# Basic functionality
apptainer exec llm-debiasing.sif python -c "import numpy; print(numpy.__version__)"
apptainer exec llm-debiasing.sif R --version
apptainer exec llm-debiasing.sif python -c "import rpy2; print(rpy2.__version__)"

# Test fitting.py DSL integration
apptainer exec llm-debiasing.sif python -c "from lib.fitting import logit_fit_dsl; print('DSL import successful')"

# Test full workflow
apptainer exec llm-debiasing.sif python tests/test_fitting.py
```

---

## Maintenance Plan

**When to rebuild the container:**
- [ ] When updating Python packages (requirements.txt)
- [ ] When updating R packages
- [ ] When changing Python/R versions
- [ ] When fixing bugs in the container setup

**Versioning strategy:**
- [ ] Use git tags: `llm-debiasing-v1.0.sif`
- [ ] Use date stamps: `llm-debiasing-20260209.sif`
- [ ] Other: _______________

---

## Optional Enhancements

Consider these for future iterations:

- [ ] **Multi-stage build** - Reduce final image size
- [ ] **Separate containers** - One for annotation, one for fitting
- [ ] **Container registry** - Push to Docker Hub or similar for sharing
- [ ] **CI/CD integration** - Auto-build on git push
- [ ] **Documentation** - Container-specific README

---

## Quick Reference: Apptainer Commands

```bash
# Build container
apptainer build <name>.sif <name>.def

# Run command in container
apptainer exec <name>.sif <command>

# Interactive shell
apptainer shell <name>.sif

# Run default runscript
apptainer run <name>.sif

# With GPU
apptainer exec --nv <name>.sif <command>

# Bind custom directory
apptainer exec --bind /path/on/host:/path/in/container <name>.sif <command>

# Pass environment variable
apptainer exec --env MY_VAR=value <name>.sif <command>
```

---

## When Ready to Build

**Hand this completed guide to your assistant with:**

1. ✅ All checkboxes filled out
2. ✅ All blank fields completed
3. ✅ Any additional notes or special requirements

**The assistant will:**
1. Create the `.def` file
2. Provide build instructions
3. Suggest SLURM script modifications
4. Help with testing and debugging

---

## Resources

- [C3SE Container Documentation](https://www.c3se.chalmers.se/documentation/miscellaneous/containers/)
- [C3SE Container Repository](https://github.com/c3se/containers)
- [Vera Cluster Info](https://www.c3se.chalmers.se/about/Vera/)
- [Apptainer Documentation](https://apptainer.org/docs/)
- [C3SE Support](https://www.c3se.chalmers.se/about/contact/)

---

## Notes / Special Requirements

[Use this space for any additional context, special configurations, or questions]

```






```
