#!/bin/bash
#SBATCH --job-name=hw-atlas
#SBATCH --array=0-6
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/nibi/atlas_%A_%a.out
# One event's IC optimization per array task, 50 iterations as configured.
#
# Walltime measured 2026-09-07 on Siberia, the longest event at 11 days:
# 24.7 s per iteration, so the configured 50 iterations take ~21 min (it was
# 75 and ~36 min when this was measured). Two hours covers compilation, the
# netCDF writes and a slower node.
#
# The 40 GB slice, not 20 GB: the backward pass through the 264-step unroll
# peaks near 16.8 GB, and on a 20 GB slice XLA hit its allocator ceiling and
# fell back to a slower plan. A full h100:1 is the next step up if a longer
# event ever needs it.
set -euo pipefail
# Printed before anything that can fail, so a dead task never leaves an empty
# log with nothing to diagnose.
echo "[$(date -Is)] optimization task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
source "$SLURM_SUBMIT_DIR/hpc/nibi/env.sh"
source "$HEATWAVE_VENV/bin/activate"
cd "$HEATWAVE_REPO"

CONFIGS=(pnw_jun2021 stjohns_aug2025 moscow_jul2010 japan_jul2018 \
         sahel_apr2024 brazil_nov2023 siberia_jun2020)
EVENT=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date -Is)] optimizing $EVENT on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python scripts/run_atlas.py \
    --configs "configs/${EVENT}.yaml" \
    --summary "data/atlas_summary_${EVENT}.csv" \
    --persist-dir "$HEATWAVE_PERSIST" \
    --quiet

# Iteration 1 of reg.npy, for this task's run. A learning rate too large for
# the regularization term shows up there and nowhere else: the run still
# writes a full set of well-formed outputs. Failing the task is what makes it
# visible in sacct instead of weeks later.
echo "[$(date -Is)] checking $EVENT"
python scripts/check_run.py --summary "data/atlas_summary_${EVENT}.csv"
