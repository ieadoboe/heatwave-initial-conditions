#!/bin/bash
#SBATCH --job-name=hw-ic
#SBATCH --array=0-6
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/nibi/ic_%A_%a.out
# Download one event's ARCO-ERA5 initial condition. CPU only: Nibi's compute
# nodes reach the internet, so the GPU jobs could do this themselves, but a
# failed or slow download would burn GPU time. Run this first and the GPU
# jobs find the stores already on scratch.
set -euo pipefail
# Printed before anything that can fail, so a dead task never leaves an empty
# log with nothing to diagnose.
echo "[$(date -Is)] IC build task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
source "$SLURM_SUBMIT_DIR/hpc/nibi/env.sh"
source "$HEATWAVE_VENV/bin/activate"
cd "$HEATWAVE_REPO"

CONFIGS=(pnw_jun2021 stjohns_aug2025 moscow_jul2010 japan_jul2018 \
         sahel_apr2024 brazil_nov2023 siberia_jun2020)
EVENT=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date -Is)] building IC for $EVENT into $HEATWAVE_ROOT"
python scripts/build_ics.py --configs "configs/${EVENT}.yaml"
