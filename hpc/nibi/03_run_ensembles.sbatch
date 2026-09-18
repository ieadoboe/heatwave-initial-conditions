#!/bin/bash
#SBATCH --job-name=hw-exp75
#SBATCH --array=0-6
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/nibi/exp75_%A_%a.out
# EXP75: one event's 75-member stochastic-physics ensemble per array task.
# Needs that event's storyline.csv from 02_run_atlas.sbatch. Members stream to
# $HEATWAVE_PERSIST/ensembles/EXP75/<event>/member_NNN.csv as they finish, so
# resubmitting after a timeout resumes at the member it stopped on rather
# than starting over.
#
# Walltime measured 2026-09-07: 7.0 s per member, so 75 members take ~14 min.
# Members are forward-only, so the 20 GB slice is ample here.
set -euo pipefail
# Printed before anything that can fail, so a dead task never leaves an empty
# log with nothing to diagnose.
echo "[$(date -Is)] EXP75 ensemble task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
source "$SLURM_SUBMIT_DIR/hpc/nibi/env.sh"
source "$HEATWAVE_VENV/bin/activate"
cd "$HEATWAVE_REPO"

CONFIGS=(pnw_jun2021 stjohns_aug2025 moscow_jul2010 japan_jul2018 \
         sahel_apr2024 brazil_nov2023 siberia_jun2020)
EVENT=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date -Is)] EXP75 ensemble for $EVENT"
python scripts/run_ensembles.py \
    --configs "configs/${EVENT}.yaml" \
    --members 75 \
    --summary "data/ensemble_summary_${EVENT}.csv" \
    --persist-dir "$HEATWAVE_PERSIST" \
    --quiet
