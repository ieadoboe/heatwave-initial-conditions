#!/bin/bash
#SBATCH --job-name=hw-pnw-repro
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/nibi/pnw_repro_%j.out
# Validation run: reproduce Whittaker & Di Luca's +3.7 C PNW storyline, then
# check the result before anything else in the atlas is trusted.
#
#     source hpc/nibi/env.sh
#     sbatch hpc/nibi/04_pnw_reproduction.sbatch
#
# Run 01_build_ics.sbatch first if the PNW initial condition is not built:
# this job will build it otherwise, on a GPU node, at GPU prices.
#
# Not resume-aware, unlike 02: it passes --rerun, so a resubmission starts the
# optimization again. At 50 iterations that is about 25 minutes of the 2-hour
# walltime, so the job has room to finish twice over.
#
# The job FAILS when the gain misses 3.7 C, which is the point of it. Two known
# differences to weigh before chasing anything else:
#   - configs/pnw_jun2021.yaml runs W&DL's released config_example.yaml
#     settings at 50 iterations. Table 1's EXP75 row is 75 iterations with
#     per-variable lambdas of 100/10/1000/1000/100/10/10, which the example
#     file does not reproduce.
#   - The last PNW run, at iteration 75 and with the old mean(x0)^2
#     regularization reference scale, gave +9.53 C. This is the first run on
#     the mean(x0^2) scale, so a different number is expected, not alarming.
set -euo pipefail
echo "[$(date -Is)] PNW reproduction on $(hostname)"
source "$SLURM_SUBMIT_DIR/hpc/nibi/env.sh"
source "$HEATWAVE_VENV/bin/activate"
cd "$HEATWAVE_REPO"

EVENT=pnw_jun2021
SUMMARY="data/atlas_summary_${EVENT}.csv"
EXPECT_GAIN=3.7          # W&DL's PNW storyline gain, in C
TOL=0.5                  # how far off that the run may land before failing

echo "[$(date -Is)] optimizing $EVENT on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python scripts/run_atlas.py \
    --configs "configs/${EVENT}.yaml" \
    --summary "$SUMMARY" \
    --persist-dir "$HEATWAVE_PERSIST" \
    --rerun \
    --quiet

echo "[$(date -Is)] checking the run"
python scripts/check_run.py \
    --summary "$SUMMARY" \
    --expect-gain "$EXPECT_GAIN" \
    --tol "$TOL"
echo "[$(date -Is)] reproduced +${EXPECT_GAIN} C within ${TOL} C"
