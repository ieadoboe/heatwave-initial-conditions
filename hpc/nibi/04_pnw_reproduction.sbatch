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
# Two GPU stages, because +3.7 C is the optimized peak minus the peak of the
# HOTTEST of 75 stochastic ensemble members, not minus the single unperturbed
# forecast (heatwave_ic/ensemble.py). The ensemble has to exist before the
# gain can be judged, so this job runs both and then checks.
#
#     source hpc/nibi/env.sh
#     sbatch hpc/nibi/04_pnw_reproduction.sbatch
#
# Run 01_build_ics.sbatch first if the PNW initial condition is not built:
# this job will build it otherwise, on a GPU node, at GPU prices.
#
# Not resume-aware, unlike 02: it passes --rerun, so a resubmission starts the
# optimization again. At 75 iterations that is about 31 minutes, and the
# ensemble adds roughly 10 more, inside a 2-hour walltime.
#
# The job FAILS when the gain misses 3.7 C, which is the point of it. Two known
# differences to weigh before chasing anything else:
#   - configs/pnw_jun2021.yaml matches Table 1's EXP75 row: 75 iterations,
#     init date 2021-06-21, and lambda 10 times per-variable weights that
#     give the paper's 100/10/1000/1000/100/10/10. The other six configs keep
#     the released config_example.yaml settings.
#   - Runs so far, as gain over the hottest ensemble member: +5.56 C at 75
#     iterations on the old mean(x0)^2 scale from a 2021-06-20 initial
#     condition. The 50-iteration run on the new scale has no ensemble yet.
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

echo "[$(date -Is)] 75-member ensemble for $EVENT"
# --rerun: the optimization above ran fresh, so the ensemble must too. Without
# it, streamed members from an earlier initial condition would be reused.
python scripts/run_ensembles.py \
    --configs "configs/${EVENT}.yaml" \
    --members 75 \
    --summary "data/ensemble_summary_${EVENT}.csv" \
    --persist-dir "$HEATWAVE_PERSIST" \
    --rerun \
    --quiet

echo "[$(date -Is)] checking the run"
python scripts/check_run.py \
    --summary "$SUMMARY" \
    --expect-gain "$EXPECT_GAIN" \
    --tol "$TOL"
echo "[$(date -Is)] reproduced +${EXPECT_GAIN} C within ${TOL} C"
