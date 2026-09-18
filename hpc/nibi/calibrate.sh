#!/bin/bash
# Measure one event before committing the whole atlas to the queue.
#
#     source hpc/nibi/env.sh
#     bash hpc/nibi/calibrate.sh                       # longest event, 20 GB slice
#     bash hpc/nibi/calibrate.sh configs/pnw_jun2021.yaml h100_3g.40gb
#
# Takes an interactive hour on one H100 slice and runs scripts/calibrate_gpu.py,
# which times a few optimizer iterations and a few ensemble members and prints
# the walltimes to put in 02_run_atlas.sbatch and 03_run_ensembles.sbatch.
set -euo pipefail

CONFIG="${1:-configs/siberia_jun2020.yaml}"
GPU="${2:-h100_2g.20gb}"

salloc --account="$SBATCH_ACCOUNT" --gpus="${GPU}:1" \
       --cpus-per-task=8 --mem=48G --time=1:00:00 \
       srun bash -c "
    source '$HEATWAVE_REPO/hpc/nibi/env.sh'
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    source \"\$HEATWAVE_VENV/bin/activate\"
    cd \"\$HEATWAVE_REPO\"
    python -c 'import jax; print(\"jax\", jax.__version__, jax.devices())'
    python scripts/calibrate_gpu.py --config '$CONFIG'
"
