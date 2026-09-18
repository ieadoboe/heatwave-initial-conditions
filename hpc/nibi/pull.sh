#!/bin/bash
# Bring the atlas results back from Nibi. Run this on your LAPTOP.
#
#     bash hpc/nibi/pull.sh              # summaries, storylines, arrays, figures
#     bash hpc/nibi/pull.sh --with-nc    # also the trajectory netCDFs (GBs)
#     bash hpc/nibi/pull.sh --with-members   # also the 525 raw member CSVs
#
# Everything lands where the project convention says it should: data/ for
# data, plots/ for figures. The trajectory netCDFs are excluded by default
# because they are several GB per event and nothing in the analysis reads them
# until you want to map a field.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-nibi}"
REMOTE_DIR="${REMOTE_DIR:-projects/def-arminnl/ieadoboe/heatwave_atlas}"
# SRC can be set directly to a local path, which is how this is tested.
SRC="${SRC:-$REMOTE_HOST:$REMOTE_DIR}"

WITH_NC=0
WITH_MEMBERS=0
for arg in "$@"; do
    case "$arg" in
        --with-nc) WITH_NC=1 ;;
        --with-members) WITH_MEMBERS=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$REPO/data/opt_runs" "$REPO/plots"

echo "=== summary tables ==="
rsync -a "$SRC"/atlas_summary.csv "$SRC"/ensemble_summary.csv "$REPO/data/" \
    2>/dev/null || echo "  (none yet; run scripts/collect_summaries.py on Nibi)"

echo "=== run directories ==="
RUN_FILTERS=(--include='*/'
             --include='storyline.csv'
             --include='ensemble.csv'
             --include='*.npy')
[ "$WITH_NC" = 1 ] && RUN_FILTERS+=(--include='*.nc')
RUN_FILTERS+=(--exclude='*')
rsync -a --prune-empty-dirs "${RUN_FILTERS[@]}" \
    "$SRC"/opt_runs/ "$REPO/data/opt_runs/"

echo "=== figures ==="
rsync -a "$SRC"/plots/ "$REPO/plots/" 2>/dev/null || echo "  (none)"

if [ "$WITH_MEMBERS" = 1 ]; then
    echo "=== raw ensemble members ==="
    rsync -a "$SRC"/ensembles/ "$REPO/data/ensembles/"
fi

echo
echo "Pulled from $SRC"
du -sh "$REPO/data/opt_runs" "$REPO/plots" 2>/dev/null || true
ls -1 "$REPO/data"/*_summary.csv 2>/dev/null || true
