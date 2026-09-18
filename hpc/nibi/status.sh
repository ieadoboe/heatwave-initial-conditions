#!/bin/bash
# What is queued, what finished, and how far the atlas has actually got.
#
#     source hpc/nibi/env.sh
#     bash hpc/nibi/status.sh
#
# Slurm answers the first two questions; the third comes from the files each
# stage leaves behind, which is the one that says whether the science ran.
set -uo pipefail

EVENTS=(pnw_jun2021 stjohns_aug2025 moscow_jul2010 japan_jul2018 \
        sahel_apr2024 brazil_nov2023 siberia_jun2020)

echo "=== queued or running ==="
squeue -u "$USER" -o '%.12i %.12j %.9T %.10M %.10l %R'
echo

echo "=== finished since yesterday ==="
sacct -u "$USER" -S "$(date -d yesterday +%F)" -X \
      --format=JobID%15,JobName%12,State%12,Elapsed,End
echo

echo "=== stage 1: initial conditions ==="
n=0
for e in "${EVENTS[@]}"; do
    store=$(ls -d "$HEATWAVE_ROOT"/data/era5_ic_"$e"_*.zarr 2>/dev/null | head -1)
    if [ -n "$store" ]; then
        printf '  %-18s %s\n' "$e" "$(du -sh "$store" 2>/dev/null | cut -f1)"
        n=$((n + 1))
    else
        printf '  %-18s missing\n' "$e"
    fi
done
echo "  $n/7 built"
echo

echo "=== stage 2: optimizations ==="
n=0
for e in "${EVENTS[@]}"; do
    run=$(ls -d "$HEATWAVE_PERSIST"/opt_runs/"$e"_* 2>/dev/null | head -1)
    if [ -n "$run" ] && [ -f "$run/storyline.csv" ]; then
        # The venv's python explicitly: a plain `python` is the bare module
        # interpreter, which has no pandas unless the venv is activated.
        # Attribute access, not d["optimized_C"]: Python 3.11 rejects a
        # backslash inside an f-string expression, and quoting one through
        # bash into python -c needs escapes this way round.
        gain=$("$HEATWAVE_VENV/bin/python" -c 'import sys, pandas as pd
d = pd.read_csv(sys.argv[1], index_col=0)
g = d.optimized_C.max() - d.unperturbed_C.max()
print(f"{g:+.2f} C")' "$run/storyline.csv" 2>/dev/null || echo "?")
        printf '  %-18s done   gain %s\n' "$e" "$gain"
        n=$((n + 1))
    elif [ -n "$run" ]; then
        printf '  %-18s running or incomplete (no storyline.csv)\n' "$e"
    else
        printf '  %-18s not started\n' "$e"
    fi
done
echo "  $n/7 with a storyline"
echo

echo "=== stage 3: EXP75 ensembles ==="
n=0
for e in "${EVENTS[@]}"; do
    dir="$HEATWAVE_PERSIST/ensembles/EXP75/$e"
    if [ -d "$dir" ]; then
        m=$(ls "$dir"/member_*.csv 2>/dev/null | wc -l)
        printf '  %-18s %3d/75 members\n' "$e" "$m"
        [ "$m" -ge 75 ] && n=$((n + 1))
    else
        printf '  %-18s not started\n' "$e"
    fi
done
echo "  $n/7 complete"
echo

echo "=== failures in the logs ==="
if ! grep -l "Traceback\|FAILED:" "$HEATWAVE_REPO"/logs/nibi/*.out 2>/dev/null; then
    echo "  none"
fi
