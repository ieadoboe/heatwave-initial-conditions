#!/bin/bash
# Shared settings for the Nibi batch scripts. Source it before submitting:
#
#     source hpc/nibi/env.sh
#     sbatch hpc/nibi/01_build_ics.sbatch
#
# Every sbatch script sources this too, so the compute-node environment
# matches the one you submitted from.

# --- allocation -----------------------------------------------------------
# The base name, not the def-arminnl_cpu / def-arminnl_gpu associations that
# sacctmgr lists: Slurm picks the right one from what the job requests.
export SBATCH_ACCOUNT="def-arminnl"
export SLURM_ACCOUNT="$SBATCH_ACCOUNT"        # srun
export SALLOC_ACCOUNT="$SBATCH_ACCOUNT"       # salloc, used by calibrate.sh

# --- paths ----------------------------------------------------------------
# Resolved BEFORE the module loads below, which reset $SCRATCH. ~/scratch is
# the symlink Alliance always creates, so it stands in when the variable is
# gone.
HEATWAVE_SCRATCH="${SCRATCH:-$HOME/scratch}"
if [ ! -d "$HEATWAVE_SCRATCH" ]; then
    echo "hpc/nibi/env.sh: no scratch directory at $HEATWAVE_SCRATCH." >&2
    echo "  Run this on a Nibi login or compute node, not locally." >&2
    return 1 2>/dev/null || exit 1
fi

export HEATWAVE_REPO="${HEATWAVE_REPO:-$HOME/heatwave-initial-conditions}"
# HEATWAVE_ROOT re-roots every relative output path in the configs, so runs
# write to scratch instead of into the checkout. The repo's data/ and plots/
# layout is preserved underneath it.
export HEATWAVE_ROOT="${HEATWAVE_ROOT:-$HEATWAVE_SCRATCH/heatwave}"
export HEATWAVE_VENV="${HEATWAVE_VENV:-$HOME/venvs/heatwave}"
export HEATWAVE_PERSIST="${HEATWAVE_PERSIST:-$HOME/projects/$SBATCH_ACCOUNT/$USER/heatwave_atlas}"

# Every path above keeps an existing value, so a shell that sourced a broken
# version of this file carries the bad value forward. Fail with the cure
# rather than a bare mkdir error.
if ! mkdir -p "$HEATWAVE_ROOT" "$HEATWAVE_PERSIST" "$HEATWAVE_REPO/logs/nibi" \
        2>/dev/null; then
    echo "hpc/nibi/env.sh: cannot create HEATWAVE_ROOT=$HEATWAVE_ROOT" >&2
    echo "  or HEATWAVE_PERSIST=$HEATWAVE_PERSIST." >&2
    echo "  If either looks wrong, this shell holds a stale value. Run:" >&2
    echo "    unset HEATWAVE_ROOT HEATWAVE_PERSIST HEATWAVE_REPO HEATWAVE_VENV" >&2
    echo "  then source this file again." >&2
    return 1 2>/dev/null || exit 1
fi

# --- modules --------------------------------------------------------------
# The venv was built against these, and the wheelhouse jaxlib links against
# the cuda/cudnn modules at runtime. A job that activates the venv without
# loading them either fails to import jaxlib or quietly falls back to CPU.
#
# cuda/12.9 is pinned deliberately. The unversioned `cuda` resolves to 12.6.77
# on Nibi, and XLA warns that ptxas at or below 12.6.2 miscompiles some
# clamping edge cases. cuda/13.2 is not an option: the wheelhouse plugin is
# jax_cuda12_*, so it needs a CUDA 12 runtime.
#
# Slurm does not pass the `module` shell function into a batch job, so Lmod
# has to be bootstrapped there. Neither Alliance's profile script nor the
# module function is written for `set -euo pipefail`: bash.sh reads
# SKIP_CC_CVMFS with no default, which aborts instantly under `set -u`. Relax
# both options across this block and restore whatever was set before.
_hw_shopts="$-"
set +eu

if ! command -v module >/dev/null 2>&1; then
    for _hw_init in /cvmfs/soft.computecanada.ca/config/profile/bash.sh \
                    /etc/profile.d/z-00-lmod.sh \
                    /etc/profile.d/modules.sh \
                    "${LMOD_PKG:-/nonexistent}/init/bash"; do
        if [ -r "$_hw_init" ]; then
            # shellcheck disable=SC1090
            . "$_hw_init"
            command -v module >/dev/null 2>&1 && break
        fi
    done
    unset _hw_init
fi

module load StdEnv/2023 python/3.11 cuda/12.9 cudnn >/dev/null
_hw_module_rc=$?

case "$_hw_shopts" in *e*) set -e ;; esac
case "$_hw_shopts" in *u*) set -u ;; esac
unset _hw_shopts

if ! command -v module >/dev/null 2>&1; then
    echo "hpc/nibi/env.sh: the module command is unavailable and no Lmod init" >&2
    echo "  script was found. Cannot load python/cuda for the venv." >&2
    return 1 2>/dev/null || exit 1
fi
if [ "$_hw_module_rc" -ne 0 ]; then
    echo "hpc/nibi/env.sh: module load failed with status $_hw_module_rc." >&2
    unset _hw_module_rc
    return 1 2>/dev/null || exit 1
fi
unset _hw_module_rc

# --- runtime --------------------------------------------------------------
export MPLBACKEND=Agg                  # no display on a compute node
export HDF5_USE_FILE_LOCKING=FALSE     # netCDF writes on the parallel filesystem
# JAX preallocates 75% of the device by default, which keeps the allocator
# from fragmenting during the long backward pass. calibrate.sh turns it off
# for itself, so nvidia-smi there reports real usage rather than the
# reservation.
