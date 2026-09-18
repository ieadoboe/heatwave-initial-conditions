#!/bin/bash
# Build the Python environment for the atlas on a Nibi LOGIN node.
#
#     source hpc/nibi/env.sh
#     bash hpc/nibi/setup_env.sh
#
# Two package sources are in play. Alliance clusters ship a local wheelhouse
# and their pip discards manylinux wheels from PyPI, so anything compiled
# (jax, numpy, pandas, xarray, matplotlib) has to come from the wheelhouse
# with --no-index. Pure-Python packages build a py3-none-any wheel, which pip
# accepts from PyPI, so neuralgcm and dinosaur-dycore install normally.
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda/12.9 cudnn

echo "=== wheelhouse contents for the compiled dependencies ==="
avail_wheels jax jaxlib jax_cuda12_plugin jax_cuda12_pjrt \
    numpy pandas xarray zarr matplotlib optax || true
echo "=========================================================="

virtualenv --no-download "$HEATWAVE_VENV"
source "$HEATWAVE_VENV/bin/activate"
pip install --no-index --upgrade pip

# Compiled, from the wheelhouse. The four jax packages are pinned together
# because they have to match: jaxlib alone is CPU, and GPU support lives in
# the separate jax_cuda12_plugin / jax_cuda12_pjrt PJRT plugin. Leaving them
# unpinned lets pip settle on a jax whose plugin version is unavailable, which
# surfaces much later as "a CUDA-enabled jaxlib is not installed".
JAX_VERSION=0.10.2
pip install --no-index \
    "jax==$JAX_VERSION" "jaxlib==$JAX_VERSION" \
    "jax_cuda12_plugin==$JAX_VERSION" "jax_cuda12_pjrt==$JAX_VERSION" \
    numpy pandas xarray zarr matplotlib optax tqdm pyyaml

# Pure Python, from PyPI.
pip install neuralgcm dinosaur-dycore gcsfs fsspec

pip install -e "$HEATWAVE_REPO"
pip freeze --local > "$HEATWAVE_REPO/hpc/nibi/requirements-nibi.txt"

python - <<'PY'
import jax, neuralgcm, dinosaur
print("jax", jax.__version__, "devices:", jax.devices())
print("neuralgcm", neuralgcm.__version__)
import importlib.util
print("cuda plugin installed:",
      importlib.util.find_spec("jax_cuda12_plugin") is not None)
PY

echo
echo "Environment ready at $HEATWAVE_VENV"
echo "jax.devices() lists CPU here because login nodes have no GPU; the cuda"
echo "plugin line above is what matters. Confirm the device with calibrate.sh."
