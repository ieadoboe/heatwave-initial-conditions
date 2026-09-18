# Running the atlas on Nibi

Nibi is the SHARCNET H100 cluster at Waterloo. You need your own Alliance
account and an allocation on it. The examples assume a `nibi` host alias in
`~/.ssh/config` with connection multiplexing, so one Duo approval covers a
whole working session:

```
Host nibi
    HostName nibi.alliancecan.ca
    User <your-alliance-username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
```

Without the alias, pass the destination to `push.sh` explicitly and set
`REMOTE_HOST` for `pull.sh`.

Nibi is chosen here because all of its nodes reach the internet, so the
pipeline's lazy reads from Google Cloud Storage (the NeuralGCM checkpoint,
ARCO-ERA5) behave the way they do on Colab.
Narval and Rorqual block outbound traffic from compute nodes and would need
every input staged by hand first.

The work runs as three SLURM job arrays, one array task per event, seven events
per array. Each task is independent, so a failure in one zone leaves the other
six alone. A fourth script, `04_pnw_reproduction.sbatch`, is a single
validation job rather than an array.

## Before the first run

1. Request access to Nibi in CCDB, under **Resources > Access Systems**. It can
   take an hour to take effect.
2. Sourcing `env.sh` works out your allocation from `sacctmgr`. When you hold
   more than one, it stops and asks you to pick:

   ```bash
   export SBATCH_ACCOUNT=def-yourpi
   ```

   What it wants is the BASE name, not the `def-yourpi_cpu` and
   `def-yourpi_gpu` associations `sacctmgr` lists: Slurm picks the right one
   from what each job requests. Sourcing `env.sh` also creates
   `HEATWAVE_PERSIST` for you, because Nibi doesn't make per-user directories
   under `/project` by default.
3. Copy the repository to Nibi and build the environment on a login node:

   ```bash
   bash hpc/nibi/push.sh          # from your laptop
   ssh nibi
   cd heatwave-initial-conditions
   source hpc/nibi/env.sh
   bash hpc/nibi/setup_env.sh
   ```

   `push.sh` skips outputs (`data/`, `plots/`, `logs/`) but carries
   `data/koppen_*`, which is a static input rather than an output:
   `zones.py` reads the Köppen grid from the checkout, and every event fails
   at `classify_event` without it. Use it for later syncs too.

   Alliance clusters discard manylinux wheels from PyPI, so jax, numpy, pandas,
   xarray and matplotlib come from the local wheelhouse with `--no-index`,
   while neuralgcm and dinosaur-dycore are pure Python and install from PyPI.
   What that resolves to, as of 2026-09-07: jax and jaxlib 0.10.2, numpy 2.4.2,
   pandas 3.0.5, xarray 2026.7.0, zarr 3.1.6, matplotlib 3.11.1, optax 0.2.8,
   neuralgcm 1.2.2, dinosaur 1.5.0. `pip freeze` lands in
   `hpc/nibi/requirements-nibi.txt`.

   The wheelhouse `jaxlib` is a CPU build. GPU support is a separate PJRT
   plugin, `jax_cuda12_plugin` and `jax_cuda12_pjrt`, whose version has to
   match jax and jaxlib exactly, so `setup_env.sh` pins all four to
   `JAX_VERSION`. Without the plugin, jobs run on CPU and say so only in one
   easily-missed line: "An NVIDIA GPU may be present on this machine, but a
   CUDA-enabled jaxlib is not installed.".

   `env.sh` loads `StdEnv/2023 python/3.11 cuda/12.9 cudnn`, which every job
   needs:
   the venv points at the module's Python and the wheelhouse jaxlib links
   against the module's CUDA at runtime. Activating the venv without them
   either breaks the jaxlib import or falls back to CPU without saying so.
   The CUDA version is pinned: unversioned `cuda` gives 12.6.77 on Nibi, and
   XLA warns that ptxas at or below 12.6.2 miscompiles some clamping edge
   cases. Don't move to `cuda/13.2`; the plugin is `jax_cuda12_*` and needs a
   CUDA 12 runtime.

## Where things go

| Variable | Default | Holds |
|---|---|---|
| `HEATWAVE_ROOT` | `$SCRATCH/heatwave` | `data/` and `plots/` for the runs |
| `HEATWAVE_PERSIST` | a `/project` directory | run dirs and streamed members |
| `HEATWAVE_VENV` | `$HOME/venvs/heatwave` | the Python environment |

`HEATWAVE_ROOT` re-roots every relative path in the event configs, so nothing
writes into the checkout. Scratch on Nibi has a 1 TB soft limit with a 60-day
grace period, which the runs stay well inside, but scratch is not backed up:
anything you want to keep belongs under `HEATWAVE_PERSIST`.

## Running

Submit from the repository root, so `$SLURM_SUBMIT_DIR` and the log paths
resolve.

```bash
source hpc/nibi/env.sh

# 1. Download the seven initial conditions (CPU, about an hour each at most).
sbatch hpc/nibi/01_build_ics.sbatch

# 2. Measure one event, then set --time in the two GPU scripts from what it
#    prints. Use the longest event: evol_days 11.
bash hpc/nibi/calibrate.sh

# 3. Validate against the paper before trusting anything else: PNW alone,
#    failing the job if the storyline gain misses W&DL's +3.7 C.
sbatch hpc/nibi/04_pnw_reproduction.sbatch

# 4. The optimizations, one event per task. Each task checks its own run and
#    fails if the first Adam step blew up the regularization term.
sbatch hpc/nibi/02_run_atlas.sbatch

# 5. EXP75: the 75-member ensembles. Needs step 4's storyline.csv files.
sbatch hpc/nibi/03_run_ensembles.sbatch

# 6. Combine the seven per-event tables into one.
"$HEATWAVE_VENV/bin/python" scripts/collect_summaries.py --persist-dir "$HEATWAVE_PERSIST"
```

To see where things stand at any point, run `bash hpc/nibi/status.sh`. It
reports the queue, jobs that finished since yesterday, and then the progress
each stage can be measured by on disk: initial conditions built, events with a
storyline and its gain, and members written per ensemble. The file counts are
the ones that matter, because a job can exit clean having skipped its work.

For the raw Slurm view, `squeue -u $USER` covers what is queued or running and
`sacct -u $USER -S today -X` covers what finished. The `-X` suppresses the
`.batch` and `.extern` sub-steps. Per-task output is under `logs/nibi/`.

Each array task writes its own `data/atlas_summary_<event>.csv` and
`data/ensemble_summary_<event>.csv`. Seven tasks running at once would
otherwise overwrite one another's rows in a single shared file, so step 6 is
what produces `atlas_summary.csv` and `ensemble_summary.csv`. It runs on a
login node in a second, and only needs pandas.

If you already have the optimization runs in Google Drive, you can skip step 4
by copying them in instead:

```bash
rsync -a <local>/heatwave_atlas/opt_runs/ nibi:$HEATWAVE_PERSIST/opt_runs/
```

Those runs were produced before the forcing-shift fix in `build_ic_zarr`, so
their initial conditions sit 24 hours later than their labels, and before the
regularization reference scale changed. Re-running step 4 keeps the optimizations and the ensembles on the same initial condition.

## GPU sizing

The scripts request `h100_3g.40gb`, three eighths of an H100 with 40 GB. A
MIG slice costs a fraction of the 12.2 reference GPU units a whole H100
charges against your allocation, and MIG jobs start sooner because roughly
half of Nibi's GPU nodes are partitioned. The ensemble members are
forward-only and small. If the optimization runs out of memory
backpropagating through a 264-step unroll, raise the request to `h100:1`.

`calibrate.sh` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX grows its
allocation on demand and `nvidia-smi` reports real usage during calibration.
The batch jobs leave JAX's default preallocation on.

## Checking a run

`scripts/check_run.py` reads `losses.npy`, `box_T_K.npy` and `reg.npy` from a
run directory and fails when the optimization did not converge. It needs numpy
only, so it runs on a login node:

```bash
python scripts/check_run.py                                  # every run
python scripts/check_run.py --summary data/atlas_summary_pnw_jun2021.csv
python scripts/check_run.py --expect-gain 3.7 data/opt_runs/pnw_*
```

What it looks at is iteration 1. `reg[0]` is zero because nothing has moved
yet, so `reg[1]` is the work of the first Adam step, and that step moves every
parameter by exactly the learning rate whatever the gradients say. Too large a
learning rate therefore shows up there, as a penalty larger than the entire
heat term, and the run spends its remaining iterations undoing it. Nothing
crashes when this happens: the run writes a complete, well-formed set of
outputs whose gains are not minima. Steps 3 and 4 above both end with this
check, so a bad run fails its task and shows up in `sacct`.

## Resuming

Both GPU steps are resume-aware, which matters because a job that hits its
walltime is killed outright.

- The atlas skips any event that already has `optimized.nc` in its run
  directory, so resubmitting the array picks up where it stopped.
- The ensembles stream each member to
  `$HEATWAVE_PERSIST/ensembles/EXP75/<event>/member_NNN.csv` as it finishes and
  read completed members back, so a resubmission resumes at the exact member it
  died on. Splitting an event across several shorter jobs is therefore free.

## Collecting the results

Run this on your laptop. It cannot ask Slurm from there, so tell it which
allocation the results sit under (`echo $HEATWAVE_PERSIST` on the cluster
prints the full path if you would rather pass `REMOTE_DIR`):

```bash
export REMOTE_ACCOUNT=def-yourpi      # once per shell

bash hpc/nibi/pull.sh                 # summaries, storylines, arrays, figures
bash hpc/nibi/pull.sh --with-nc       # add the trajectory netCDFs (GBs)
bash hpc/nibi/pull.sh --with-members  # add the 525 raw member CSVs
```

The default skips the trajectory netCDFs, which are several GB per event and
which nothing in the analysis reads until you want to map a field. Everything
lands under `data/` and `plots/` per the project convention.

`collect_summaries.py` merges the ensemble metrics into `atlas_summary.csv`, so
that one table carries both the storyline gain and the natural-variability
yardstick for every zone.
