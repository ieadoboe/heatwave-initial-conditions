# heatwave-initial-conditions

Differentiable optimization of initial conditions for heat waves, using
[NeuralGCM](https://github.com/neuralgcm/neuralgcm) as the forward model.

The question is how much hotter a heat wave could plausibly have been. Instead
of perturbing the initial state at random and waiting for a hot member, this
differentiates the simulated temperature back through the model to the initial
condition and takes gradient steps that make the event hotter, subject to a
penalty that keeps the new initial condition close to the observed one. The
result is a *storyline*: a physically consistent worst case for the event that
actually happened, and a **storyline gain** in degrees Celsius over the
observed peak.

The method is that of Whittaker & Di Luca (2026), reproduced here on their
Pacific Northwest June 2021 case and then extended across Köppen climate zones
and to the 2026 European heat waves.

## Getting started

```bash
uv sync
uv run python scripts/detect_heatwaves_europe.py --help
```

The detection and analysis half runs on a laptop: numpy, pandas, xarray, scipy
and matplotlib, no jax. The optimization half needs a GPU and outbound network
access, and does not run usefully on a local CPU environment.

Real runs go on **Nibi**, the SHARCNET H100 cluster. Its environment is built
by `hpc/nibi/setup_env.sh` rather than by `uv sync`, because Alliance clusters
reject PyPI manylinux wheels and the CUDA plugin has to be version-matched to
jax by hand. Read [`hpc/nibi/README.md`](hpc/nibi/README.md) before submitting
anything.

## Layout

| Path | Holds |
|---|---|
| `heatwave_ic/` | The library: config, data, model, optimize, ensemble, evaluate, detect, zones, plots |
| `configs/` | One YAML per event. Event box, dates, optimizer and loss hyperparameters |
| `scripts/` | Command-line entry points, one per pipeline stage |
| `hpc/nibi/` | SLURM job arrays and environment setup for the cluster |
| `tests/` | Self-checking scripts, run directly rather than through pytest |
| `data/` | All data outputs. Large ones are gitignored |
| `plots/` | All figures |

Two conventions the code assumes. Data outputs go in `data/` and figures in
`plots/`, always by a path relative to the project root, never beside the
script that wrote them. Any code using `matplotlib.pyplot` calls
`mpl_apply()` from `scripts/common.py` first, so figures stay consistent;
extend `MPL_CONFIG` there rather than setting rcParams inline.

Setting `HEATWAVE_ROOT` moves every relative output path under another
directory, which is how batch jobs write to scratch instead of into the
checkout.

### What is not in the repository

ERA5 downloads, the percentile climatology, the optimizer run directories and
the initial-condition zarr stores are gitignored, because a script rebuilds
each of them and git would keep every version forever. Clone this and those
directories will be empty; run the scripts to fill them. The small summary
CSVs in `data/` are tracked, since those hold the numbers worth citing.

## The optimization pipeline

Four stages. Each is resume-aware, so re-running skips completed work.

```bash
python scripts/build_ics.py       # CPU + network: fetch the initial conditions
python scripts/run_atlas.py       # GPU: optimize every event
python scripts/run_ensembles.py   # GPU: 75-member stochastic baselines
python scripts/check_run.py       # CPU: did the optimization actually converge
```

`build_ics.py` is split out because it needs the network but no GPU, so a
cluster can run it as a cheap CPU job first. `run_atlas.py` does IC build,
optimize, evaluate, plot and Köppen tagging per event, and writes
`data/atlas_summary.csv`. `run_ensembles.py` unrolls 75 stochastic members
from the unperturbed initial condition, which is the natural-variability
yardstick the storyline gain has to beat, and merges its metrics into the same
table.

On a cluster the three stages run as job arrays, one task per event. Parallel
tasks must not share an output file, so each writes its own
`data/atlas_summary_<event>.csv` and `scripts/collect_summaries.py` combines
them afterwards.

Validate before trusting anything: `run_atlas.py` puts the PNW event first,
and if it does not reproduce the paper's +3.7 °C over the most extreme
ensemble member, fix that before reading any other zone.

## Heat-wave detection

A separate, jax-free entry point that builds the event catalogue the
optimization runs on.

```bash
python scripts/download_era5_europe_daily.py --probe   # validate the request
python scripts/download_era5_europe_daily.py           # baseline + target years
python scripts/detect_heatwaves_europe.py
python tests/test_detect.py
```

`heatwave_ic/detect.py` implements Perkins & Alexander CTX90pct on a grid: a
cell is in a heat wave when its daily maximum exceeds the calendar-day 90th
percentile of a 1991-2020 baseline for at least three consecutive days.
Connected-component labelling in (time, lat, lon) then turns those cell-days
into discrete events. The 2026 European catalogue has 32 events between 1 May
and 5 September.

Downloads come from the CDS `derived-era5-single-levels-daily-statistics`
entry, which aggregates to a daily statistic server-side. The ARCO-ERA5 store
is chunked one hour per chunk, so a Europe-only request from it would still
transfer the whole globe.

## Results

`data/atlas_summary.csv` carries one row per event: the storyline gain, the
ensemble spread, and the gain measured against both.

The committed table is **not current**. Those rows came from runs that used
the old regularization reference scale and the old learning rate, so their
gains are not comparable with anything produced after 2026-09-16, and one
event finished on a blow-up rather than a minimum. Its `run_dir` column also
holds absolute cluster paths. Re-run before citing any of it, and run
`scripts/check_run.py` on what comes back.

## Gotchas

These cost real time to find. The cluster-specific ones are in
[`hpc/nibi/README.md`](hpc/nibi/README.md).

**Shift the forcings before selecting the window, never after.** dinosaur's
`selective_temporal_shift` truncates the head of whatever window it is given
by the shift amount, so a +24 h shift on hourly data drops the first 24
snapshots. Selecting first leaves an initial condition 24 hours after the date
its config claims. `build_ic_zarr` had this backwards until 2026-09-07, so
every atlas result from before then ran from the wrong day.

**An IC store only needs its first few snapshots.** Every consumer goes
through `load_ic_on_model_grid`, which reads two, and `encode_initial_state`
takes its forcings from the first. `build_ic_zarr` writes three by default,
which is the difference between a few GB and about 50 GB per event.

**The regularization reference scale is `mean(x0^2)`, not `mean(x0)^2`.** This
is a deliberate deviation from the paper and from the reference code, made
2026-09-16 and documented in the `heatwave_ic/optimize.py` docstring. The
published form is degenerate for near-zero-mean spectral fields, where
positive and negative lobes cancel and the denominator collapses toward zero.
Combined with Adam, whose first update moves every parameter by exactly the
learning rate regardless of gradient size, that made the penalty jump to 1e5
on iteration 1 and spend the rest of the run shrinking back. Several events
ended above their starting loss. `scripts/stjohns_optimize.py` keeps the old
scale on purpose as a faithful port of the reference code; do not "fix" it.

**In detection, filter each day for spatial extent before the space-time
labelling.** A 90th-percentile threshold puts about a tenth of all cell-days
above it by construction, so scattered exceedances are everywhere, and
labelling first lets them bridge unrelated blobs through single-cell chains.
In testing that turned a planted 5-day event into a 12-day one.
`tests/test_detect.py` plants known events and asserts their durations, so run
it after touching the detection code.

**Datasets from `model.data_to_xarray` carry attributes netCDF cannot
store**, such as dinosaur's `basis_as_jax_arrays=None`. Recent xarray rejects
a None-valued attribute outright, and it also passes booleans through its
validator that the netCDF4 writer then refuses. Anything writing one of these
datasets goes through `outputs.drop_unwritable_attrs` first.

**`data/koppen_geiger_0p1_1991_2020.npz` is a static input, not an output**,
despite living in `data/`. `heatwave_ic/zones.py` reads it from the checkout
at a fixed path, so any sync that excludes `data/` wholesale breaks every
event at `classify_event`.

## Tests

The test files are self-checking scripts with assertions, not a pytest suite,
so run them directly. `pytest tests/` collects nothing.

```bash
python tests/test_detect.py      # detection: persistence, area, blob separation
python tests/test_check_run.py   # convergence checks on run directories
```

## References

- Whittaker, T. & Di Luca, A. (2026). Constructing extreme heatwave storylines
  with differentiable climate models. *Weather and Climate Dynamics*, 7,
  393-410.
- Kochkov, D. et al. (2024). Neural general circulation models for weather and
  climate. *Nature*.
- Perkins, S. E. & Alexander, L. V. (2013). On the measurement of heat waves.
  *Journal of Climate*, 26(13).
- Shepherd, T. G. et al. (2018). Storylines: an alternative approach to
  representing uncertainty in physical aspects of climate change. *Climatic
  Change*, 151.
- Beck, H. E. et al. (2023). High-resolution (1 km) Köppen-Geiger maps for
  1901-2099. *Scientific Data*, 10.

## License

MIT, see [LICENSE](LICENSE).
