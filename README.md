# heatwave-initial-conditions

How much hotter could a heat wave plausibly have been?

Rather than perturbing the initial state at random and waiting for a hot
member, this differentiates simulated temperature back through
[NeuralGCM](https://github.com/neuralgcm/neuralgcm) to the initial condition
and takes gradient steps that make the event hotter, penalized to keep the new
initial condition close to the observed one. The result is a **storyline gain**
in °C over the observed peak. Method from Whittaker & Di Luca
([2026](https://doi.org/10.5194/wcd-7-393-2026)), reproduced on their Pacific
Northwest 2021 case, then extended across Köppen zones and to the 2026
European heat waves.

## Getting started

```bash
uv sync
source .venv/bin/activate          # later blocks assume this
python scripts/run_atlas.py --help
```

Analysis runs on a laptop. The optimization needs a GPU
and outbound network access. Real runs go on **Nibi**, the SHARCNET H100
cluster, whose environment is built by `hpc/nibi/setup_env.sh` rather than
`uv sync`. Read [`hpc/nibi/README.md`](hpc/nibi/README.md) before submitting.

## Pipeline

Each stage is resume-aware, so re-running skips completed work.

```bash
python scripts/build_ics.py       # CPU + network: fetch initial conditions
python scripts/run_atlas.py       # GPU: optimize every event -> data/atlas_summary.csv
python scripts/run_ensembles.py   # GPU: 75-member stochastic baselines
python scripts/check_run.py       # CPU: did it actually converge?
```

`run_atlas.py` puts the PNW event first. If it does not reproduce the paper's
+3.7 °C over the most extreme ensemble member, fix that before reading any
other zone. The ensembles are the natural-variability yardstick the gain has
to beat.

On a cluster these run as job arrays, one task per event. Parallel tasks must
not share an output file, so each writes `data/atlas_summary_<event>.csv` and
`scripts/collect_summaries.py` combines them.

Heat-wave detection lives in its own repository,
[heatwave-catalogue](https://github.com/ieadoboe/heatwave-catalogue): the
`hwcat` package, installed here as a dependency pinned to a tag. Its
`hwcat robustness` run supplies the catalogue that turns detected events into
storyline proposals:

```bash
HWCAT_WORKDIR=../heatwave-catalogue/hwcat-data python scripts/storyline_candidates.py
python scripts/storyline_candidates.py --make-config 2003_r01   # a proposal to review
python scripts/export_storyline_events.py                       # every config's dates
```

## Layout

| Path | Holds |
|---|---|
| `heatwave_ic/` | The library: config, data, model, optimize, ensemble, evaluate, intensity, outputs, pipeline, plots (Köppen zones come from `hwcat`) |
| `configs/` | One YAML per event: box, dates, optimizer and loss hyperparameters |
| `scripts/` | Command-line entry points, one per stage |
| `hpc/nibi/` | SLURM job arrays and cluster environment setup |
| `tests/` | Self-checking scripts. Run them directly; `pytest` collects nothing |
| `data/` | All data outputs, by path relative to the project root |
| `plots/` | All figures, grouped by experiment: `atlas/`, `leadtime/`, `exploratory/` |
| `archive/` | Superseded scripts, notebooks and tools, kept for reference (including the faithful port of Tim's code) |
| `reference/` | Tim Whittaker's released code, read-only |

ERA5 downloads, the percentile climatology, run directories and IC zarr stores
are gitignored: a script rebuilds each, and git would keep every version
forever. Clone and those are empty. The small summary CSVs are tracked.

Any code using `matplotlib.pyplot` calls `mpl_apply()` from
`scripts/common.py` first; extend `MPL_CONFIG` there rather than setting
rcParams inline. `HEATWAVE_ROOT` re-roots every relative output path, which is
how batch jobs write to scratch instead of into the checkout.

## Gotchas

These cost real time to find. Cluster-specific ones are in
[`hpc/nibi/README.md`](hpc/nibi/README.md).

- **Shift forcings before selecting the window, never after.** dinosaur's
  `selective_temporal_shift` truncates the head of what it is given by the
  shift, so selecting first lands the IC 24 h after its label. Wrong until
  2026-09-07, so every earlier atlas result ran from the wrong day.
  (`heatwave_ic/data.py`)
- **The regularization reference scale is `mean(x0^2)`, not `mean(x0)^2`.**
  Deliberate deviation from the paper, 2026-09-16. The published form is
  degenerate for near-zero-mean spectral fields and made the penalty jump to
  1e5 on iteration 1. `archive/scripts/stjohns_optimize.py` keeps the old scale on
  purpose as a faithful port; do not "fix" it. (`heatwave_ic/optimize.py`
  docstring has the full story.)
- **An IC store only needs its first few snapshots.** Consumers read two.
  Writing three instead of a full window is the difference between a few GB
  and ~50 GB per event.
- **`model.data_to_xarray` output carries attributes netCDF cannot store.**
  Write it through `outputs.drop_unwritable_attrs` first.

## Results

`data/atlas_summary.csv` has one row per event: storyline gain, ensemble
spread, and the gain against both.

**The committed table is not current.** Those rows predate the
reference-scale fix, so their gains are not comparable with anything after
2026-09-16, and one event finished on a blow-up rather than a minimum.
Re-run before citing, then check with `scripts/check_run.py`.

## References

Whittaker & Di Luca 2026 (*WCD* 7, 393-410) for the method; Kochkov et al.
2024 (*Nature*) for NeuralGCM; Perkins & Alexander 2013 (*J. Climate* 26) for
CTX90pct; Shepherd et al. 2018 (*Climatic Change* 151) for storylines; Beck et
al. 2023 (*Sci. Data* 10) for the Köppen maps.

MIT licensed, see [LICENSE](LICENSE).
