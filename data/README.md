# data/

Small tracked results and inputs for the storyline work. Large and
regenerable files (ERA5 downloads, optimizer runs, ensembles, IC stores) are
git-ignored; see `.gitignore`. The heat-wave catalogue and its tables live in
the heatwave-catalogue repository.

| File | What it is | Written by |
|---|---|---|
| `atlas_summary.csv` | One row per storyline run: settings, final loss, storyline gain, ensemble comparison | `scripts/run_atlas.py`, combined by `scripts/collect_summaries.py` |
| `ensemble_summary.csv` | One row per event's control ensemble: member count, peak statistics | `scripts/run_ensembles.py`, combined by `scripts/collect_summaries.py` |
| `storyline_events.csv` | Every storyline config's dates, start date, run length and target window | `scripts/export_storyline_events.py` |
| `heatwave_event_candidates.csv` | The candidate events considered for the original seven-event atlas, with references | hand-curated (static) |
| `pnw_box_t1000_jja_1981_2020.csv` | ERA5 June-August temperature over the PNW target box, the climatology behind the Eq. 4 threshold | `heatwave_ic/intensity.py` (cached on first use) |
| `pnw_jun2021_vs_paper.csv` | The PNW reproduction set against Whittaker and Di Luca's published numbers | `notebooks/pnw_reproduction_vs_paper.ipynb` |
| `stjohns_storyline_best.csv` | Unperturbed and optimized temperature series of the best St. John's tuning run | `notebooks/heatwave_stjohns_hyperparam_tuning.ipynb` |
| `koppen_subzones.txt` | Reference table of the Köppen-Geiger classes and their criteria | hand-written (static) |
| `heatwave_events.csv` | Single-point St. John's CTX90pct events from the early exploratory work | `archive/scripts/identify_heatwaves.py` |
| `heatwave_summary.csv` | Yearly summary of those St. John's events | `archive/scripts/identify_heatwaves.py` |
