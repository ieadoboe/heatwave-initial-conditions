"""heatwave_ic — reusable pipeline for differentiable heat-wave IC optimization.

Event-agnostic refactor of scripts/stjohns_optimize.py +
notebooks/heatwave_stjohns_hyperparam_tuning.ipynb (which are themselves
adapted from Tim Whittaker's ExtremeStorylines / Whittaker & Di Luca 2026).
An "event" (St. John's Aug 2025, PNW Jun 2021, ...) is a YAML file in
configs/; notebooks call these functions instead of re-writing the pipeline.

Typical notebook usage (GPU + GCS access, i.e. Colab):

    from heatwave_ic import (load_config, load_model, build_ic_zarr,
                             load_ic_on_model_grid, optimize_event)
    cfg = load_config("configs/stjohns_aug2025.yaml")
    model = load_model(cfg["model_name"])
    build_ic_zarr(model, cfg)                      # once per event/init date
    eval_era5 = load_ic_on_model_grid(model, cfg["paths"]["ic_zarr"])
    result = optimize_event(model, eval_era5, cfg)          # tuned defaults
    sweep  = optimize_event(model, eval_era5, cfg, beta=20, iterations=30)
"""

from heatwave_ic.config import ARCO_ERA5_PATH, load_config, resolve_config, describe
from heatwave_ic.model import load_model
from heatwave_ic.data import (
    open_arco_era5,
    build_ic_zarr,
    load_ic_on_model_grid,
    make_regridder,
    regrid_window,
    shift_forcings,
)
from heatwave_ic.optimize import (
    WEIGHT_KEYS,
    encode_initial_state,
    extract_non_diff,
    reconstruct_full_state,
    target_indices,
    make_loss_fn,
    optimize_ic,
    optimize_event,
)
from heatwave_ic.evaluate import (
    box_t1000_trajectory,
    daily_max_t1000_at,
    event_skill,
    run_leadtime_sweep,
    unroll_to_xarray,
)
from heatwave_ic.outputs import (
    make_run_dir,
    save_losses,
    save_state_fields,
    save_trajectory_nc,
)
from heatwave_ic.zones import (
    GROUP_NAMES,
    classify_event,
    koppen_class,
    koppen_class_modal,
    koppen_group,
)
from heatwave_ic import plots
