"""Standard figures for event-optimization runs (plots/ per project convention).

Each function returns (fig, ax); pass save=<path> to also write the file."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project plotting convention (CLAUDE.md): rcParams come from scripts/common.py.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from common import mpl_apply  # noqa: E402

mpl_apply()


def _finish(fig, save):
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
        print(f"Plot -> {save}")
    return fig


def plot_loss(result: dict, title: str = "IC optimization --- loss", save=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result["losses"], marker="o", ms=3)
    ax.set(xlabel="iteration", ylabel="loss", title=title)
    ax.grid(alpha=0.3)
    return _finish(fig, save), ax


def plot_storyline(traj_original: pd.Series, traj_optimized: pd.Series,
                   event_start, event_end, title: str = "Storyline", save=None):
    """Box-mean T1000 trajectories: unperturbed vs optimized ICs."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(traj_original.index, traj_original.values, "k", lw=2,
            label="unperturbed (ERA5 IC)")
    ax.plot(traj_optimized.index, traj_optimized.values, "C3", lw=2,
            label="optimized IC")
    ax.axvspan(pd.Timestamp(event_start), pd.Timestamp(event_end),
               color="red", alpha=0.08, label="event window")
    ax.set(xlabel="date", ylabel=r"box-mean T$_{1000}$ ($^\circ$C)", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save), ax


def plot_leadtime_spaghetti(traj: dict, era5_truth: pd.Series,
                            event_start, event_end,
                            title: str = "Forecasts by lead time", save=None):
    """One line per lead time vs the ERA5 truth series."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(era5_truth.index, era5_truth.values, color="black", lw=2.5,
            label="ERA5 (truth)", zorder=10)
    leads = list(traj.keys())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(leads)))
    for c, lead in zip(cmap, leads):
        s = traj[lead]
        ax.plot(s.index, s.values, color=c, lw=1.3, alpha=0.9,
                label=f"init $-${lead} d")
    ax.axvspan(pd.Timestamp(event_start), pd.Timestamp(event_end),
               color="red", alpha=0.08, label="event window")
    ax.set_ylabel(r"daily-max T$_{1000}$ ($^\circ$C)")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, save), ax


def plot_ensemble_storyline(ens: pd.DataFrame, traj_original: pd.Series,
                            traj_optimized: pd.Series, event_start, event_end,
                            title: str = "Storyline vs ensemble", save=None):
    """Optimized-IC storyline against the stochastic-ensemble envelope:
    thin gray members, black control, red optimized trajectory."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    t = pd.to_datetime(ens.index)
    for col in ens.columns:
        ax.plot(t, ens[col].values, color="0.75", lw=0.5, alpha=0.5, zorder=1)
    ax.plot([], [], color="0.75", lw=1.2,
            label=f"stochastic ensemble ({ens.shape[1]} members)")
    ax.plot(pd.to_datetime(traj_original.index), traj_original.values,
            "k", lw=2, label="unperturbed (control seed)", zorder=3)
    ax.plot(pd.to_datetime(traj_optimized.index), traj_optimized.values,
            "C3", lw=2, label="optimized IC", zorder=4)
    ax.axvspan(pd.Timestamp(event_start), pd.Timestamp(event_end),
               color="red", alpha=0.08, label="event window", zorder=0)
    ax.set(xlabel="date", ylabel=r"box-mean T$_{1000}$ ($^\circ$C)", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save), ax


def plot_koppen_map(events: dict | None = None, extent=None, save=None,
                    title="Koppen-Geiger climate classes, 1991-2020 (Beck et al. 2023)"):
    """Global (or zoomed) map of the 30 Koppen-Geiger sub-zones in the official
    Beck et al. colors, with optional event markers.

    events: {label: (lat, lon)} — white-filled ringed markers with labels
            (lon in signed degrees East).
    extent: (lon_min, lon_max, lat_min, lat_max) to zoom a region.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    from heatwave_ic.zones import KOPPEN_COLORS, koppen_grid

    classes, _, _, codes = koppen_grid()
    cmap = ListedColormap(["#ffffff"] + [KOPPEN_COLORS[c] for c in codes[1:]])
    norm = BoundaryNorm(np.arange(-0.5, 31.5), cmap.N)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.imshow(classes, cmap=cmap, norm=norm, interpolation="nearest",
              extent=[-180, 180, -90, 90], origin="upper")
    if events:
        for name, (la, lo) in events.items():
            lo = ((lo + 180.0) % 360.0) - 180.0
            ax.scatter([lo], [la], s=55, facecolor="white", edgecolor="black",
                       linewidth=1.2, zorder=5)
            ax.annotate(name, (lo, la), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color="black",
                        bbox=dict(facecolor="white", alpha=0.75,
                                  edgecolor="none", pad=1.5))
    if extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set(xlabel="longitude", ylabel="latitude", title=title)
    handles = [Patch(facecolor=KOPPEN_COLORS[c], label=c) for c in codes[1:]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=10, fontsize=7, frameon=False, handlelength=1.2,
              columnspacing=1.0)
    return _finish(fig, save), ax


def plot_skill_vs_lead(skill: pd.DataFrame, save=None):
    """Peak error and event-window RMSE vs lead time (short lead on the right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.axhline(0, color="grey", lw=1, ls="--")
    ax1.plot(skill["lead_days"], skill["peak_err_C"], marker="o")
    ax1.set(xlabel="lead time (days before peak)",
            ylabel="peak Tmax error ($^\\circ$C)\nforecast $-$ ERA5",
            title="Peak error vs lead time")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.3)
    ax2.plot(skill["lead_days"], skill["window_rmse_C"], marker="s", color="C3")
    ax2.set(xlabel="lead time (days before peak)",
            ylabel="event-window RMSE ($^\\circ$C)",
            title="Event-window RMSE vs lead time")
    ax2.invert_xaxis()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return _finish(fig, save), (ax1, ax2)


def plot_europe_heatwaves(hw_days, lat, lon, events=None, top_n: int = 8,
                          year: int = 2026, save=None):
    """Map of heat-wave days per grid cell, with the largest events marked.

    hw_days: (lat, lon) count of days each cell spent in a detected event.
    events:  the event_table DataFrame; the top_n by severity are labelled at
             their centroids."""
    from matplotlib.colors import BoundaryNorm

    counts = np.where(hw_days > 0, hw_days, np.nan)
    vmax = max(6, int(np.nanmax(counts)) if np.isfinite(counts).any() else 6)
    levels = np.arange(0, vmax + 2)

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(counts, origin="upper", cmap="inferno_r",
                   norm=BoundaryNorm(levels, 256),
                   extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                   interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, label="heat-wave days", shrink=0.85, pad=0.02)

    if events is not None and len(events):
        for _, e in events.head(top_n).iterrows():
            ax.scatter([e.centroid_lon], [e.centroid_lat], s=48,
                       facecolor="white", edgecolor="black", linewidth=1.1,
                       zorder=5)
            ax.annotate(f"{int(e['rank'])}. {e['peak']}",
                        (e.centroid_lon, e.centroid_lat), xytext=(6, 5),
                        textcoords="offset points", fontsize=7.5,
                        bbox=dict(facecolor="white", alpha=0.8,
                                  edgecolor="none", pad=1.4), zorder=6)
    ax.set(xlabel="longitude", ylabel="latitude",
           title=f"European heat waves {year}: days above the calendar-day "
                 f"90th percentile\n(CTX90pct, 1991-2020 baseline; "
                 f"numbers mark the {top_n} most severe events)")
    return _finish(fig, save), ax
