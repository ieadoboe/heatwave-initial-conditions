# common.py
"""
Common utilities for plotting and configuration.

Usage:
    from common import mpl_apply
    mpl_apply()
"""

import matplotlib as mpl

MPL_CONFIG = {
    'savefig.dpi': 300,  # Publication quality
    'figure.dpi': 150,   # Screen display
    'font.family': 'serif',  # Font family
    'font.size': 11,  # Global font size
    'font.serif': ['Computer Modern'],
    'text.usetex': True,  # Enable LaTeX rendering
    'axes.labelsize': 10,  # Font size for axis labels
    'axes.titlesize': 11,  # Font size for subplot titles
    'xtick.labelsize': 9,  # Legend text size
    'ytick.labelsize': 9,  # Legend text size
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.linewidth': 0.8,
    'savefig.bbox': 'tight',
    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
}


def mpl_apply(config: dict | None = None) -> None:
    """
    Apply Matplotlib configuration globally.

    Parameters
    ----------
    config : dict, optional
        Dictionary of rcParams to apply. Defaults to MPL_CONFIG.
    """
    cfg = config if config is not None else MPL_CONFIG
    mpl.rcParams.update(cfg)
