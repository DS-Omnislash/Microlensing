"""Pre-computed reference distribution curves as Plotly.js-compatible JSON.

Each curve is built by: (1) sampling 100k points via the exact same functions
as distributions.py, then (2) applying kernel-density estimation, giving a
smooth shape that matches the TDR_ROC.pdf histograms without clipping-boundary
spikes. Rendered as a filled area in the web minimalistic style.
"""

import json

import numpy as np
from scipy.stats import gaussian_kde

from .distributions import (
    sample_eccentricity,
    sample_impact_parameter,
    sample_lens_mass,
    sample_lens_source_distance,
    sample_lens_velocity,
    sample_distance_to_lens,
    sample_mass_ratio,
    sample_semi_major_axis,
    sample_trajectory_angle,
)

_N = 100_000
_RNG = np.random.default_rng(42)

_LAYOUT_BASE = {
    "margin": {"l": 44, "r": 8, "t": 8, "b": 40},
    "height": 160,
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "showlegend": False,
    "font": {
        "family": "DM Sans, system-ui, -apple-system, sans-serif",
        "color": "#78786E",
        "size": 10,
    },
    "xaxis": {
        "tickfont": {"size": 9},
        "gridcolor": "#E8E8E4",
        "zeroline": False,
        "showline": False,
    },
    "yaxis": {
        "title": {"text": "Density", "font": {"size": 9}, "standoff": 2},
        "tickfont": {"size": 9},
        "gridcolor": "#E8E8E4",
        "zeroline": False,
        "showline": False,
        "nticks": 4,
    },
}


def _fig(samples, xlabel, x_lo, x_hi, log10=False, bw=None, zero_outside=True):
    """KDE-based filled area chart in the web minimalistic style."""
    data = np.log10(samples) if log10 else np.asarray(samples, dtype=float)

    kw = {"bw_method": bw} if bw is not None else {}
    kde = gaussian_kde(data, **kw)

    x = np.linspace(x_lo, x_hi, 400)
    y = kde(x)

    if zero_outside:
        lo = float(data.min())
        hi = float(data.max())
        y[x < lo] = 0.0
        y[x > hi] = 0.0

    # Close the curve to zero at both ends for a clean fill
    x_plot = np.concatenate([[x_lo], x, [x_hi]])
    y_plot = np.concatenate([[0.0], y, [0.0]])

    layout = {**_LAYOUT_BASE}
    layout["xaxis"] = {
        **_LAYOUT_BASE["xaxis"],
        "title": {"text": xlabel, "font": {"size": 9}, "standoff": 2},
    }

    return {
        "data": [{
            "type": "scatter",
            "x": x_plot.tolist(),
            "y": y_plot.tolist(),
            "mode": "lines",
            "fill": "tozeroy",
            "line": {"color": "#111110", "width": 1.2},
            "fillcolor": "rgba(17,17,16,0.08)",
            "hoverinfo": "skip",
        }],
        "layout": layout,
    }


def _compute_all() -> dict:
    rng = _RNG
    n = _N
    return {
        "M_star_solar": _fig(
            sample_lens_mass(n, rng),
            "Lens Mass (Msun)", x_lo=0.0, x_hi=1.25,
        ),
        "D_l_pc": _fig(
            sample_distance_to_lens(n, rng),
            "D_l (pc)", x_lo=0.0, x_hi=9000.0,
        ),
        "D_ls_pc": _fig(
            sample_lens_source_distance(n, rng),
            "D_ls (pc)", x_lo=0.0, x_hi=8500.0,
        ),
        "v_perp_kms": _fig(
            sample_lens_velocity(n, rng),
            "v_perp (km/s)", x_lo=0.0, x_hi=850.0,
        ),
        "u0": _fig(
            sample_impact_parameter(n, rng),
            "Impact parameter u0", x_lo=0.0, x_hi=1.0,
        ),
        "q": _fig(
            sample_mass_ratio(n, rng),
            "log₁₀(q)", x_lo=-6.5, x_hi=0.5, log10=True,
        ),
        "a_pc": _fig(
            sample_semi_major_axis(n, rng),
            "log₁₀(a) [pc]", x_lo=-8.5, x_hi=-2.5, log10=True,
        ),
        "eccentricity": _fig(
            sample_eccentricity(n, rng),
            "Eccentricity e", x_lo=0.0, x_hi=0.55,
        ),
        "alpha_ref_rad": _fig(
            sample_trajectory_angle(n, rng),
            "α_ref (rad)", x_lo=0.0, x_hi=6.284, zero_outside=False,
        ),
    }


DISTRIBUTION_PLOTS: dict = _compute_all()
DISTRIBUTION_PLOTS_JSON: str = json.dumps(DISTRIBUTION_PLOTS, separators=(",", ":"))