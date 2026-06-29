"""Apply OGLE-IV photometric noise and cadence gaps to I(t) light curves.

Loads the pre-fitted noise model and empirical cadence distribution from
noise_analysis/ (produced by noise_model.py and cadence_model.py) and
applies them to synthetic I(t) light curves to simulate realistic OGLE-IV
observations.

Public API
----------
apply_ogle_imperfections(lightcurves_I, t_E_days, rng) -> np.ndarray
    Adds Gaussian photometric noise sigma(I) and sets unobserved time points
    to NaN based on the OGLE-IV cadence distribution.
"""

import warnings
from pathlib import Path

import numpy as np

_NOISE_DIR = Path(__file__).resolve().parent.parent.parent / "noise_analysis"

_MODELS_AVAILABLE = False
_SIGMA_FLOOR: float = 0.0
_SIGMA_PHOT0: float = 0.0
_I_REF: float = 18.0
_DT_INSEASON: np.ndarray = np.array([0.04])  # fallback: 1-element so code won't crash
_MEDIAN_DT: float = 0.04

try:
    _nm = np.load(_NOISE_DIR / "noise_model.npz")
    _SIGMA_FLOOR = float(_nm["fit_params"][0])
    _SIGMA_PHOT0 = float(_nm["fit_params"][1])
    _I_REF = float(_nm["I_ref"][0])

    _cm = np.load(_NOISE_DIR / "cadence_model.npz")
    _DT_INSEASON = _cm["dt_inseason"]
    _MEDIAN_DT = float(np.median(_DT_INSEASON))

    _MODELS_AVAILABLE = True
except FileNotFoundError as _e:
    warnings.warn(
        f"OGLE noise models not found ({_e}). "
        "OGLE imperfections will be skipped — run noise_analysis/noise_model.py "
        "and noise_analysis/cadence_model.py first.",
        UserWarning,
        stacklevel=2,
    )


def _compute_sigma(I_vals: np.ndarray) -> np.ndarray:
    """sigma(I) = sqrt(sigma_floor^2 + sigma_phot0^2 * 10^(0.4*(I - I_ref)))"""
    return np.sqrt(
        _SIGMA_FLOOR ** 2
        + _SIGMA_PHOT0 ** 2 * 10.0 ** (0.4 * (I_vals - _I_REF))
    )


def _cadence_mask(n_time: int, t_E: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of shape (n_time,): True where OGLE-IV would observe."""
    if n_time <= 1:
        return np.ones(n_time, dtype=bool)

    T_span = max(6.0 * t_E, 1e-6)          # physical time span [days]
    dt_grid = T_span / (n_time - 1)
    tol = max(dt_grid / 2.0, 10.0 / 1440.0)  # 10-minute floor

    n_draws = int(np.ceil(T_span / _MEDIAN_DT)) + 50
    dt_samples = rng.choice(_DT_INSEASON, size=max(n_draws, 1), replace=True)
    t_obs = np.cumsum(dt_samples)  # synthetic obs times relative to window start

    t_grid = np.linspace(0.0, T_span, n_time)

    idx = np.searchsorted(t_obs, t_grid)
    idx_lo = np.clip(idx - 1, 0, len(t_obs) - 1)
    idx_hi = np.clip(idx,     0, len(t_obs) - 1)
    dist_lo = np.abs(t_grid - t_obs[idx_lo])
    dist_hi = np.abs(t_grid - t_obs[idx_hi])
    return np.minimum(dist_lo, dist_hi) <= tol


def apply_ogle_imperfections(
    lightcurves_I: np.ndarray,
    t_E_days: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add OGLE-IV photometric noise and cadence gaps to I(t) light curves.

    Parameters
    ----------
    lightcurves_I : (n_events, n_time) array of I-band magnitudes.
    t_E_days : (n_events,) array of Einstein crossing times in days.
    rng : NumPy Generator for reproducibility.

    Returns
    -------
    Array of same shape. Observed points have Gaussian noise N(0, sigma(I)^2)
    added; unobserved points are NaN.
    """
    if not _MODELS_AVAILABLE:
        warnings.warn(
            "OGLE noise models unavailable — returning light curves unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return lightcurves_I

    n_events, n_time = lightcurves_I.shape
    result = lightcurves_I.copy()

    for i in range(n_events):
        sigma_vals = _compute_sigma(result[i])
        result[i] += rng.normal(0.0, sigma_vals)

        obs_mask = _cadence_mask(n_time, float(t_E_days[i]), rng)
        result[i, ~obs_mask] = np.nan

    return result
