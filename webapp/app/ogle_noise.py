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

# Minimum fraction of a light curve that must survive the observing schedule. OGLE
# only catalogues an event it observed often enough to detect, so a curve with (next
# to) no points is not something the real survey could ever produce. Enforced by
# redrawing the schedule, which mirrors that detection requirement rather than
# distorting the underlying cadence distribution.
_MIN_OBS_FRAC: float = 0.05
_MAX_SCHEDULE_TRIES: int = 20

# Empirical OGLE-IV (I_s, f_s) pairs from blend_model.npz -- the source magnitude
# and its blend fraction, taken from the SAME catalogue event so their real
# correlation is preserved (brighter sources are less blended). The observed
# baseline follows as I_base = I_s + 2.5*log10(f_s).
_BLEND_AVAILABLE = False
_FS_BLEND: np.ndarray = np.array([1.0])
_IS_BLEND: np.ndarray = np.array([19.5])

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

try:
    _bm = np.load(_NOISE_DIR / "blend_model.npz")
    _FS_BLEND = _bm["fs"]
    _IS_BLEND = _bm["Is"]          # paired element-wise with _FS_BLEND
    _BLEND_AVAILABLE = True
except (FileNotFoundError, KeyError):
    warnings.warn(
        "OGLE blend model not found or outdated (blend_model.npz missing the "
        "paired Is array). Light curves will not be blend-diluted — re-run "
        "noise_analysis/blend_model.py.",
        UserWarning,
        stacklevel=2,
    )


def sample_blend_pair(n: int, rng: np.random.Generator):
    """Draw ``n`` paired (I_s, f_s) values from the OGLE-IV event catalogue.

    Whole catalogue ROWS are bootstrap-resampled, so the source magnitude and its
    blend fraction always come from the same real event. This preserves their
    correlation -- brighter sources are measurably less blended -- which drawing
    the two independently would destroy.

    Returns ``None`` if the blend model is unavailable, so the caller can fall
    back to the idealised unblended light curve.
    """
    if not _BLEND_AVAILABLE or n <= 0:
        return None
    idx = rng.integers(0, len(_FS_BLEND), size=n)
    return _IS_BLEND[idx], _FS_BLEND[idx]


def _compute_sigma(I_vals: np.ndarray) -> np.ndarray:
    """sigma(I) = sqrt(sigma_floor^2 + sigma_phot0^2 * 10^(0.4*(I - I_ref)))"""
    return np.sqrt(
        _SIGMA_FLOOR ** 2
        + _SIGMA_PHOT0 ** 2 * 10.0 ** (0.4 * (I_vals - _I_REF))
    )


def _cadence_mask(n_time: int, t_E: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of shape (n_time,): True where OGLE-IV would observe.

    Guarantees a minimum coverage. The bootstrapped schedule can otherwise leave a
    short event (small t_E, hence a short window) almost or completely unobserved:
    the gap distribution has a tail reaching ~100 days, so a single unlucky first
    draw can skip a whole 11-day window and return an all-NaN light curve. Such a
    curve cannot exist in the real data -- OGLE only catalogues an event if it
    actually observed it enough times to detect it. The schedule is therefore
    redrawn until the minimum is met, which mirrors that detection requirement.
    """
    if n_time <= 1:
        return np.ones(n_time, dtype=bool)

    T_span = max(6.0 * t_E, 1e-6)          # physical time span [days]
    dt_grid = T_span / (n_time - 1)
    tol = max(dt_grid / 2.0, 10.0 / 1440.0)  # 10-minute floor
    t_grid = np.linspace(0.0, T_span, n_time)

    min_obs = max(3, int(round(_MIN_OBS_FRAC * n_time)))
    n_draws = int(np.ceil(T_span / _MEDIAN_DT)) + 50

    best_mask: np.ndarray | None = None
    best_dist: np.ndarray | None = None

    for _ in range(_MAX_SCHEDULE_TRIES):
        dt_samples = rng.choice(_DT_INSEASON, size=max(n_draws, 1), replace=True)
        t_obs = np.cumsum(dt_samples)  # synthetic obs times relative to window start

        idx = np.searchsorted(t_obs, t_grid)
        idx_lo = np.clip(idx - 1, 0, len(t_obs) - 1)
        idx_hi = np.clip(idx,     0, len(t_obs) - 1)
        dist_lo = np.abs(t_grid - t_obs[idx_lo])
        dist_hi = np.abs(t_grid - t_obs[idx_hi])
        dist = np.minimum(dist_lo, dist_hi)
        mask = dist <= tol

        if mask.sum() >= min_obs:
            return mask
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask, best_dist = mask, dist

    # Fallback, effectively never reached: no schedule cleared the bar in
    # _MAX_SCHEDULE_TRIES attempts. Keep the best one and top it up to min_obs with
    # the grid points that came closest to a scheduled observation, so the curve is
    # never left empty.
    assert best_mask is not None and best_dist is not None
    mask = best_mask.copy()
    if mask.sum() < min_obs:
        mask[np.argsort(best_dist)[:min_obs]] = True
    return mask


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
