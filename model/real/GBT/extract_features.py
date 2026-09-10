"""
Model 1 (Real) -- chi^2 single-lens-fit feature extractor
==========================================================

Turns each OGLE-like light curve into a small table of engineered features whose
centrepiece is the **chi^2 excess of a best-fit single-lens (Paczynski) model**.
This is the classical microlensing anomaly-detection signal (RoboNet / ARTEMiS
style): fit the smooth single-lens curve a real event *would* have if it had no
companion, then measure how badly that fit fails. A true single lens fits well
(low chi^2); a binary/planet leaves a caustic residual the single-lens family
physically cannot absorb (high chi^2).

Why bother, when the CNN already gets a fold-residual channel?
-------------------------------------------------------------
The CNN's binary signal is the fold residual R(tau) = I(tau) - I(-tau), which
only carries information where BOTH tau and -tau were observed. On a curve that
is ~78% cadence gaps, co-observed mirror pairs are a small fraction of an already
sparse curve, and any anomaly whose mirror point fell in a gap is invisible to
it. A single-lens FIT uses *every* observed point -- it never needs a matching
partner -- so it extracts signal from ~3x more of the data. If a gradient-boosted
tree on these features clears the CNN's ~0.55 AUC, the win is this representation
and the natural next step is to feed the chi^2 signal into the CNN as an extra
channel. If it does NOT clear it, we have confirmed the ~0.55 information ceiling
a fourth, representation-independent way (a full-curve matched filter), which is a
stronger result than the three fold-based measurements already on record.

The fit (cheap because the curves are already registered)
---------------------------------------------------------
The dataset stores I(tau) on a FIXED grid tau = linspace(-3, 3, 400), i.e. the
curves are already aligned: t0 sits at tau = 0 and time is in units of t_E. So we
do NOT refit t0 / t_E in physical units -- only the light-curve SHAPE. In flux,
the blended single-lens model is linear:

    F(tau) = F_source * A(u0, tau) + F_blend ,    A = (u^2+2)/(u*sqrt(u^2+4)),
             u = sqrt(u0^2 + tau^2)

For a fixed u0 the best (F_source, F_blend) is a closed-form 2-parameter weighted
least-squares, so the whole fit is a 1-D search over u0 with a linear solve
inside -- fully vectorised over curves, memory-cheap, no per-curve scipy loop.
Non-negativity (F_source, F_blend >= 0) is enforced (2-variable NNLS), otherwise
the single-lens family could cheat by using a negative blend to swallow a caustic
and understate the chi^2 excess.

Residuals are weighted by the project's real OGLE noise model
    sigma(I) = sqrt(sigma_floor^2 + sigma_phot0^2 * 10^(0.4*(I - 18)))
(loaded from noise_analysis/noise_model.npz), so chi^2 is a genuine
noise-normalised statistic, not an arbitrary sum of squared magnitudes.

Output
------
A feature-table CSV (one row per event) written next to this script:

    event_index   row number in the source dataset (ground-truth key)
    event_lenses  1 = single, 2 = binary  (the label, carried through)
    coverage      fraction of the 400 slots observed
    mag_std/skew/kurt   per-curve magnitude moments (trivial baseline features)
    fold_max      max|I(tau)-I(-tau)| over co-observed pairs (the CNN's signal)
    fold_n        number of co-observed mirror pairs
    u0_fit        best-fit single-lens impact parameter
    n_obs         observed points used in the fit
    chi2          weighted residual sum of squares of the best single-lens fit
    chi2_dof      chi2 / (n_obs - 2)   <-- the headline anomaly feature
    max_abs_r     largest standardised residual |F-model|/sigma_F
    tau_absmax    |tau| where that peak residual falls (off-centre anomalies)
    n_r_gt3       number of points with standardised residual > 3
    pos_excess    sum of positive standardised residual^2 (caustics brighten)
    resid_span    tau-range spanned by residuals over 2 sigma

With --report it also prints the univariate ROC-AUC of each feature against the
label, i.e. an immediate answer to "does the chi^2 fit separate the classes, and
by how much more than the fold feature?" -- before any LightGBM is trained.

Run
---
    venv/Scripts/python.exe model/real/GBT/extract_features.py --report
    venv/Scripts/python.exe model/real/GBT/extract_features.py \
        --max-events 20000 --report          # quick subset scout
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REAL_DIR = HERE.parent / "CNN"                      # where the training CSV lives
NOISE_NPZ = HERE.parents[2] / "noise_analysis" / "noise_model.npz"

N_POINTS = 400
TAU = np.linspace(-3.0, 3.0, N_POINTS, dtype=np.float64)

# Single-lens fit search grid for the impact parameter u0. u0 is drawn TruncExp on
# [0, 1] in the generator; the upper end is padded a little because a noisy/blended
# curve can be best matched by a slightly larger u0.
U0_GRID = np.geomspace(1e-3, 2.0, 64)

# OGLE noise model sigma(I) = sqrt(floor^2 + phot0^2 * 10^(0.4*(I - I_ref))).
# Loaded from the fitted npz so this stays consistent with the generator; the
# hard-coded fallback matches noise_analysis/noise_model.npz at time of writing.
_SIGMA_FLOOR, _SIGMA_PHOT0, _I_REF = 0.0001, 0.026192, 18.0
try:
    _nm = np.load(NOISE_NPZ)
    _SIGMA_FLOOR = float(_nm["fit_params"][0])
    _SIGMA_PHOT0 = float(_nm["fit_params"][1])
    _I_REF = float(_nm["I_ref"][0])
except (FileNotFoundError, KeyError):
    pass


def sigma_I(I_vals: np.ndarray) -> np.ndarray:
    """OGLE photometric scatter sigma(I) at magnitude I (same model as generator)."""
    return np.sqrt(_SIGMA_FLOOR ** 2
                   + _SIGMA_PHOT0 ** 2 * 10.0 ** (0.4 * (I_vals - _I_REF)))


def find_dataset(folder: Path) -> Path:
    """Locate the source CSV (prefers a filename containing 'OGLE')."""
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV dataset found in {folder}")
    ogle = [c for c in csvs if "ogle" in c.name.lower()]
    if ogle:
        return ogle[0]
    if len(csvs) == 1:
        return csvs[0]
    raise FileNotFoundError(
        f"Multiple CSVs in {folder} and none named *OGLE*; leave only one:\n  "
        + "\n  ".join(c.name for c in csvs))


def point_columns(header: pd.DataFrame) -> list[str]:
    """The t_000..t_399 light-curve columns, in tau order.

    Guards against startswith('t_') alone, which also matches t_E_days -- only
    columns whose suffix is all digits are light-curve samples.
    """
    cols = [c for c in header.columns if c.startswith("t_") and str(c)[2:].isdigit()]
    if len(cols) != N_POINTS:
        raise ValueError(f"Expected {N_POINTS} light-curve columns, found {len(cols)}")
    return sorted(cols, key=lambda c: int(str(c)[2:]))


def _nnls2_chi2(SW, SWF, SWFF, SWA, SWAA, SWAF):
    """Weighted 2-variable NNLS residual chi^2 for F ~ a*A + b, a>=0, b>=0.

    All inputs are length-C arrays of the weighted sums for one u0 candidate over
    a chunk of C curves (S** = sum_i w_i * ...). Returns (chi2, a, b) arrays.

    The interior (unconstrained) optimum is the global least-squares solution; if
    it is feasible (a>=0, b>=0) it is optimal. Otherwise the constrained optimum
    lies on an edge, so we compare the b=0 edge (a = SWAF/SWAA) and the a=0 edge
    (b = SWF/SW, always feasible since flux > 0) and take the feasible one with the
    lower chi^2. This is exact NNLS for two variables, fully vectorised.
    """
    det = SWAA * SW - SWA ** 2
    safe = det > 1e-30                                  # enough distinct observed points
    det_s = np.where(safe, det, 1.0)
    a_int = (SWAF * SW - SWF * SWA) / det_s
    b_int = (SWAA * SWF - SWA * SWAF) / det_s
    interior_ok = safe & (a_int >= 0) & (b_int >= 0)

    # Edge b = 0: pure single-lens amplitude, no blend.
    a_b0 = np.where(SWAA > 1e-30, SWAF / np.where(SWAA > 1e-30, SWAA, 1.0), 0.0)
    a_b0 = np.maximum(a_b0, 0.0)
    chi2_b0 = SWFF - a_b0 * SWAF

    # Edge a = 0: flat baseline (no event); b is the weighted mean flux.
    b_a0 = np.where(SW > 1e-30, SWF / np.where(SW > 1e-30, SW, 1.0), 0.0)
    b_a0 = np.maximum(b_a0, 0.0)
    chi2_a0 = SWFF - b_a0 * SWF

    # Pick the better edge where the interior is infeasible.
    use_b0 = (a_b0 >= 0) & (chi2_b0 <= chi2_a0)
    a_edge = np.where(use_b0, a_b0, 0.0)
    b_edge = np.where(use_b0, 0.0, b_a0)
    chi2_edge = np.where(use_b0, chi2_b0, chi2_a0)

    a = np.where(interior_ok, a_int, a_edge)
    b = np.where(interior_ok, b_int, b_edge)
    chi2_int = SWFF - a_int * SWAF - b_int * SWF
    chi2 = np.where(interior_ok, chi2_int, chi2_edge)
    chi2 = np.maximum(chi2, 0.0)                         # kill tiny negative round-off
    return chi2, a, b


def single_lens_residual(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-fit single-lens (Paczynski) fit of each curve; return its residual.

    X : (C, 400) I-band magnitudes for one chunk of curves, NaN at cadence gaps.

    Returns
        r         (C, 400) STANDARDISED residual (F_obs - F_model) / sigma_F of the
                  best single-lens fit, zero at cadence gaps. This is the extra CNN
                  input channel: a binary leaves a caustic residual the single-lens
                  family cannot absorb, and unlike the fold residual it is defined at
                  EVERY observed point, not only co-observed tau/-tau pairs.
        best_chi2 (C,) weighted residual sum of squares of the best fit.
        best_k    (C,) index into U0_GRID of the best-fit impact parameter.

    In flux the blended single-lens model F = a*A(u0,tau) + b is linear in (a, b),
    so the fit is a 1-D search over u0 with a closed-form non-negative 2-parameter
    weighted least-squares inside (weights from the OGLE sigma(I) noise model). This
    is the single shared implementation used by the feature extractor AND by the
    CNN's training/inference preprocessing, so the channel is identical everywhere.
    """
    obs = ~np.isnan(X)                                  # (C, 400)

    # Work in flux with a zero point at I_ref so numbers are O(1). The fit is
    # invariant to this choice (flux and its sigma scale together).
    Xz = np.where(obs, X, _I_REF)                       # fill gaps with a harmless value
    F = 10.0 ** (-0.4 * (Xz - _I_REF))
    sig_I = sigma_I(Xz)
    sig_F = F * (np.log(10.0) / 2.5) * sig_I             # error propagation mag -> flux
    W = np.where(obs, 1.0 / sig_F ** 2, 0.0)            # gaps carry zero weight
    F = np.where(obs, F, 0.0)

    # Weighted sums that do not depend on u0 (computed once per chunk).
    SW = W.sum(axis=1)
    SWF = (W * F).sum(axis=1)
    SWFF = (W * F * F).sum(axis=1)

    C = X.shape[0]
    best_chi2 = np.full(C, np.inf)
    best_k = np.zeros(C, dtype=np.int32)
    A_all = (U0_GRID[:, None] ** 2 + TAU[None, :] ** 2)              # (K, 400) = u^2
    A_all = (A_all + 2.0) / (np.sqrt(A_all) * np.sqrt(A_all + 4.0))  # Paczynski A(u)

    for k in range(len(U0_GRID)):
        A = A_all[k]                                    # (400,)
        WA = W * A                                      # (C, 400)
        SWA = WA.sum(axis=1)
        SWAA = (WA * A).sum(axis=1)
        SWAF = (WA * F).sum(axis=1)
        chi2, _, _ = _nnls2_chi2(SW, SWF, SWFF, SWA, SWAA, SWAF)
        better = chi2 < best_chi2
        best_chi2[better] = chi2[better]
        best_k[better] = k

    # Recompute a, b and the residual vector for the best u0 of each curve.
    A_best = A_all[best_k]                               # (C, 400)
    WA = W * A_best
    _, a, b = _nnls2_chi2(SW, SWF, SWFF,
                          WA.sum(axis=1), (WA * A_best).sum(axis=1),
                          (WA * F).sum(axis=1))
    model = a[:, None] * A_best + b[:, None]
    r = np.where(obs, (F - model) * np.sqrt(W), 0.0)    # standardised residuals
    return r.astype(np.float64), best_chi2, best_k


def fit_single_lens(X: np.ndarray) -> dict:
    """Fit a single-lens model to each curve and return per-curve scalar features."""
    obs = ~np.isnan(X)                                  # (C, 400)
    n_obs = obs.sum(axis=1).astype(np.float64)
    r, best_chi2, best_k = single_lens_residual(X)

    dof = np.maximum(n_obs - 2.0, 1.0)
    abs_r = np.abs(r)
    imax = np.argmax(abs_r, axis=1)
    over2 = abs_r > 2.0
    # tau-range spanned by >2 sigma residuals (0 if fewer than one such point).
    tau_grid = TAU[None, :]
    tau_hi = np.where(over2, tau_grid, -np.inf).max(axis=1)
    tau_lo = np.where(over2, tau_grid, np.inf).min(axis=1)
    resid_span = np.where(over2.any(axis=1), tau_hi - tau_lo, 0.0)

    return {
        "u0_fit": U0_GRID[best_k],
        "n_obs": n_obs,
        "chi2": best_chi2,
        "chi2_dof": best_chi2 / dof,
        "max_abs_r": abs_r.max(axis=1),
        "tau_absmax": np.abs(TAU[imax]),
        "n_r_gt3": (abs_r > 3.0).sum(axis=1).astype(np.float64),
        "pos_excess": (np.maximum(r, 0.0) ** 2).sum(axis=1),
        "resid_span": resid_span,
    }


def trivial_features(X: np.ndarray) -> dict:
    """Cheap baseline features: coverage, magnitude moments, and the fold signal.

    fold_max is exactly the statistic the CNN's fold channel is built on, included
    here so the value added by the chi^2 fit can be read off directly.
    """
    obs = ~np.isnan(X)
    n = obs.sum(axis=1).astype(np.float64)
    n_safe = np.maximum(n, 1.0)
    Xf = np.where(obs, X, 0.0)

    mean = Xf.sum(axis=1) / n_safe
    d = np.where(obs, X - mean[:, None], 0.0)
    m2 = (d ** 2).sum(axis=1) / n_safe
    std = np.sqrt(m2)
    std_safe = np.where(std < 1e-8, 1.0, std)
    skew = (d ** 3).sum(axis=1) / n_safe / std_safe ** 3
    kurt = (d ** 4).sum(axis=1) / n_safe / np.maximum(m2, 1e-16) ** 2 - 3.0

    # Fold residual over co-observed mirror pairs (the CNN's binary signature).
    rev = X[:, ::-1]
    both = obs & obs[:, ::-1]
    fold = np.where(both, np.abs(X - rev), 0.0)
    fold_max = fold.max(axis=1)
    fold_n = both.sum(axis=1).astype(np.float64) / 2.0   # each pair counted twice

    return {
        "coverage": n / N_POINTS,
        "mag_std": std,
        "mag_skew": skew,
        "mag_kurt": kurt,
        "fold_max": fold_max,
        "fold_n": fold_n,
    }


FEATURE_ORDER = [
    "coverage", "mag_std", "mag_skew", "mag_kurt", "fold_max", "fold_n",
    "u0_fit", "n_obs", "chi2", "chi2_dof", "max_abs_r", "tau_absmax",
    "n_r_gt3", "pos_excess", "resid_span",
]


def extract(csv_path: Path, out_path: Path, max_events: int | None,
            chunk: int, report: bool) -> None:
    label_col = "event_lenses"
    header = pd.read_csv(csv_path, nrows=0)
    cols = point_columns(header)
    dtypes = {c: np.float32 for c in cols}
    dtypes[label_col] = np.int16

    print(f"Reading {csv_path.name} ...")
    df = pd.read_csv(csv_path, usecols=[label_col] + cols, dtype=dtypes)
    y = df[label_col].to_numpy(dtype=np.int16)
    X_all = df[cols].to_numpy(dtype=np.float32)
    del df

    # Optional STRATIFIED subsample for a quick scout. The dataset is stored
    # unshuffled (all singles, then all binaries), so a head-read would give a
    # single class -- we sample each class in proportion instead, keeping the
    # event_index pointing back at the original row.
    event_index = np.arange(len(y))
    if max_events is not None and max_events < len(y):
        rng = np.random.RandomState(42)
        frac = max_events / len(y)
        keep = np.concatenate([
            rng.choice(np.where(y == 1)[0], int(round((y == 1).sum() * frac)), replace=False),
            rng.choice(np.where(y == 2)[0], int(round((y == 2).sum() * frac)), replace=False),
        ])
        keep.sort()
        X_all, y, event_index = X_all[keep], y[keep], event_index[keep]
        print(f"  stratified subsample -> {len(y):,} events")
    N = len(y)
    print(f"  events: {N:,}  |  single: {(y == 1).sum():,}  binary: {(y == 2).sum():,}")
    print(f"  fitting single-lens model over {len(U0_GRID)} u0 grid points, "
          f"chunk={chunk:,}")

    parts: list[dict] = []
    t0 = time.time()
    for start in range(0, N, chunk):
        Xc = X_all[start:start + chunk].astype(np.float64)
        feats = trivial_features(Xc)
        feats.update(fit_single_lens(Xc))
        parts.append(feats)
        done = min(start + chunk, N)
        print(f"    {done:>7,}/{N:,}  ({time.time() - t0:5.1f}s)", end="\r")
    print()

    out = {"event_index": event_index, "event_lenses": y}
    for name in FEATURE_ORDER:
        out[name] = np.concatenate([p[name] for p in parts])
    table = pd.DataFrame(out)
    table.to_csv(out_path, index=False)
    print(f"Wrote {len(table):,} rows x {len(FEATURE_ORDER)} features -> {out_path.name}")

    if report:
        univariate_auc_report(table)


def univariate_auc_report(table: pd.DataFrame) -> None:
    """Print each feature's standalone ROC-AUC against the binary label.

    A quick, model-free answer to the core question: does the single-lens fit
    residual separate binaries from singles, and by how much more than the fold
    feature the CNN already uses? AUC is direction-agnostic here (reported as
    max(auc, 1-auc)) so a feature that is *lower* for binaries still scores high.
    """
    from sklearn.metrics import roc_auc_score

    y = (table["event_lenses"].to_numpy() == 2).astype(int)
    print("\nUnivariate ROC-AUC vs binary label (direction-agnostic)")
    print("  feature           AUC")
    rows = []
    for name in FEATURE_ORDER:
        v = table[name].to_numpy(dtype=np.float64)
        good = np.isfinite(v)
        if good.sum() < 10 or len(np.unique(v[good])) < 2:
            continue
        auc = roc_auc_score(y[good], v[good])
        rows.append((name, max(auc, 1.0 - auc)))
    for name, auc in sorted(rows, key=lambda r: -r[1]):
        star = "  <-- fold feature (CNN's signal)" if name == "fold_max" else ""
        star = "  <-- chi^2 headline" if name == "chi2_dof" else star
        print(f"  {name:14s}   {auc:.4f}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None,
                    help="source dataset CSV (default: the OGLE CSV in ../Real/)")
    ap.add_argument("--out", type=Path, default=HERE / "features_real.csv",
                    help="output feature-table CSV")
    ap.add_argument("--max-events", type=int, default=None,
                    help="only read the first N events (quick subset scout)")
    ap.add_argument("--chunk", type=int, default=5000,
                    help="curves processed per vectorised batch (memory knob)")
    ap.add_argument("--report", action="store_true",
                    help="print per-feature univariate ROC-AUC after extraction")
    args = ap.parse_args()

    csv_path = args.csv or find_dataset(REAL_DIR)
    extract(csv_path, args.out, args.max_events, args.chunk, args.report)


if __name__ == "__main__":
    main()
