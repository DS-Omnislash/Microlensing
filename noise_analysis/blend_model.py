"""
OGLE-IV blending model from the Mroz et al. (2019) event catalogue.

Real OGLE-IV photometry is *blended*: the flux the detector records is the sum
of the (magnified) source flux and a constant blend flux -- the lens plus
unresolved neighbouring bulge stars sharing the aperture:

    F_obs(t) = A(t) * F_source + F_blend

The catalogue reports, per event, the source-flux fraction

    f_s = F_source / F_baseline = F_source / (F_source + F_blend)   (column fs_med)

so that, in magnitudes, the observed light curve is

    I(t) = I_base - 2.5*log10( f_s * A(t) + (1 - f_s) )

where I_base is the OBSERVED baseline magnitude (the value the detector records
at A = 1). Only the source is magnified; the blend is constant, which both
places the baseline at the real observed level AND correctly dilutes the peaks
(a fully magnified source is watered down by the steady blend).

Setting f_s = 1 (no blend) recovers the idealised formula I(t) = I_s - 2.5*log10 A.

This script extracts the (I_s, f_s) pairs so the webapp can bootstrap-resample
them at generation time (no network dependency at runtime), exactly as
cadence_model.py stores the empirical cadence gaps.

They are stored as PAIRS, not as two independent distributions, because the two
are correlated: brighter sources are measurably less blended (median f_s falls
from 0.88 at I_s < 17 to 0.66 at I_s ~ 20). Drawing them independently would
destroy that relationship. Resampling whole catalogue rows keeps each synthetic
event self-consistent, and the resulting baseline

    I_base = I_s + 2.5*log10(f_s)

then reproduces the real observed baseline distribution measured independently
from the photometry by baseline_model.py (median 18.86 vs 18.94 mag).

Outputs
-------
  blend_model.npz   -- paired (I_s, f_s) arrays + summary stats
  blend_model.png   -- empirical f_s distribution
"""

import io
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")  # suppress astrouw SSL-verify warning

HERE = Path(__file__).parent
OUTPUT_NPZ = HERE / "blend_model.npz"
OUTPUT_PNG = HERE / "blend_model.png"

URL = "https://www.astrouw.edu.pl/ogle/ogle4/microlensing_maps/table3.dat"

# Fixed-width column spec from Mroz et al. (2019) table3.dat (matches
# distributions/I_baseline_mag.py). fs_med is the source-flux fraction.
COL_SPECS = [
    (0, 16),   (17, 26),  (27, 33),  (34, 36),  (37, 39),  (40, 45),
    (46, 49),  (50, 52),  (53, 57),  (58, 67),  (68, 77),  (78, 87),
    (88, 97),  (98, 109), (110, 117),(118, 123),(124, 130),(131, 136),
    (137, 148),(149, 156),(157, 163),(164, 171),(172, 179),(180, 186),
    (187, 192),(193, 199),(200, 205),(206, 212),(213, 219),(220, 225),
    (226, 231),(232, 238),(239, 244),(245, 251),(252, 270),
]
COLS = [
    "name", "field", "star_id", "ra_h", "ra_m", "ra_s", "dec_d", "dec_m", "dec_s",
    "ra", "dec", "glon", "glat", "t0", "tE", "u0", "Is_best", "fs_best",
    "t0_med", "t0_err1", "t0_err2", "tE_med", "tE_err1", "tE_err2",
    "u0_med", "u0_err1", "u0_err2", "Is_med", "Is_err1", "Is_err2",
    "fs_med", "fs_err1", "fs_err2", "weight", "ews_id",
]

# Physical plausibility filter for f_s = F_source / F_baseline.
# f_s in (0, 1]  -> normal blending (0 = fully blended, 1 = no blend).
# f_s slightly > 1 -> mild over-subtraction of the blend in the OGLE fit; kept
# up to a cap because it occurs in the real catalogue (~8% of events).
FS_MIN, FS_MAX = 1e-3, 2.0

# Sanity window on the fitted source magnitude, to drop catalogue rows where the
# fit failed outright. Deliberately wide -- it removes junk, not real events.
I_S_MIN, I_S_MAX = 10.0, 24.0


def main():
    import pandas as pd

    print(f"Fetching {URL} ...")
    resp = requests.get(URL, verify=False, timeout=60)
    df = pd.read_fwf(io.StringIO(resp.text), colspecs=COL_SPECS, names=COLS, comment="#")

    fs = pd.to_numeric(df["fs_med"], errors="coerce").to_numpy(dtype=float)
    Is = pd.to_numeric(df["Is_med"], errors="coerce").to_numpy(dtype=float)

    # Keep only rows where BOTH are usable, so every stored pair is a real event.
    good = (
        np.isfinite(fs) & (fs >= FS_MIN) & (fs <= FS_MAX)
        & np.isfinite(Is) & (Is > I_S_MIN) & (Is < I_S_MAX)
    )
    fs, Is = fs[good], Is[good]
    print(f"  {good.sum():,} events with a usable (I_s, f_s) pair")
    print(
        f"  f_s: median={np.median(fs):.3f}  "
        f"P10={np.percentile(fs, 10):.3f}  P90={np.percentile(fs, 90):.3f}"
    )
    print(f"  I_s: median={np.median(Is):.2f} mag")

    # The pairing is the point: check the correlation survives, and that the
    # implied baseline matches the one measured independently from photometry.
    I_base = Is + 2.5 * np.log10(fs)
    print()
    print("--- Source-blend correlation (why pairs, not independent draws) ---")
    for lo, hi in [(12, 17), (17, 18), (18, 19), (19, 20), (20, 21)]:
        m = (Is >= lo) & (Is < hi)
        if m.sum() > 20:
            print(f"  I_s {lo}-{hi}:  n={m.sum():5d}   median f_s = {np.median(fs[m]):.3f}")
    print(f"\n  implied I_base = I_s + 2.5*log10(f_s):  median = {np.median(I_base):.2f} mag")
    print("  (compare with baseline_model.py, measured from photometry)")

    np.savez_compressed(
        OUTPUT_NPZ,
        fs=fs.astype(np.float64),
        Is=Is.astype(np.float64),          # paired element-wise with fs
        stats=np.array(
            [float(np.median(fs)), float(np.percentile(fs, 16)),
             float(np.percentile(fs, 84)), float(np.median(I_base))]
        ),
        n_events=np.array([len(fs)]),
    )
    print(f"\nSaved blend model -> {OUTPUT_NPZ}  ({len(fs):,} (I_s, f_s) pairs)")

    plot(fs)


def plot(fs: np.ndarray) -> None:
    """Empirical distribution of the source flux fraction f_s."""
    med = float(np.median(fs))
    p10, p90 = np.percentile(fs, [10, 90])
    frac_over_one = float((fs > 1.0).mean()) * 100.0

    fig, ax = plt.subplots(figsize=(9, 6))

    bins = np.linspace(0.0, FS_MAX, 80)
    ax.hist(fs, bins=bins, density=True, color="#2563eb", alpha=0.7,
            edgecolor="none", label="OGLE-IV events")

    kde = gaussian_kde(fs, bw_method=0.15)
    x = np.linspace(0.0, FS_MAX, 400)
    ax.plot(x, kde(x), "k-", lw=2, label="KDE")

    # f_s > 1 implies a negative blend flux, so it cannot be physical. It appears
    # because f_s is a FITTED parameter: for nearly-unblended events the noise can
    # push the best fit just past the boundary. Kept as it appears in the catalogue.
    ax.axvspan(1.0, FS_MAX, color="red", alpha=0.08, zorder=0)
    ax.axvline(1.0, color="red", ls="--", lw=1.5,
               label=f"$f_s>1$: fit scatter past\nphysical limit ({frac_over_one:.1f}%)")

    ax.axvline(med, color="orange", ls="-", lw=2, label=f"Median: {med:.3f}")
    ax.axvline(p10, color="gray", ls=":", lw=1.5, label=f"10th pct: {p10:.3f}")
    ax.axvline(p90, color="gray", ls=":", lw=1.5, label=f"90th pct: {p90:.3f}")

    ax.set_xlabel(r"Source flux fraction  $f_s = F_{source}\,/\,F_{baseline}$", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        f"OGLE-IV Blending Model  ({len(fs):,} events)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0.0, FS_MAX)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved figure     -> {OUTPUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
