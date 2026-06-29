"""
OGLE-IV photometric noise model from 3000 EWS event light curves.

Loads ogle_phot_raw.npz (produced by fetch_phot.py) and builds an empirical
sigma(I) noise curve — the photometric uncertainty as a function of I-band
magnitude. All observations are used (baseline + lensed) because sigma(I) is
a physical property of the detector/pipeline that depends only on received
flux, not on whether the source is being gravitationally amplified.

Fit strategy: residuals are minimised in log10(sigma) space so that every
magnitude bin contributes equally regardless of how many observations it
contains. Only bins with >= FIT_MIN_OBS observations enter the fit, which
excludes the sparse I < 14 regime dominated by extreme lensing events.

Outputs
-------
  noise_model.npz   -- binned noise curve + fitted parametric model
  noise_model.png   -- two-panel figure
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
INPUT_NPZ  = HERE / "ogle_phot_raw.npz"
OUTPUT_NPZ = HERE / "noise_model.npz"

# ── Quality filter ────────────────────────────────────────────────────────────
I_MIN,   I_MAX   = 12.0, 22.5
SIG_MIN, SIG_MAX =  0.001, 1.0

# ── Binning ───────────────────────────────────────────────────────────────────
BIN_WIDTH    = 0.2    # mag
DISPLAY_MIN  = 10     # min obs per bin to show the median / percentile lines
FIT_MIN_OBS  = 200    # min obs per bin to enter the parametric fit
                      # (excludes sparse I < ~14 region biased by lensed events)

I_REF = 18.0          # reference magnitude for the parametric model


# ── Noise model: systematic floor + photon noise ──────────────────────────────
def noise_model(I, sigma_floor, sigma_phot0):
    """sigma(I) = sqrt(sigma_floor^2 + sigma_phot0^2 * 10^(0.4*(I - I_ref)))"""
    return np.sqrt(sigma_floor**2 + sigma_phot0**2 * 10.0 ** (0.4 * (I - I_REF)))


def log_noise_model(I, sigma_floor, sigma_phot0):
    """Same model in log10 space — used for the fit so all bins weight equally."""
    return np.log10(noise_model(I, sigma_floor, sigma_phot0))


def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"Loading {INPUT_NPZ.name}...")
    d        = np.load(INPUT_NPZ)
    imag_all = d["imag"]
    sig_all  = d["sigma"]
    n_events = int(d["n_events"][0])
    print(f"  {len(imag_all):,} observations from {n_events} events")

    # ── 2. Quality filter ─────────────────────────────────────────────────────
    mask = (
        (imag_all >= I_MIN) & (imag_all <= I_MAX) &
        (sig_all  >= SIG_MIN) & (sig_all <= SIG_MAX)
    )
    imag  = imag_all[mask]
    sigma = sig_all[mask]
    print(f"  {mask.sum():,} pass quality filter  "
          f"(I [{I_MIN}, {I_MAX}]  sigma [{SIG_MIN}, {SIG_MAX}])")

    # ── 3. Bin by magnitude ───────────────────────────────────────────────────
    bin_edges   = np.arange(I_MIN, I_MAX + BIN_WIDTH, BIN_WIDTH)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins      = len(bin_centers)

    med_sigma   = np.full(n_bins, np.nan)
    pct16_sigma = np.full(n_bins, np.nan)
    pct84_sigma = np.full(n_bins, np.nan)
    n_per_bin   = np.zeros(n_bins, dtype=int)

    for k, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        s = sigma[(imag >= lo) & (imag < hi)]
        n_per_bin[k] = len(s)
        if len(s) < DISPLAY_MIN:
            continue
        med_sigma[k]   = np.median(s)
        pct16_sigma[k] = np.percentile(s, 16)
        pct84_sigma[k] = np.percentile(s, 84)

    display_mask = np.isfinite(med_sigma)
    fit_mask     = display_mask & (n_per_bin >= FIT_MIN_OBS)
    print(f"\nBins for display : {display_mask.sum()}  "
          f"(>= {DISPLAY_MIN} obs)")
    print(f"Bins for fit     : {fit_mask.sum()}  "
          f"(>= {FIT_MIN_OBS} obs, I >= ~{bin_centers[fit_mask].min():.1f})")

    # ── 4. Parametric fit in log-space ────────────────────────────────────────
    fit_x    = bin_centers[fit_mask]
    fit_logy = np.log10(med_sigma[fit_mask])

    popt, pcov = curve_fit(
        log_noise_model, fit_x, fit_logy,
        p0=[0.003, 0.025],
        bounds=([1e-4, 1e-4], [0.05, 0.5]),
        maxfev=10_000,
    )
    sigma_floor_fit, sigma_phot0_fit = popt
    perr = np.sqrt(np.diag(pcov))

    I_curve     = np.linspace(I_MIN, I_MAX, 300)
    sigma_curve = noise_model(I_curve, *popt)

    print("\n--- sigma(I) = sqrt(a^2 + b^2 * 10^(0.4*(I-18)))  [log-space fit] ---")
    print(f"  sigma_floor (a)  = {sigma_floor_fit:.5f} +/- {perr[0]:.5f} mag")
    print(f"  sigma_phot0 (b)  = {sigma_phot0_fit:.5f} +/- {perr[1]:.5f} mag  "
          f"[photon noise at I={I_REF}]")
    print()
    for I_check in [14, 16, 17, 18, 19, 20, 21, 22]:
        print(f"  sigma(I={I_check:2d})  = {noise_model(I_check, *popt):.4f} mag")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    np.savez_compressed(
        OUTPUT_NPZ,
        bin_centers    = bin_centers[display_mask],
        med_sigma      = med_sigma[display_mask],
        pct16_sigma    = pct16_sigma[display_mask],
        pct84_sigma    = pct84_sigma[display_mask],
        n_per_bin      = n_per_bin[display_mask],
        fit_params     = np.array([sigma_floor_fit, sigma_phot0_fit]),
        fit_params_err = np.array(perr),
        I_ref          = np.array([I_REF]),
        I_curve        = I_curve,
        sigma_curve    = sigma_curve,
    )
    print(f"\nSaved noise model -> {OUTPUT_NPZ}")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"OGLE-IV Photometric Noise Model  ({n_events} events, "
        f"{len(imag):,} observations)",
        fontsize=14, fontweight="bold",
    )

    # ── Panel A: sigma(I) curve ───────────────────────────────────────────────
    ax = axes[0]

    hb = ax.hexbin(
        imag, sigma,
        gridsize=120,
        xscale="linear", yscale="log",
        cmap="Blues", mincnt=5,
        linewidths=0.2,
    )
    fig.colorbar(hb, ax=ax, label="Observations per cell")

    # 16–84 percentile envelope (only well-populated bins)
    ax.fill_between(
        bin_centers[display_mask],
        pct16_sigma[display_mask],
        pct84_sigma[display_mask],
        alpha=0.35, color="orange", label="16th–84th percentile",
    )
    # Median per bin
    ax.plot(
        bin_centers[display_mask], med_sigma[display_mask],
        "o-", color="orange", ms=3, lw=1.5, label="Median per bin",
    )
    # Fitted curve — plot only over the fitted range to avoid extrapolation artefacts
    I_fit_lo = bin_centers[fit_mask].min()
    fit_range = I_curve >= I_fit_lo
    ax.plot(
        I_curve[fit_range], sigma_curve[fit_range],
        "r-", lw=2.0,
        label=(
            r"Fit: $\sqrt{a^2 + b^2 \cdot 10^{0.4(I-18)}}$"
            f"\na={sigma_floor_fit:.4f} mag,  b={sigma_phot0_fit:.4f} mag"
        ),
    )

    ax.set_xlabel("I-band magnitude  [mag]", fontsize=12)
    ax.set_ylabel(r"$\sigma_I$  [mag]", fontsize=12)
    ax.set_title("Noise vs. Magnitude", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(I_MIN, I_MAX)
    ax.set_ylim(SIG_MIN, SIG_MAX)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    # ── Panel B: marginal distribution of sigma_I (log-spaced bins) ───────────
    ax2 = axes[1]

    log_bins = np.logspace(np.log10(SIG_MIN), np.log10(SIG_MAX), 80)
    ax2.hist(
        sigma,
        bins=log_bins, density=True,
        color="#2563eb", edgecolor="none", alpha=0.7,
        label="All observations",
    )
    ax2.set_xscale("log")

    # KDE on log10(sigma) then transform back for a smooth overlay
    log_s   = np.log10(sigma)
    kde     = gaussian_kde(log_s, bw_method=0.08)
    x_log   = np.linspace(np.log10(SIG_MIN), np.log10(SIG_MAX), 400)
    x_lin   = 10.0 ** x_log
    # Jacobian: density in linear space = density_in_log / (x * ln(10))
    y_kde   = kde(x_log) / (x_lin * np.log(10))
    ax2.plot(x_lin, y_kde, "k-", lw=2, label="KDE")

    med_s = np.median(sigma)
    p16_s = np.percentile(sigma, 16)
    p84_s = np.percentile(sigma, 84)
    ax2.axvline(med_s, color="red",    ls="--", lw=1.5,
                label=f"Median: {med_s:.4f} mag")
    ax2.axvline(p16_s, color="orange", ls=":",  lw=1.5,
                label=f"16th pct: {p16_s:.4f} mag")
    ax2.axvline(p84_s, color="orange", ls=":",  lw=1.5,
                label=f"84th pct: {p84_s:.4f} mag")

    ax2.set_xlabel(r"$\sigma_I$  [mag]", fontsize=12)
    ax2.set_ylabel("Probability Density", fontsize=12)
    ax2.set_title(r"Marginal Distribution of $\sigma_I$", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_fig = HERE / "noise_model.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure     -> {out_fig}")
    plt.close(fig)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print()
    print("--- Marginal sigma_I distribution ---")
    print(f"  Median sigma_I   : {med_s:.4f} mag")
    print(f"  16th pct         : {p16_s:.4f} mag")
    print(f"  84th pct         : {p84_s:.4f} mag")
    print(f"  90th pct         : {np.percentile(sigma, 90):.4f} mag")


main()