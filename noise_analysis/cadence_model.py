"""
OGLE-IV observing cadence distribution from 3000 EWS event light curves.

Loads ogle_phot_raw.npz (produced by fetch_phot.py) and characterises the
time-sampling pattern of OGLE-IV microlensing observations:

  - Within-season cadence : Δt < 100 days  (actual observing gaps)
  - Seasonal gaps         : Δt >= 100 days  (Galactic Bulge behind the Sun)
  - Visibility fraction   : fraction of the year the bulge is observable

Outputs
-------
  cadence_model.npz  -- Δt distribution + key statistics
  cadence_model.png  -- two-panel annotated figure
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
INPUT_NPZ  = HERE / "ogle_phot_raw.npz"
OUTPUT_NPZ = HERE / "cadence_model.npz"

# ── Thresholds ────────────────────────────────────────────────────────────────
SEASONAL_GAP = 100.0    # Δt >= 100 days → bulge out of season
INTRANIGHT   = 0.5      # Δt < 0.5 days  → multiple obs within one night


def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"Loading {INPUT_NPZ.name}...")
    d        = np.load(INPUT_NPZ)
    dt_all   = d["dt"]
    n_events = int(d["n_events"][0])
    print(f"  {len(dt_all):,} time gaps from {n_events} events")

    # ── 2. Split into regimes ─────────────────────────────────────────────────
    dt_season  = dt_all[dt_all >= SEASONAL_GAP]
    dt_inseason = dt_all[(dt_all > 0) & (dt_all < SEASONAL_GAP)]

    dt_intranight  = dt_inseason[dt_inseason <  INTRANIGHT]
    dt_internight  = dt_inseason[dt_inseason >= INTRANIGHT]

    frac_season   = len(dt_season)   / len(dt_all) * 100
    frac_innight  = len(dt_intranight) / len(dt_inseason) * 100
    frac_internight = len(dt_internight) / len(dt_inseason) * 100

    # Seasonal visibility: fraction of year the bulge is observable
    all_gaps_season = dt_season
    if len(all_gaps_season):
        mean_gap_days = np.mean(all_gaps_season)
        visibility_frac = 1.0 - mean_gap_days / 365.25
    else:
        mean_gap_days = 0
        visibility_frac = 1.0

    # ── 3. Key statistics ─────────────────────────────────────────────────────
    med_inseason  = np.median(dt_inseason)
    p10_inseason  = np.percentile(dt_inseason, 10)
    p90_inseason  = np.percentile(dt_inseason, 90)
    med_intranight = np.median(dt_intranight) if len(dt_intranight) else np.nan
    med_internight = np.median(dt_internight) if len(dt_internight) else np.nan

    print()
    print("--- Cadence summary ---")
    print(f"  Total dt values           : {len(dt_all):>10,}")
    print(f"  Within-season (<100 d)    : {len(dt_inseason):>10,}  ({100-frac_season:.1f}%)")
    print(f"    Intra-night (<0.5 d)    : {len(dt_intranight):>10,}  ({frac_innight:.1f}% of within-season)")
    print(f"    Night-to-night (>=0.5 d): {len(dt_internight):>10,}  ({frac_internight:.1f}%)")
    print(f"  Seasonal gaps (>=100 d)   : {len(dt_season):>10,}  ({frac_season:.1f}%)")
    print()
    print(f"  Median dt (within-season) : {med_inseason:.4f} d  ({med_inseason*24*60:.1f} min)")
    print(f"  10th pct dt               : {p10_inseason:.4f} d  ({p10_inseason*24*60:.1f} min)")
    print(f"  90th pct dt               : {p90_inseason:.4f} d  ({p90_inseason*24*60:.1f} h)")
    if not np.isnan(med_intranight):
        print(f"  Median intra-night dt     : {med_intranight:.4f} d  ({med_intranight*24*60:.1f} min)")
    print(f"  Median night-to-night dt  : {med_internight:.3f} d")
    print(f"  Mean seasonal gap         : {mean_gap_days:.0f} days")
    print(f"  Bulge visibility fraction : {visibility_frac:.2f}  ({visibility_frac*365.25:.0f} days/year)")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    np.savez_compressed(
        OUTPUT_NPZ,
        dt_inseason    = dt_inseason,
        dt_season      = dt_season,
        stats = np.array([
            med_inseason, p10_inseason, p90_inseason,
            med_intranight, med_internight,
            mean_gap_days, visibility_frac,
        ]),
    )
    print(f"\nSaved cadence model -> {OUTPUT_NPZ}")

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"OGLE-IV Observing Cadence  ({n_events} events, "
        f"{len(dt_all):,} time gaps)",
        fontsize=14, fontweight="bold",
    )

    # ── Panel A: full Δt distribution (all regimes) ───────────────────────────
    ax = axes[0]

    bins_full = np.logspace(np.log10(0.001), np.log10(500), 120)
    ax.hist(dt_all[dt_all > 0], bins=bins_full, density=True,
            color="#2563eb", edgecolor="none", alpha=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Annotate regimes with shaded bands
    ax.axvspan(0.001,      INTRANIGHT,   alpha=0.08, color="green",  zorder=0)
    ax.axvspan(INTRANIGHT, SEASONAL_GAP, alpha=0.08, color="orange", zorder=0)
    ax.axvspan(SEASONAL_GAP, 500,        alpha=0.08, color="red",    zorder=0)

    # Vertical boundaries
    ax.axvline(INTRANIGHT,   color="green",  ls="--", lw=1.2, alpha=0.8)
    ax.axvline(SEASONAL_GAP, color="red",    ls="--", lw=1.2, alpha=0.8)

    # Labels inside the bands
    ymax_ax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10
    ax.text(0.015,       0.92, "Intra-night\n(< 0.5 d)",
            transform=ax.transAxes, color="green",  fontsize=9, va="top")
    ax.text(0.32,        0.92, "Night-to-night\n(0.5 – 100 d)",
            transform=ax.transAxes, color="darkorange", fontsize=9, va="top")
    ax.text(0.78,        0.92, "Seasonal\ngap",
            transform=ax.transAxes, color="red",    fontsize=9, va="top")

    # Median line
    ax.axvline(med_inseason, color="black", ls="-", lw=1.5,
               label=f"Median (within-season): {med_inseason:.3f} d  "
                     f"({med_inseason*24*60:.0f} min)")

    ax.set_xlabel(r"$\Delta t$ between consecutive observations  [days]", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Full Cadence Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    # Custom x-tick labels in human-readable units
    ticks = [0.01, 0.1, 0.5, 1, 7, 30, 100, 365]
    labels = ["15 min", "2.4 h", "0.5 d", "1 d", "1 wk", "1 mo", "100 d", "1 yr"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(0.001, 500)

    # ── Panel B: within-season cadence zoomed in ──────────────────────────────
    ax2 = axes[1]

    bins_ws = np.logspace(np.log10(0.005), np.log10(SEASONAL_GAP), 100)
    counts, edges = np.histogram(dt_inseason, bins=bins_ws, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean for log bins

    ax2.fill_between(centers, counts, alpha=0.4, color="#2563eb", step="mid")
    ax2.step(centers, counts, where="mid", color="#2563eb", lw=1.2)
    ax2.set_xscale("log")

    # Intra-night / inter-night boundary
    ax2.axvline(INTRANIGHT, color="green", ls="--", lw=1.5,
                label=f"Intra-night boundary (0.5 d)")
    ax2.axvspan(0.005,      INTRANIGHT,   alpha=0.08, color="green")
    ax2.axvspan(INTRANIGHT, SEASONAL_GAP, alpha=0.08, color="orange")

    # Median and percentile markers
    ax2.axvline(med_inseason, color="red", ls="-", lw=1.8,
                label=f"Median: {med_inseason:.3f} d  ({med_inseason*24*60:.0f} min)")
    ax2.axvline(p10_inseason, color="gray", ls=":", lw=1.2,
                label=f"10th pct: {p10_inseason*24*60:.0f} min")
    ax2.axvline(p90_inseason, color="gray", ls=":", lw=1.2,
                label=f"90th pct: {p90_inseason:.2f} d")

    if not np.isnan(med_intranight):
        ax2.axvline(med_intranight, color="green", ls="-", lw=1.2, alpha=0.7,
                    label=f"Median intra-night: {med_intranight*24*60:.0f} min")
    ax2.axvline(med_internight, color="darkorange", ls="-", lw=1.2, alpha=0.7,
                label=f"Median night-to-night: {med_internight:.2f} d")

    ticks2 = [0.01, 0.05, 0.5, 1, 3, 10, 30, 100]
    labels2 = ["15 min", "1.2 h", "0.5 d", "1 d", "3 d", "10 d", "30 d", "100 d"]
    ax2.set_xticks(ticks2)
    ax2.set_xticklabels(labels2, fontsize=8)
    ax2.set_xlim(0.005, SEASONAL_GAP)

    ax2.set_xlabel(r"$\Delta t$ between consecutive observations  [days]", fontsize=12)
    ax2.set_ylabel("Probability Density", fontsize=12)
    ax2.set_title("Within-Season Cadence  (Δt < 100 days)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    out_fig = HERE / "cadence_model.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure     -> {out_fig}")
    plt.close(fig)


main()