"""
OGLE-IV observed baseline magnitude distribution, measured per event.

The generator needs a baseline magnitude I_base for every synthetic event: the
brightness the detector records when no lensing is happening (source + blend).

Naive approach -- pool all 7.6M observations and histogram them -- is biased two
ways:

  1. It is OBSERVATION-weighted, so heavily-monitored events count many times
     over, while a sparsely-observed event barely counts. A dataset of EVENTS
     should draw from the distribution of EVENT baselines, one vote each.
  2. It includes the magnified points. An event's record spans ~3.7 years on
     average while the star is noticeably magnified for only ~4 months, so ~9%
     of the points sit above baseline and drag the distribution bright.

Both are fixed by taking, per event, the MEDIAN of its own magnitudes. A median
is unmoved by a ~9% contamination on one side, so it recovers the flat baseline
level, and each event then contributes exactly one value.

Outputs
-------
  baseline_model.npz  -- per-event baselines + binned histogram
  baseline_model.png  -- the distribution
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

HERE = Path(__file__).parent
INPUT_NPZ = HERE / "ogle_phot_raw.npz"
OUTPUT_NPZ = HERE / "baseline_model.npz"
OUTPUT_PNG = HERE / "baseline_model.png"

# Same quality window as noise_model.py, so the two models describe the same data.
I_MIN, I_MAX = 12.0, 22.5
BIN_WIDTH = 0.2
MIN_OBS_PER_EVENT = 20   # need enough points for the median to be meaningful


def main() -> None:
    print(f"Loading {INPUT_NPZ.name}...")
    d = np.load(INPUT_NPZ)
    imag = d["imag"]
    ei = d["event_index"]
    n_events = int(d["n_events"][0])
    print(f"  {len(imag):,} observations from {n_events} events")

    keep = (imag >= I_MIN) & (imag <= I_MAX)
    imag, ei = imag[keep], ei[keep]
    print(f"  {keep.sum():,} pass the quality window I in [{I_MIN}, {I_MAX}]")

    # --- One baseline per event: the median of that event's own magnitudes ----
    order = np.argsort(ei, kind="stable")
    imag_s, ei_s = imag[order], ei[order]
    bounds = np.searchsorted(ei_s, np.arange(n_events + 1))

    baselines = []
    for i in range(n_events):
        seg = imag_s[bounds[i]:bounds[i + 1]]
        if len(seg) >= MIN_OBS_PER_EVENT:
            baselines.append(np.median(seg))
    baselines = np.asarray(baselines, dtype=np.float64)

    print(f"\n  {len(baselines):,} events with >= {MIN_OBS_PER_EVENT} observations")

    # --- Compare against the biased, observation-weighted version -------------
    pooled_median = float(np.median(imag))
    med = float(np.median(baselines))
    print()
    print("--- Baseline magnitude ---")
    print(f"  observation-weighted (biased) : {pooled_median:.2f} mag")
    print(f"  per-event (this model)        : {med:.2f} mag")
    print(f"  correction                    : {med - pooled_median:+.2f} mag")
    print()
    print(f"  P16 / P84 : {np.percentile(baselines, 16):.2f} / "
          f"{np.percentile(baselines, 84):.2f} mag")

    # --- Bin it the same way ogle_noise.py samples it ------------------------
    bin_edges = np.arange(I_MIN, I_MAX + BIN_WIDTH, BIN_WIDTH)
    n_per_bin, _ = np.histogram(baselines, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    nz = n_per_bin > 0  # drop empty bins so sampling never draws a zero-probability bin
    np.savez_compressed(
        OUTPUT_NPZ,
        baselines=baselines,
        bin_centers=bin_centers[nz],
        n_per_bin=n_per_bin[nz],
        stats=np.array([
            med,
            float(np.percentile(baselines, 16)),
            float(np.percentile(baselines, 84)),
            pooled_median,
        ]),
        n_events=np.array([len(baselines)]),
    )
    print(f"\nSaved baseline model -> {OUTPUT_NPZ}")

    plot(baselines, med)


def plot(baselines: np.ndarray, med: float) -> None:
    p16, p84 = np.percentile(baselines, [16, 84])

    fig, ax = plt.subplots(figsize=(9, 6))

    bins = np.arange(I_MIN, I_MAX + BIN_WIDTH, BIN_WIDTH)
    ax.hist(baselines, bins=bins, density=True, color="#2563eb", alpha=0.7,
            edgecolor="none", label="OGLE-IV events")

    kde = gaussian_kde(baselines, bw_method=0.15)
    x = np.linspace(I_MIN, I_MAX, 400)
    ax.plot(x, kde(x), "k-", lw=2, label="KDE")

    ax.axvline(med, color="orange", ls="-", lw=2, label=f"Median: {med:.2f} mag")
    ax.axvline(p16, color="gray", ls=":", lw=1.5, label=f"16th pct: {p16:.2f} mag")
    ax.axvline(p84, color="gray", ls=":", lw=1.5, label=f"84th pct: {p84:.2f} mag")

    ax.set_xlabel(r"Observed baseline magnitude  $I_{base}$  [mag]", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        f"OGLE-IV Observed Baseline Magnitude  ({len(baselines):,} events)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(I_MIN, I_MAX)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved figure         -> {OUTPUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
