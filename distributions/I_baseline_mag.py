"""
Source Baseline I-band Magnitude (I_s) distribution from OGLE-IV survey data.

I_s is the apparent I-band magnitude of the source star at baseline (no lensing).
It reflects the stellar luminosity function along the line of sight to the
Galactic Bulge, as sampled by OGLE-IV's detection efficiency.

Source: Mroz et al. (2019), table3.dat — Warsaw Observatory
        https://www.astrouw.edu.pl/ogle/ogle4/microlensing_maps/table3.dat
        Column: Is_med (source magnitude, median posterior)

Color: #e67e22 (orange)
Reference: TdR_RocRC.pdf — baseline magnitude for photon noise model
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

URL = "https://www.astrouw.edu.pl/ogle/ogle4/microlensing_maps/table3.dat"

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


def plot_baseline_magnitude():
    try:
        print("Fetching OGLE-IV table3.dat...")
        response = requests.get(URL, verify=False)
        df = pd.read_fwf(io.StringIO(response.text), colspecs=COL_SPECS, names=COLS, comment="#")

        df["Is_med"] = pd.to_numeric(df["Is_med"], errors="coerce")
        data = df["Is_med"].dropna()

        # Restrict to physically plausible OGLE-IV range
        data = data[(data >= 12) & (data <= 22)]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(data, bins=50, color="#e67e22", edgecolor="black", alpha=0.75,
                density=True, label="OGLE-IV events")

        median_val = data.median()
        mode_counts, mode_edges = np.histogram(data, bins=100)
        mode_idx = mode_counts.argmax()
        mode_val = (mode_edges[mode_idx] + mode_edges[mode_idx + 1]) / 2

        ax.axvline(median_val, color="navy", linestyle="--", linewidth=2,
                   label=f"Median: {median_val:.2f} mag")
        ax.axvline(mode_val, color="red", linestyle=":", linewidth=2,
                   label=f"Mode: {mode_val:.2f} mag")

        ax.set_title("OGLE-IV Source Baseline I-band Magnitude Distribution",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel(r"Baseline Magnitude $I_s$ [mag]", fontsize=13)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

        print(f"\n--- Statistical Summary ---")
        print(f"Total events: {len(data)}")
        print(f"Median I_s: {median_val:.2f} mag")
        print(f"Mode I_s:   {mode_val:.2f} mag")
        print(f"Range: [{data.min():.2f}, {data.max():.2f}] mag")

    except Exception as e:
        print(f"Error: {e}")


plot_baseline_magnitude()