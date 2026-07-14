"""
Semi-Major Axis (a) distribution from NASA Exoplanet Archive - Planetary Systems table.

Units converted from AU to parsecs using the IAU 2015 B2 definition:
    1 pc = 648 000 / pi  AU  ->  1 AU = pi / 648 000 pc

Source: NASA Exoplanet Archive TAP service
        https://exoplanetarchive.ipac.caltech.edu - table: ps, column: pl_orbsmax
        Filtered to default parameters (default_flag = 1) AND to planets actually
        discovered by MICROLENSING (discoverymethod = 'Microlensing').

Why the discovery-method filter is essential
--------------------------------------------
Filtering only on default_flag=1 returns ~3 900 planets, of which ~2 360 are transit
detections and ~1 120 radial-velocity. Both techniques are most sensitive to close-in
planets, so that sample has a median a of 0.12 AU with 76% inside 1 AU. Microlensing
is blind to those planets: its sensitivity peaks at 1-10 AU, and the 274
microlensing-discovered planets have a median a of 2.37 AU with only 13% inside 1 AU
-- a factor of 20 apart.

Using the unfiltered archive therefore imports the *transit* survey's selection
function into a *microlensing* simulator, with two consequences:
  1. the companion separation in Einstein radii, d = a / r_E, came out at ~0.08
     instead of ~1 (r_E is itself ~2 AU), so the companion sat deep inside the
     Einstein ring where it barely perturbs the light curve; and
  2. the orbital period became so short that the companion completed ~2 full
     revolutions during a single event, whereas real microlensing companions are
     quasi-static over the ~1-month event (periods of years).

No analytic fit
---------------
The 274 real values are NOT fitted to an analytic family. In log10-space the
distribution is distinctly more peaked than a Gaussian (excess kurtosis +0.85), so a
log-normal misses the shape: measured against the real values it gives KS = 0.060,
whereas resampling the real values directly (with a small 0.05 dex jitter to make the
draw continuous) gives KS = 0.023.

``webapp/app/distributions.py`` therefore embeds the real values as
``A_MICROLENSING_AU`` and resamples them -- the same "bootstrap the real catalogue"
approach already used for the OGLE (I_s, f_s) blending pairs. Running this script
prints a refreshed copy of that array.

Color: #FFD300 (yellow)
Reference: NASA Exoplanet Archive (ps table, discoverymethod = 'Microlensing');
           supersedes the all-methods distribution of TdR_RocRC.pdf Image 8, p.26
"""

import io
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

# IAU 2015 Res B2: 1 pc = 648 000 / pi AU
AU_TO_PC = np.pi / 648_000.0

# Jitter used by the sampler, in dex. Deliberately below Silverman's rule (0.073),
# which is tuned for smooth density estimation and would broaden the sharp peak.
JITTER_DEX = 0.05

PS_DATA_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+pl_name,pl_orbsmax,discoverymethod+from+ps"
    "+where+default_flag=1+and+discoverymethod='Microlensing'&format=csv"
)


def plot_semi_major_axis():
    try:
        print("Fetching microlensing planets from NASA Exoplanet Archive...")
        response = requests.get(PS_DATA_URL, verify=False)
        df = pd.read_csv(io.StringIO(response.text), comment="#")

        if "pl_orbsmax" not in df.columns:
            print("Error: Column 'pl_orbsmax' not found.")
            return

        data_au = df["pl_orbsmax"].dropna().sort_values()
        data_pc = data_au * AU_TO_PC
        log10_au = np.log10(data_au)

        print(f"n = {len(data_au)} microlensing planets")
        print(f"median a = {data_au.median():.3f} AU   ({data_pc.median():.3e} pc)")
        print(f"log10(a/AU): mean={log10_au.mean():.4f}  std={log10_au.std(ddof=1):.4f}  "
              f"excess kurtosis={log10_au.kurtosis():.3f}  (0 = Gaussian)")

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 7))

        sns.histplot(data_pc, bins=25, kde=True, color="#FFD300",
                     edgecolor="black", log_scale=True, stat="density")

        plt.title(r"Semi-Major Axis ($a$) - microlensing-discovered planets",
                  fontsize=16, fontweight="bold", pad=15)
        plt.xlabel("Semi-Major Axis [pc]", fontsize=13)
        plt.ylabel("Density", fontsize=13)

        median_pc = data_pc.median()
        plt.axvline(median_pc, color="navy", linestyle="--",
                    label=f"Median: {median_pc:.2e} pc ({data_au.median():.2f} AU)")

        earth_pc = 1.0 * AU_TO_PC
        plt.axvline(earth_pc, color="darkgreen", linestyle=":",
                    label=f"Earth (1 AU): {earth_pc:.2e} pc")

        plt.legend()
        plt.tight_layout()
        plt.show()

        # Emit the array that webapp/app/distributions.py embeds.
        print("\n--- paste into webapp/app/distributions.py as A_MICROLENSING_AU ---")
        values = ", ".join(f"{v:g}" for v in data_au)
        print(textwrap.fill(values, 92, initial_indent="    ",
                            subsequent_indent="    ") + ",")

        print(f"\nPlot completed using {len(data_pc)} data points.")

    except Exception as e:
        print(f"An error occurred: {e}")


plot_semi_major_axis()