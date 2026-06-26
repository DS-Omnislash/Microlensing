"""
Semi-Major Axis (a) distribution from NASA Exoplanet Archive — Planetary Systems table.

Units converted from AU to parsecs using the IAU 2015 B2 definition:
    1 pc = 648 000 / pi  AU  →  1 AU = pi / 648 000 pc

Source: NASA Exoplanet Archive TAP service
        https://exoplanetarchive.ipac.caltech.edu — table: ps, column: pl_orbsmax
        Filtered to default parameters only (default_flag = 1).

Color: #FFD300 (yellow)
Reference: TDR_ROC.pdf Image 8, p.26
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

# IAU 2015 Res B2: 1 pc = 648 000 / pi AU
AU_TO_PC = np.pi / 648_000.0

PS_DATA_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+pl_name,pl_orbsmax,default_flag+from+ps+where+default_flag=1&format=csv"
)


def plot_semi_major_axis():
    try:
        print("Fetching data from NASA Exoplanet Archive...")
        response = requests.get(PS_DATA_URL, verify=False)
        df = pd.read_csv(io.StringIO(response.text), comment="#")

        if "pl_orbsmax" in df.columns:
            data_au = df["pl_orbsmax"].dropna()
            data_pc = data_au * AU_TO_PC

            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(12, 7))

            sns.histplot(data_pc, bins=60, kde=True, color="#FFD300",
                         edgecolor="black", log_scale=True)

            plt.title(r"Distribution of Planet Semi-Major Axis ($a$)", fontsize=16, fontweight="bold", pad=15)
            plt.xlabel("Semi-Major Axis [pc]", fontsize=13)
            plt.ylabel("Frequency", fontsize=13)

            median_pc = data_pc.median()
            plt.axvline(median_pc, color="navy", linestyle="--",
                        label=f"Median: {median_pc:.2e} pc")

            earth_pc = 1.0 * AU_TO_PC
            plt.axvline(earth_pc, color="darkgreen", linestyle=":",
                        label=f"Earth (1 AU): {earth_pc:.2e} pc")

            plt.legend()
            plt.tight_layout()
            plt.show()

            print(f"Plot completed using {len(data_pc)} data points.")
        else:
            print("Error: Column 'pl_orbsmax' not found.")

    except Exception as e:
        print(f"An error occurred: {e}")


plot_semi_major_axis()