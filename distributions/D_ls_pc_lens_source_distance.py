"""
Lens-Source Distance (D_ls) distribution from NASA Exoplanet Archive — Microlensing table.

D_ls is reconstructed as D_ls = D_source - D_lens (line-of-sight separation).

Source: NASA Exoplanet Archive TAP service
        https://exoplanetarchive.ipac.caltech.edu — table: ml
        Columns: ml_dists (source distance), sy_dist (lens distance)
        Filtered to default models only (ml_modeldef = 1).

Color: darkmagenta
Reference: TDR_ROC.pdf Image 4, p.22
"""

import io

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

LENS_DATA_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+*+from+ml+where+ml_modeldef=1&format=csv"
)


def plot_lens_source_distance():
    try:
        print("Fetching data from NASA...")
        response = requests.get(LENS_DATA_URL, verify=False)
        df = pd.read_csv(io.StringIO(response.text), comment="#")

        d_source_col = "ml_dists"
        d_lens_col = "sy_dist"

        df_clean = df.dropna(subset=[d_source_col, d_lens_col]).copy()

        if not df_clean.empty:
            df_clean["delta_dist"] = df_clean[d_source_col] - df_clean[d_lens_col]

            plt.figure(figsize=(9, 6))
            sns.set_style("whitegrid")

            sns.histplot(df_clean["delta_dist"], bins=35, kde=True,
                         color="darkmagenta", edgecolor="black")

            plt.title("Calculated Physical Distance (Source - Lens)", fontsize=15, fontweight="bold")
            plt.xlabel("Distance Difference D_ls [pc]", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)

            delta_median = df_clean["delta_dist"].median()
            plt.axvline(delta_median, color="darkorange", linestyle="--",
                        label=f"Median: {delta_median:.1f} pc")

            plt.legend()
            plt.tight_layout()
            plt.show()

            print(f"Plot generated using {len(df_clean)} events.")
        else:
            print("Error: No data available for both distances.")

    except Exception as e:
        print(f"An error occurred: {e}")


plot_lens_source_distance()