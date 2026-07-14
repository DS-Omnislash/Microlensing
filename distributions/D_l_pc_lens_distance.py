"""
Distance to Lens (D_l) distribution from NASA Exoplanet Archive — Microlensing table.

Source: NASA Exoplanet Archive TAP service
        https://exoplanetarchive.ipac.caltech.edu — table: ml, column: sy_dist
        Filtered to default models only (ml_modeldef = 1).

Color: darkcyan
Reference: TdR_RocRC.pdf Image 5, p.23
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


def plot_lens_distance():
    try:
        print("Fetching distance data...")
        response = requests.get(LENS_DATA_URL, verify=False)
        df = pd.read_csv(io.StringIO(response.text), comment="#")

        dist_col = "sy_dist"

        if dist_col in df.columns:
            dist_values = df[dist_col].dropna()

            plt.figure(figsize=(9, 6))
            sns.set_style("whitegrid")

            sns.histplot(dist_values, bins=35, kde=True, color="darkcyan", edgecolor="black")

            plt.title("Distribution of Distance to Lens", fontsize=15, fontweight="bold")
            plt.xlabel("Lens Distance [pc]", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)

            median_val = dist_values.median()
            plt.axvline(median_val, color="darkorange", linestyle="--",
                        label=f"Median: {median_val:.1f} pc")

            plt.legend()
            plt.tight_layout()
            plt.show()

            print(f"Distance plot generated using {len(dist_values)} data points.")
        else:
            print(f"Error: Column '{dist_col}' not found in dataset.")

    except Exception as e:
        print(f"An error occurred: {e}")


plot_lens_distance()