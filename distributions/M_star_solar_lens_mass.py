"""
Lens Mass (M*) distribution from NASA Exoplanet Archive — Microlensing table.

Source: NASA Exoplanet Archive TAP service
        https://exoplanetarchive.ipac.caltech.edu — table: ml, column: st_mass
        Filtered to default models only (ml_modeldef = 1).

Color: dodgerblue
Reference: TdR_RocRC.pdf Image 2, p.20
"""

import io

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

DATA_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+*+from+ml+where+ml_modeldef=1&format=csv"
)


def plot_lens_mass_distribution(source):
    try:
        response = requests.get(source, verify=False)
        df = pd.read_csv(io.StringIO(response.text), comment="#")

        target_col = "Lens Mass [Solar mass]"
        if target_col not in df.columns:
            if "st_mass" in df.columns:
                target_col = "st_mass"
            else:
                print(f"Error: Could not find mass column. Available: {df.columns.tolist()[:5]}")
                return

        mass_values = df[target_col].dropna()

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        sns.histplot(mass_values, bins=30, kde=True, color="dodgerblue", edgecolor="black")

        plt.title("Distribution of Lens Mass in Microlensing Systems", fontsize=16, fontweight="bold")
        plt.xlabel(r"Lens Mass ($M_\odot$)", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)

        median_mass = mass_values.median()
        plt.axvline(median_mass, color="red", linestyle="--",
                    label=f"Median: {median_mass:.2f} $M_{{\\odot}}$")
        plt.legend()

        plt.tight_layout()
        plt.show()
        print(f"Successfully plotted {len(mass_values)} data points.")

    except Exception as e:
        print(f"An error occurred: {e}")


plot_lens_mass_distribution(DATA_URL)