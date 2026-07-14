"""
Orbital Eccentricity (e) distribution from NASA Exoplanet Archive — Planetary Systems table.

Filtered to planets with a measured upper uncertainty (pl_orbeccenerr1 IS NOT NULL),
which excludes entries where e=0 was assumed (circular orbit convention) rather than
physically measured. This gives the real underlying eccentricity distribution.

Source: NASA Exoplanet Archive TAP service
        https://exoplanetarchive.ipac.caltech.edu — table: ps
        Columns: pl_orbeccen, pl_orbeccenerr1
        Filtered to default_flag = 1 AND pl_orbeccenerr1 IS NOT NULL.

Color: #2563eb (blue)
Reference: TdR_RocRC.pdf Image 9, p.26-27
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

QUERY = (
    "SELECT pl_name, pl_orbeccen, pl_orbeccenerr1 "
    "FROM ps "
    "WHERE default_flag = 1 AND pl_orbeccenerr1 IS NOT NULL"
)
URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    f"?query={QUERY.replace(' ', '+')}&format=csv"
)


def plot_eccentricity():
    try:
        print("Fetching planets with confirmed orbital measurements...")
        response = requests.get(URL, verify=False)
        df = pd.read_csv(io.StringIO(response.text), comment="#")

        real_data = df["pl_orbeccen"].dropna()

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(14, 8))

        sns.histplot(real_data, bins=45, kde=True, color="#2563eb",
                     edgecolor="black", alpha=0.6, label="Physically Measured Orbits")

        plt.title("Distribution of Planet Orbital Eccentricity", fontsize=18, fontweight="bold", pad=20)
        plt.xlabel("Orbital Eccentricity (e)", fontsize=14)
        plt.ylabel("Frequency (Count)", fontsize=14)
        plt.xlim(0, 1)

        EARTH_E = 0.0167
        plt.axvline(EARTH_E, color="#16a34a", linestyle="-", linewidth=3,
                    label=r"Earth Reference: $e \approx 0.0167$")

        # Mode in linear space
        counts, bin_edges = np.histogram(real_data, bins=100)
        mode_idx = counts.argmax()
        most_probable_e = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2

        plt.axvline(most_probable_e, color="#dc2626", linestyle="--", linewidth=2,
                    label=f"Most Probable e: ~{most_probable_e:.3f}")

        plt.legend(frameon=True, fontsize=12)
        plt.tight_layout()
        plt.show()

        print(f"\n--- Statistical Summary ---")
        print(f"Total measured planets: {len(real_data)}")
        print(f"Mode (Most Probable e): {most_probable_e:.3f}")
        print(f"Median e: {real_data.median():.3f}")

    except Exception as e:
        print(f"Error fetching data: {e}")


plot_eccentricity()