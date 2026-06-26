"""
Mass Ratio (q) distribution from NASA Exoplanet Archive — Microlensing table.

q = M_planet / M_lens, computed from host mass (st_mass, Solar) and planet
mass (pl_massj, Jupiter), converted to kg for a dimensionally rigorous ratio.

The peak (mode) in log-space is identified and annotated.

Source: NASA Exoplanet Archive TAP service
        https://exoplanetarchive.ipac.caltech.edu — table: ml
        Columns: st_mass (host mass, Msun), pl_massj (planet mass, Mjup)
        Filtered to default models only (ml_modeldef = 1).

Color: darkslateblue
Reference: TDR_ROC.pdf Image 3, p.21
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

LENS_DATA_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+*+from+ml+where+ml_modeldef=1&format=csv"
)

M_SUN_KG  = 1.988e30
M_JUP_KG  = 1.898e27


def plot_mass_ratio():
    try:
        print("Fetching mass data from NASA...")
        response = requests.get(LENS_DATA_URL, verify=False)
        df = pd.read_csv(io.StringIO(response.text), comment="#")

        df_clean = df.dropna(subset=["st_mass", "pl_massj"]).copy()

        if not df_clean.empty:
            df_clean["m_lens_kg"]   = df_clean["st_mass"]  * M_SUN_KG
            df_clean["m_planet_kg"] = df_clean["pl_massj"] * M_JUP_KG
            df_clean["m_tot"]       = df_clean["m_lens_kg"] + df_clean["m_planet_kg"]

            # q = M_planet / M_lens  (equivalent to the ratio of fractional masses)
            df_clean["q"] = (df_clean["m_planet_kg"] / df_clean["m_tot"]) / \
                            (df_clean["m_lens_kg"]   / df_clean["m_tot"])

            # Peak (mode) in log-space
            log_data = np.log10(df_clean["q"])
            counts, bin_edges = np.histogram(log_data, bins=35)
            max_bin = np.argmax(counts)
            peak_val = 10 ** ((bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2)

            plt.figure(figsize=(9, 6))
            sns.set_style("whitegrid")

            sns.histplot(df_clean["q"], bins=35, kde=True, color="darkslateblue",
                         edgecolor="black", log_scale=True)

            plt.axvline(peak_val, color="red", linestyle="--", linewidth=2,
                        label=f"Peak (Mode): {peak_val:.2e}")

            plt.title("Distribution of Mass Ratio (Peak identified)", fontsize=15, fontweight="bold")
            plt.xlabel(r"$q = M_\mathrm{planet} / M_\mathrm{lens}$", fontsize=14)
            plt.ylabel("Frequency", fontsize=12)

            plt.legend()
            plt.tight_layout()
            plt.show()

            print(f"Plot generated with {len(df_clean)} events.")
            print(f"Peak value: {peak_val:.2e}")
        else:
            print("Error: No data available for mass columns.")

    except Exception as e:
        print(f"An error occurred: {e}")


plot_mass_ratio()