"""
Impact Parameter (u0) distribution from OGLE-IV survey data.

Source: Mroz et al. (2019), table3.dat — Warsaw Observatory
        https://www.astrouw.edu.pl/ogle/ogle4/microlensing_maps/table3.dat

Two distributions are shown:
  - Grey  : raw observed distribution (biased by telescope detection efficiency)
  - Blue  : efficiency-corrected real physical distribution (used in TdR_RocRC.pdf)
"""

import io

import matplotlib.pyplot as plt
import pandas as pd
import requests

col_specs = [
    (0, 16),   # name
    (17, 26),  # field
    (27, 33),  # star_id
    (34, 36),  # ra_h
    (37, 39),  # ra_m
    (40, 45),  # ra_s
    (46, 49),  # dec_d
    (50, 52),  # dec_m
    (53, 57),  # dec_s
    (58, 67),  # ra
    (68, 77),  # dec
    (78, 87),  # glon
    (88, 97),  # glat
    (98, 109), # t0_best
    (110, 117),# tE_best
    (118, 123),# u0_best
    (124, 130),# Is_best
    (131, 136),# fs_best
    (137, 148),# t0_med
    (149, 156),# t0_err1
    (157, 163),# t0_err2
    (164, 171),# tE_med
    (172, 179),# tE_err1
    (180, 186),# tE_err2
    (187, 192),# u0_med
    (193, 199),# u0_err1
    (200, 205),# u0_err2
    (206, 212),# Is_med
    (213, 219),# Is_err1
    (220, 225),# Is_err2
    (226, 231),# fs_med
    (232, 238),# fs_err1
    (239, 244),# fs_err2
    (245, 251),# weight  — efficiency correction
    (252, 270), # ews_id
]

cols = [
    "name", "field", "star_id", "ra_h", "ra_m", "ra_s", "dec_d", "dec_m", "dec_s",
    "ra", "dec", "glon", "glat", "t0", "tE", "u0", "Is", "fs", "t0_med", "t0_err1",
    "t0_err2", "tE_med", "tE_err1", "tE_err2", "u0_med", "u0_err1", "u0_err2",
    "Is_med", "Is_err1", "Is_err2", "fs_med", "fs_err1", "fs_err2", "weight", "ews_id",
]

url = "https://www.astrouw.edu.pl/ogle/ogle4/microlensing_maps/table3.dat"

print("Parsing fixed-width data...")
response = requests.get(url, verify=False)
df = pd.read_fwf(io.StringIO(response.text), colspecs=col_specs, names=cols, comment="#")

df["u0"] = pd.to_numeric(df["u0"], errors="coerce")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df = df.dropna(subset=["u0", "weight"])

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(df["u0"], bins=40, range=(0, 1), density=True,
        alpha=0.3, color="gray", label="Observed (Telescope Bias)")

ax.hist(df["u0"], bins=40, range=(0, 1), weights=df["weight"], density=True,
        histtype="step", linewidth=2, color="blue", label="Corrected (Real Physical Dist.)")

ax.axhline(y=1.0, color="red", linestyle="--", label=r"Theoretical Uniform ($u_0 \sim 1$)")

ax.set_title(r"OGLE-IV Impact Parameter ($u_0$) Distribution", fontsize=14)
ax.set_xlabel(r"$u_0$ (Einstein radii)", fontsize=12)
ax.set_ylabel("Probability Density", fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()

print("Done.")