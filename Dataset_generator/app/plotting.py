"""Matplotlib figure generation for the dataset generator web UI.

All figures are rendered server-side and returned as base64-encoded PNG
strings so they can be embedded directly in the HTML response.
"""

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "figure.dpi": 100,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
})


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _kde_curve(values, x_range, n_points=300):
    kde = gaussian_kde(values)
    x = np.linspace(x_range[0], x_range[1], n_points)
    return x, kde(x)


# ---------------------------------------------------------------------------
# "Generated data" plots (shown right after generation)
# ---------------------------------------------------------------------------

def plot_distributions_common(data):
    """Histograms of the shared parameters for the generated dataset."""
    M_star_solar = data["M_star_solar"]
    D_l_pc = data["D_l_pc"]
    D_ls_pc = data["D_ls_pc"]
    v_perp_kms = data["v_perp_kms"]
    u0_all = data["u0_all"]
    t_E_days = data["t_E_days"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    fig.suptitle(f"Generated Parameter Distributions (n = {data['n_total']:,})",
                  fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(M_star_solar, bins=50, density=True, alpha=0.75, color="steelblue", edgecolor="white")
    x, y = _kde_curve(M_star_solar, (0.01, 1.2))
    ax.plot(x, y, "navy", lw=2)
    ax.axvline(np.median(M_star_solar), color="red", ls="--", lw=1.5,
               label=f"Median: {np.median(M_star_solar):.2f} Msun")
    ax.set_xlabel("Lens Mass (Msun)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Lens Mass M*")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(D_l_pc, bins=50, density=True, alpha=0.75, color="teal", edgecolor="white")
    x, y = _kde_curve(D_l_pc, (300, 8500))
    ax.plot(x, y, "darkgreen", lw=2)
    ax.axvline(np.median(D_l_pc), color="orange", ls="--", lw=1.5,
               label=f"Median: {np.median(D_l_pc):.0f} pc")
    ax.set_xlabel("Lens Distance D_l (pc)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Distance to Lens D_l")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(D_ls_pc, bins=50, density=True, alpha=0.75, color="orchid", edgecolor="white")
    x, y = _kde_curve(D_ls_pc, (100, 8000))
    ax.plot(x, y, "purple", lw=2)
    ax.axvline(np.median(D_ls_pc), color="orange", ls="--", lw=1.5,
               label=f"Median: {np.median(D_ls_pc):.0f} pc")
    ax.set_xlabel("Lens-Source Distance D_ls (pc)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Lens-Source Distance D_ls")
    ax.legend()

    ax = axes[1, 1]
    ax.hist(v_perp_kms, bins=50, density=True, alpha=0.75, color="salmon", edgecolor="white")
    ax.axvline(np.median(v_perp_kms), color="darkred", ls="--", lw=1.5,
               label=f"Median: {np.median(v_perp_kms):.0f} km/s")
    ax.set_xlabel("Transversal Velocity v_perp (km/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Lens Velocity v_perp")
    ax.legend()

    ax = axes[2, 0]
    ax.hist(u0_all, bins=50, density=True, alpha=0.75, color="silver", edgecolor="gray")
    ax.axvline(np.median(u0_all), color="blue", ls="--", lw=1.5,
               label=f"Median: {np.median(u0_all):.3f}")
    ax.set_xlabel("Impact Parameter u0")
    ax.set_ylabel("Probability Density")
    ax.set_title("Impact Parameter u0")
    ax.legend()

    ax = axes[2, 1]
    t_E_plot = np.clip(t_E_days, 0, np.percentile(t_E_days, 99))
    ax.hist(t_E_plot, bins=50, density=True, alpha=0.75, color="gold", edgecolor="white")
    ax.axvline(np.median(t_E_days), color="red", ls="--", lw=1.5,
               label=f"Median: {np.median(t_E_days):.1f} days")
    ax.set_xlabel("Einstein Time t_E (days)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Einstein Time t_E (derived)")
    ax.legend()

    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_distributions_binary(data):
    """Histograms of the binary-only parameters. Returns None if n_binary == 0."""
    if data["n_binary"] == 0:
        return None

    q_binary = data["q_binary"]
    a_pc_binary = data["a_pc_binary"]
    e_binary = data["e_binary"]
    alpha_ref_binary = data["alpha_ref_binary"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Generated Binary-Lens Parameter Distributions (n = {data['n_binary']:,})",
                  fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    log10_q = np.log10(q_binary)
    ax.hist(log10_q, bins=40, density=True, alpha=0.75, color="mediumpurple", edgecolor="white")
    ax.axvline(np.median(log10_q), color="orange", ls="--", lw=1.5,
               label=f"Median q: {np.median(q_binary):.2e}")
    ax.set_xlabel("log10(q)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Mass Ratio q")
    ax.legend()

    ax = axes[0, 1]
    log10_a = np.log10(a_pc_binary)
    ax.hist(log10_a, bins=40, density=True, alpha=0.75, color="goldenrod", edgecolor="white")
    ax.axvline(np.median(log10_a), color="blue", ls="--", lw=1.5,
               label=f"Median a: {np.median(a_pc_binary):.2e} pc")
    ax.set_xlabel("log10(a) [pc]")
    ax.set_ylabel("Probability Density")
    ax.set_title("Semi-major Axis a")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(e_binary, bins=40, density=True, alpha=0.75, color="cornflowerblue", edgecolor="white")
    ax.axvline(np.median(e_binary), color="red", ls="--", lw=1.5,
               label=f"Median: {np.median(e_binary):.3f}")
    ax.set_xlabel("Orbital Eccentricity e")
    ax.set_ylabel("Probability Density")
    ax.set_title("Eccentricity e")
    ax.legend()

    ax = axes[1, 1]
    ax.hist(alpha_ref_binary, bins=40, density=True, alpha=0.75, color="lightcoral", edgecolor="white")
    ax.set_xlabel("Trajectory Angle alpha_ref (rad)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Trajectory Angle alpha_ref")

    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_sample_single_lightcurves(data, seed=42):
    """A handful of random single-lens light curves."""
    n_time = data["n_time"]
    tau = np.linspace(-3, 3, n_time)
    u0_all = data["u0_all"]
    n_single = data["n_single"]
    single_lc = data["single_lightcurves"]

    if n_single == 0:
        return None

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), squeeze=False)
    fig.suptitle("Sample Single-Lens Light Curves", fontsize=15, fontweight="bold")

    n_show = min(5, n_single)
    rng_plot = np.random.default_rng(seed)
    single_indices = rng_plot.choice(n_single, size=n_show, replace=False)

    for j in range(5):
        ax = axes[0, j]
        if j < len(single_indices):
            idx = single_indices[j]
            ax.plot(tau, single_lc[idx, :], "b-", lw=1.5)
            ax.set_title(f"Single #{idx}\nu0={u0_all[idx]:.3f}", fontsize=10)
            ax.set_xlabel(r"$\tau = (t-t_0)/t_E$")
            ax.set_ylabel("A(u)")
            ax.set_ylim(bottom=0.9)
        else:
            ax.axis("off")

    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_sample_binary_lightcurves(data, seed=42):
    """A handful of random binary-lens light curves."""
    n_time = data["n_time"]
    tau = np.linspace(-3, 3, n_time)
    n_binary = data["n_binary"]
    binary_lc = data["binary_lightcurves"]

    if n_binary == 0:
        return None

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), squeeze=False)
    fig.suptitle("Sample Binary-Lens Light Curves", fontsize=15, fontweight="bold")

    binary_peaks = binary_lc.max(axis=1)
    n_show_b = min(5, n_binary)
    rng_plot = np.random.default_rng(seed)
    binary_interesting = rng_plot.choice(n_binary, size=n_show_b, replace=False)

    for j in range(5):
        ax = axes[0, j]
        if j < len(binary_interesting):
            idx = binary_interesting[j]
            ax.plot(tau, binary_lc[idx, :], "r-", lw=1.5)
            ax.set_title(
                f"Binary #{idx}\nq={data['q_binary'][idx]:.2e}, A_max={binary_peaks[idx]:.1f}",
                fontsize=10,
            )
            ax.set_xlabel(r"$\tau = (t-t_0)/t_E$")
            ax.set_ylabel("A")
        else:
            ax.axis("off")

    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_coverage(data):
    """Coverage scatter of lens mass vs lens distance (cf. TdR Image 6, p.24-25)."""
    M_star_solar = data["M_star_solar"]
    D_l_pc = data["D_l_pc"]
    n_show = min(5000, len(M_star_solar))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(M_star_solar[:n_show], D_l_pc[:n_show], c="slateblue", alpha=0.3, s=10, edgecolors="none")
    ax.axvline(1.0, color="orange", ls="--", lw=1.5, label="Sun (1 Msun)")
    ax.set_xlabel("Lens Mass [Msun]")
    ax.set_ylabel("Lens Distance D_l [pc]")
    ax.set_title("Coverage: Lens Mass vs. Lens Distance (cf. TdR Image 6)")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Validation plots: generated histogram + literature reference curve overlay
# ---------------------------------------------------------------------------

def plot_validation_common(data):
    """Generated histograms overlaid with the literature reference shapes
    described in TDR_ROC.pdf pp. 21-26 (Images 2, 4, 5, 7).

    Returns (image_b64, stats) where stats is a list of dicts describing
    each check.
    """
    M_star_solar = data["M_star_solar"]
    D_l_pc = data["D_l_pc"]
    D_ls_pc = data["D_ls_pc"]
    v_perp_kms = data["v_perp_kms"]
    u0_all = data["u0_all"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Validation vs. Literature Distributions (TdR pp. 21-26)",
                  fontsize=15, fontweight="bold")
    stats_out = []

    # --- Lens mass: bimodal mixture, expect peak/median near 0.45 Msun (p.21, Image 2) ---
    ax = axes[0, 0]
    ax.hist(M_star_solar, bins=50, density=True, alpha=0.6, color="steelblue", edgecolor="white",
            label="Generated data")
    x, y = _kde_curve(M_star_solar, (0.01, 1.2))
    ax.plot(x, y, "navy", lw=2, label="KDE of generated data")
    ax.axvline(0.45, color="red", ls="--", lw=1.5, label="TdR Image 2 peak (~0.45 Msun)")
    median_mass = float(np.median(M_star_solar))
    ax.set_xlabel("Lens Mass (Msun)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Lens Mass M* [Image 2, p.21]")
    ax.legend()
    in_range = 0.30 <= median_mass <= 0.60
    stats_out.append({
        "parameter": "Lens Mass M* (Msun)",
        "reference": "TdR Image 2 (p.21): bimodal, primary peak ~0.45 Msun, decline beyond 0.8 Msun",
        "observed": f"median = {median_mass:.3f} Msun",
        "expected": "median in [0.30, 0.60] Msun",
        "status": "OK" if in_range else "CHECK",
    })

    # --- Distance to lens: bulge-peaked, expect median ~6-7 kpc (p.24, Image 5) ---
    ax = axes[0, 1]
    ax.hist(D_l_pc, bins=50, density=True, alpha=0.6, color="teal", edgecolor="white",
            label="Generated data")
    x, y = _kde_curve(D_l_pc, (300, 8500))
    ax.plot(x, y, "darkgreen", lw=2, label="KDE of generated data")
    ax.axvline(6800, color="red", ls="--", lw=1.5, label="TdR Image 5 bulge peak (~6800 pc)")
    median_dl = float(np.median(D_l_pc))
    ax.set_xlabel("Lens Distance D_l (pc)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Distance to Lens D_l [Image 5, p.24]")
    ax.legend()
    in_range = 4500 <= median_dl <= 8000
    stats_out.append({
        "parameter": "Distance to Lens D_l (pc)",
        "reference": "TdR Image 5 (p.24): Galactic bulge peak at 6-7 kpc (Paczynski 1991)",
        "observed": f"median = {median_dl:.0f} pc",
        "expected": "median in [4500, 8000] pc",
        "status": "OK" if in_range else "CHECK",
    })

    # --- Lens-source distance: structured/irregular, ~uniform with mild peaks (p.23, Image 4) ---
    ax = axes[1, 0]
    ax.hist(D_ls_pc, bins=50, density=True, alpha=0.6, color="orchid", edgecolor="white",
            label="Generated data")
    ax.axhline(1.0 / (8000 - 100), color="red", ls="--", lw=1.5, label="Uniform[100,8000] reference")
    median_dls = float(np.median(D_ls_pc))
    ax.set_xlabel("Lens-Source Distance D_ls (pc)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Lens-Source Distance D_ls [Image 4, p.23]")
    ax.legend()
    in_range = 1500 <= median_dls <= 4500
    stats_out.append({
        "parameter": "Lens-Source Distance D_ls (pc)",
        "reference": "TdR Image 4 (p.23): irregular, roughly uniform on [100, 8000] pc, median ~2900 pc",
        "observed": f"median = {median_dls:.0f} pc",
        "expected": "median in [1500, 4500] pc",
        "status": "OK" if in_range else "CHECK",
    })

    # --- Impact parameter: truncated exponential, OGLE-IV bias (p.26, Image 7) ---
    ax = axes[1, 1]
    ax.hist(u0_all, bins=50, density=True, alpha=0.6, color="silver", edgecolor="gray",
            label="Generated data")
    x_u0 = np.linspace(0, 1, 300)
    trunc_exp_pdf = 3.0 * np.exp(-3.0 * x_u0) / (1.0 - np.exp(-3.0))
    ax.plot(x_u0, trunc_exp_pdf, "blue", lw=2, label="Trunc. Exp(lambda=3) [TdR Image 7]")
    ax.axhline(1.0, color="green", ls=":", lw=1.5, label="Theoretical uniform(0,1) [Paczynski 1986]")
    ax.set_xlabel("Impact Parameter u0")
    ax.set_ylabel("Probability Density")
    ax.set_title("Impact Parameter u0 [Image 7, p.26]")
    ax.legend()

    ks_stat, ks_p = stats.kstest(u0_all, "truncexpon", args=(1.0 * 3.0, 0, 1.0 / 3.0))
    stats_out.append({
        "parameter": "Impact Parameter u0",
        "reference": "TdR Image 7 (p.26): OGLE-IV observational bias, truncated exponential (lambda=3) on [0,1]",
        "observed": f"KS statistic = {ks_stat:.4f} (p = {ks_p:.3g}), median = {np.median(u0_all):.3f}",
        "expected": "Distribution concentrated near 0, decaying toward 1 (not flat/uniform)",
        "status": "OK" if ks_p > 0.01 else "CHECK",
    })

    # --- Lens velocity: Maxwell-Boltzmann, mode 200 km/s (p.25) ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(v_perp_kms, bins=50, density=True, alpha=0.6, color="salmon", edgecolor="white",
             label="Generated data")
    x_vel = np.linspace(0, v_perp_kms.max(), 300)
    mb_scale = 200.0 / np.sqrt(2.0)
    mb_pdf = stats.maxwell.pdf(x_vel, scale=mb_scale)
    ax2.plot(x_vel, mb_pdf, "darkred", lw=2, label="Maxwell-Boltzmann (mode=200 km/s) [TdR p.25]")
    ax2.set_xlabel("Transversal Velocity v_perp (km/s)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Lens Velocity v_perp [TdR p.25, Rahvar 2015]")
    ax2.legend()
    fig2.tight_layout()
    velocity_img = _fig_to_b64(fig2)

    ks_stat_v, ks_p_v = stats.kstest(v_perp_kms, "maxwell", args=(0, mb_scale))
    stats_out.append({
        "parameter": "Lens Velocity v_perp (km/s)",
        "reference": "TdR p.25 (Rahvar 2015): Maxwell-Boltzmann distribution, mode = 200 km/s",
        "observed": f"KS statistic = {ks_stat_v:.4f} (p = {ks_p_v:.3g}), median = {np.median(v_perp_kms):.1f} km/s",
        "expected": "Maxwell-Boltzmann shape with mode near 200 km/s",
        "status": "OK" if ks_p_v > 0.01 else "CHECK",
    })

    fig.tight_layout()
    return _fig_to_b64(fig), velocity_img, stats_out


def plot_validation_binary(data):
    """Generated binary-only histograms overlaid with TdR pp. 22, 27-29 references.

    Returns (image_b64, stats) or (None, []) if n_binary == 0.
    """
    if data["n_binary"] == 0:
        return None, []

    q_binary = data["q_binary"]
    a_pc_binary = data["a_pc_binary"]
    e_binary = data["e_binary"]
    alpha_ref_binary = data["alpha_ref_binary"]
    stats_out = []

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Validation of Binary-Lens Parameters vs. Literature (TdR pp. 22, 27-29)",
                  fontsize=15, fontweight="bold")

    # --- Mass ratio q: log-normal, peak ~1.43e-3 (p.22, Image 3) ---
    ax = axes[0, 0]
    log10_q = np.log10(q_binary)
    ax.hist(log10_q, bins=40, density=True, alpha=0.6, color="mediumpurple", edgecolor="white",
            label="Generated data")
    ax.axvline(np.log10(1.43e-3), color="red", ls="--", lw=1.5, label="TdR Image 3 peak (q~1.43e-3)")
    median_q = float(np.median(q_binary))
    ax.set_xlabel("log10(q)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Mass Ratio q [Image 3, p.22]")
    ax.legend()
    in_range = 1e-4 <= median_q <= 1e-2
    stats_out.append({
        "parameter": "Mass Ratio q",
        "reference": "TdR Image 3 (p.22): log-normal, median ~1.43e-3 (NASA Exoplanet Archive)",
        "observed": f"median = {median_q:.2e}",
        "expected": "median in [1e-4, 1e-2]",
        "status": "OK" if in_range else "CHECK",
    })

    # --- Semi-major axis a: bimodal log-space, dominant peak 1e-7..1e-6, secondary ~1e-5 (p.27, Image 8) ---
    ax = axes[0, 1]
    log10_a = np.log10(a_pc_binary)
    ax.hist(log10_a, bins=40, density=True, alpha=0.6, color="goldenrod", edgecolor="white",
            label="Generated data")
    ax.axvline(-6.4, color="red", ls="--", lw=1.5, label="Dominant peak ~1e-6.4 pc [Image 8]")
    ax.axvline(-5.0, color="blue", ls="--", lw=1.5, label="Secondary bump ~1e-5 pc [Image 8]")
    median_a = float(np.median(a_pc_binary))
    ax.set_xlabel("log10(a) [pc]")
    ax.set_ylabel("Probability Density")
    ax.set_title("Semi-major Axis a [Image 8, p.27]")
    ax.legend()
    in_range = -7.5 <= np.log10(median_a) <= -5.0
    stats_out.append({
        "parameter": "Semi-major Axis a (pc)",
        "reference": "TdR Image 8 (p.27): bimodal in log-space, dominant peak ~1e-6.4 pc, secondary ~1e-5 pc",
        "observed": f"median = {median_a:.2e} pc",
        "expected": "median in [10^-7.5, 10^-5] pc",
        "status": "OK" if in_range else "CHECK",
    })

    # --- Eccentricity e: Beta(1.5, 12), mode ~0.043 (p.27-28, Image 9) ---
    ax = axes[1, 0]
    ax.hist(e_binary, bins=40, density=True, alpha=0.6, color="cornflowerblue", edgecolor="white",
            label="Generated data")
    x_ecc = np.linspace(0, 0.99, 300)
    beta_pdf = stats.beta.pdf(x_ecc, 1.5, 12.0)
    ax.plot(x_ecc, beta_pdf, "darkblue", lw=2, label="Beta(1.5, 12) [TdR Image 9]")
    ax.set_xlabel("Orbital Eccentricity e")
    ax.set_ylabel("Probability Density")
    ax.set_title("Eccentricity e [Image 9, p.27-28]")
    ax.legend()
    ks_stat_e, ks_p_e = stats.kstest(e_binary, "beta", args=(1.5, 12.0))
    stats_out.append({
        "parameter": "Orbital Eccentricity e",
        "reference": "TdR Image 9 (p.27-28): exponential-like decay, mode ~0.043, modeled as Beta(1.5, 12)",
        "observed": f"KS statistic = {ks_stat_e:.4f} (p = {ks_p_e:.3g}), median = {np.median(e_binary):.3f}",
        "expected": "Distribution concentrated near 0 with long tail toward 1",
        "status": "OK" if ks_p_e > 0.01 else "CHECK",
    })

    # --- Trajectory angle alpha_ref: uniform(0, 2pi) (p.29) ---
    ax = axes[1, 1]
    ax.hist(alpha_ref_binary, bins=40, density=True, alpha=0.6, color="lightcoral", edgecolor="white",
            label="Generated data")
    ax.axhline(1.0 / (2 * np.pi), color="red", ls="--", lw=1.5, label="Uniform(0, 2*pi) [TdR p.29]")
    ax.set_xlabel("Trajectory Angle alpha_ref (rad)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Trajectory Angle alpha_ref [p.29]")
    ax.legend()
    ks_stat_a, ks_p_a = stats.kstest(alpha_ref_binary, "uniform", args=(0, 2 * np.pi))
    stats_out.append({
        "parameter": "Trajectory Angle alpha_ref (rad)",
        "reference": "TdR p.29: 'theoretically completely random' -> uniform(0, 2*pi)",
        "observed": f"KS statistic = {ks_stat_a:.4f} (p = {ks_p_a:.3g})",
        "expected": "Flat (uniform) distribution over [0, 2*pi]",
        "status": "OK" if ks_p_a > 0.01 else "CHECK",
    })

    fig.tight_layout()
    return _fig_to_b64(fig), stats_out
