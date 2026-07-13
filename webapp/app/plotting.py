"""Matplotlib figure generation for the dataset generator web UI.

All figures are rendered server-side and returned as base64-encoded PNG
strings so they can be embedded directly in the HTML response.
"""

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

_NOISE_DIR = Path(__file__).resolve().parent.parent.parent / "noise_analysis"

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


# ---------------------------------------------------------------------------
# Validation: flexible, works for both generated and uploaded datasets.
# Only plots parameters that are present in `data`.
# Page references follow TDR_ROC.pdf physical page numbering (document
# starts counting from PDF page 2, so PDF p.N = physical p.(N-1)).
# ---------------------------------------------------------------------------

def plot_validation_available(data):
    """Validate a dataset against TdR reference distributions.

    Accepts any data dict produced by generate_dataset() or _data_from_df().
    Only plots parameters that are present and have sufficient samples.

    Returns (common_img, velocity_img, binary_img, stats_list).
    Any image may be None if no data is available for that group.
    """
    stats_out = []

    # ── Which shared-parameter panels can we plot? ──────────────────────────
    panels = []
    if "M_star_solar" in data and len(data["M_star_solar"]) > 1:
        panels.append("mass")
    if "D_l_pc" in data and len(data["D_l_pc"]) > 1:
        panels.append("dl")
    if "D_ls_pc" in data and len(data["D_ls_pc"]) > 1:
        panels.append("dls")
    if "u0_all" in data and len(data["u0_all"]) > 1:
        panels.append("u0")
    if "I_s_mag" in data and data["I_s_mag"] is not None and len(data["I_s_mag"]) > 1:
        panels.append("is_mag")

    common_img = None
    if panels:
        n = len(panels)
        ncols = 2
        nrows = (n + 1) // 2
        fig, axes_grid = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows), squeeze=False)
        n_total = data.get("n_total", None)
        n_label = f"n = {n_total:,}" if isinstance(n_total, int) else "uploaded dataset"
        fig.suptitle(f"Validation vs. Literature Distributions ({n_label})",
                     fontsize=15, fontweight="bold")
        ax_list = [axes_grid[r][c] for r in range(nrows) for c in range(ncols)]
        for ax in ax_list[n:]:
            ax.axis("off")

        idx = 0

        if "mass" in panels:
            M_star = data["M_star_solar"]
            ax = ax_list[idx]; idx += 1
            ax.hist(M_star, bins=50, density=True, alpha=0.6, color="steelblue",
                    edgecolor="white", label="Data")
            x, y = _kde_curve(M_star, (0.01, 1.2))
            ax.plot(x, y, "navy", lw=2, label="KDE of data")
            ax.axvline(0.45, color="red", ls="--", lw=1.5, label="TdR Image 2 peak (~0.45 Msun)")
            ax.set_xlabel("Lens Mass (Msun)")
            ax.set_ylabel("Probability Density")
            ax.set_title("Lens Mass M* [Image 2, p.20]")
            ax.legend()
            median_mass = float(np.median(M_star))
            in_range = 0.30 <= median_mass <= 0.60
            stats_out.append({
                "parameter": "Lens Mass M* (Msun)",
                "reference": "TdR Image 2 (p.20): bimodal, primary peak ~0.45 Msun, decline beyond 0.8 Msun",
                "observed": f"median = {median_mass:.3f} Msun",
                "expected": "median in [0.30, 0.60] Msun",
                "status": "OK" if in_range else "CHECK",
            })

        if "dl" in panels:
            D_l = data["D_l_pc"]
            ax = ax_list[idx]; idx += 1
            ax.hist(D_l, bins=50, density=True, alpha=0.6, color="teal",
                    edgecolor="white", label="Data")
            x, y = _kde_curve(D_l, (300, 8500))
            ax.plot(x, y, "darkgreen", lw=2, label="KDE of data")
            ax.axvline(6800, color="red", ls="--", lw=1.5, label="TdR Image 5 bulge peak (~6800 pc)")
            ax.set_xlabel("Lens Distance D_l (pc)")
            ax.set_ylabel("Probability Density")
            ax.set_title("Distance to Lens D_l [Image 5, p.23]")
            ax.legend()
            median_dl = float(np.median(D_l))
            in_range = 4500 <= median_dl <= 8000
            stats_out.append({
                "parameter": "Distance to Lens D_l (pc)",
                "reference": "TdR Image 5 (p.23): Galactic bulge peak at 6-7 kpc (Paczynski 1991)",
                "observed": f"median = {median_dl:.0f} pc",
                "expected": "median in [4500, 8000] pc",
                "status": "OK" if in_range else "CHECK",
            })

        if "dls" in panels:
            D_ls = data["D_ls_pc"]
            ax = ax_list[idx]; idx += 1
            ax.hist(D_ls, bins=50, density=True, alpha=0.6, color="orchid",
                    edgecolor="white", label="Data")
            ax.axhline(1.0 / (8000 - 100), color="red", ls="--", lw=1.5,
                       label="Uniform[100, 8000] reference")
            ax.set_xlabel("Lens-Source Distance D_ls (pc)")
            ax.set_ylabel("Probability Density")
            ax.set_title("Lens-Source Distance D_ls [Image 4, p.22]")
            ax.legend()
            median_dls = float(np.median(D_ls))
            in_range = 1500 <= median_dls <= 4500
            stats_out.append({
                "parameter": "Lens-Source Distance D_ls (pc)",
                "reference": "TdR Image 4 (p.22): irregular, roughly uniform on [100, 8000] pc, median ~2900 pc",
                "observed": f"median = {median_dls:.0f} pc",
                "expected": "median in [1500, 4500] pc",
                "status": "OK" if in_range else "CHECK",
            })

        if "u0" in panels:
            u0 = data["u0_all"]
            ax = ax_list[idx]; idx += 1
            ax.hist(u0, bins=50, density=True, alpha=0.6, color="silver",
                    edgecolor="gray", label="Data")
            x_u0 = np.linspace(0, 1, 300)
            trunc_exp_pdf = 3.0 * np.exp(-3.0 * x_u0) / (1.0 - np.exp(-3.0))
            ax.plot(x_u0, trunc_exp_pdf, "blue", lw=2, label="Trunc. Exp(lambda=3) [TdR Image 7]")
            ax.axhline(1.0, color="green", ls=":", lw=1.5,
                       label="Uniform(0,1) [Paczynski 1986]")
            ax.set_xlabel("Impact Parameter u0")
            ax.set_ylabel("Probability Density")
            ax.set_title("Impact Parameter u0 [Image 7, p.25]")
            ax.legend()
            ks_stat, ks_p = stats.kstest(u0, "truncexpon", args=(1.0 * 3.0, 0, 1.0 / 3.0))
            stats_out.append({
                "parameter": "Impact Parameter u0",
                "reference": "TdR Image 7 (p.25): OGLE-IV observational bias, truncated exponential (lambda=3) on [0,1]",
                "observed": f"KS statistic = {ks_stat:.4f} (p = {ks_p:.3g}), median = {np.median(u0):.3f}",
                "expected": "Distribution concentrated near 0, decaying toward 1 (not flat/uniform)",
                "status": "OK" if ks_p > 0.01 else "CHECK",
            })

        if "is_mag" in panels:
            I_s = data["I_s_mag"]
            ax = ax_list[idx]; idx += 1
            ax.hist(I_s, bins=50, density=True, alpha=0.6, color="#e67e22",
                    edgecolor="white", label="Data")

            # In OGLE mode I_s is not sampled from the theoretical TdR Beta: it is
            # bootstrapped from the real OGLE-IV event catalogue (paired with its
            # blend fraction f_s), so it must be validated against that catalogue.
            Is_ref = None
            if data.get("ogle_noise"):
                try:
                    Is_ref = np.load(_NOISE_DIR / "blend_model.npz")["Is"]
                except (FileNotFoundError, KeyError):
                    Is_ref = None

            if Is_ref is not None:
                ax.hist(Is_ref, bins=50, density=True, histtype="step", lw=2,
                        color="darkorange", label="OGLE-IV catalogue (Is_med)")
                ax.axvline(np.median(I_s), color="navy", ls="--", lw=1.5,
                           label=f"Median: {np.median(I_s):.2f} mag")
                ax.set_xlabel("Source Baseline Magnitude I_s (mag)")
                ax.set_ylabel("Probability Density")
                ax.set_title("Source Baseline Magnitude I_s [OGLE-IV catalogue]")
                ax.legend()

                rng_is = np.random.default_rng(0)
                n_is = min(len(I_s), 10_000)
                ks_stat_is, ks_p_is = stats.ks_2samp(
                    rng_is.choice(I_s, size=n_is, replace=True), Is_ref
                )
                median_is = float(np.median(I_s))
                ref_med = float(np.median(Is_ref))
                # Judge on the KS statistic (effect size): the generated values are
                # bootstrapped from this very catalogue, so they should match closely.
                in_range_is = ks_stat_is < 0.05
                stats_out.append({
                    "parameter": "Source Baseline Magnitude I_s (mag)",
                    "reference": f"OGLE-IV event catalogue (blend_model.npz, Is_med), median {ref_med:.2f} mag",
                    "observed": f"KS statistic = {ks_stat_is:.4f} (p = {ks_p_is:.3g}), median = {median_is:.2f} mag",
                    "expected": f"Bootstrapped from the catalogue (KS stat < 0.05, median ~{ref_med:.2f} mag)",
                    "status": "OK" if in_range_is else "CHECK",
                })
            else:
                x_is = np.linspace(14, 22, 300)
                z_ref = (x_is - 14.0) / 8.0
                beta_pdf = stats.beta.pdf(z_ref, 15.0, 6.0) / 8.0
                ax.plot(x_is, beta_pdf, "darkorange", lw=2,
                        label="Beta(15,6) scaled [TdR Image 10]")
                ax.axvline(np.median(I_s), color="navy", ls="--", lw=1.5,
                           label=f"Median: {np.median(I_s):.2f} mag")
                ax.set_xlabel("Source Baseline Magnitude I_s (mag)")
                ax.set_ylabel("Probability Density")
                ax.set_title("Source Baseline Magnitude I_s [Image 10, p.28]")
                ax.legend()
                z_data = np.clip((I_s - 14.0) / 8.0, 0.0, 1.0)
                ks_stat_is, ks_p_is = stats.kstest(z_data, "beta", args=(15.0, 6.0))
                median_is = float(np.median(I_s))
                in_range_is = 19.0 <= median_is <= 20.5
                stats_out.append({
                    "parameter": "Source Baseline Magnitude I_s (mag)",
                    "reference": "TdR Image 10 (p.28): OGLE-IV, Beta(15,6) scaled to [14,22], mode ~19.85 mag",
                    "observed": f"KS statistic = {ks_stat_is:.4f} (p = {ks_p_is:.3g}), median = {median_is:.2f} mag",
                    "expected": "Distribution peaking near 19.85 mag, range [14, 22] mag",
                    "status": "OK" if in_range_is else "CHECK",
                })

        fig.tight_layout()
        common_img = _fig_to_b64(fig)

    # ── Lens velocity (separate figure) ────────────────────────────────────
    velocity_img = None
    if "v_perp_kms" in data and len(data["v_perp_kms"]) > 1:
        v_perp = data["v_perp_kms"]
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.hist(v_perp, bins=50, density=True, alpha=0.6, color="salmon",
                 edgecolor="white", label="Data")
        x_vel = np.linspace(0, v_perp.max(), 300)
        mb_scale = 200.0 / np.sqrt(2.0)
        mb_pdf = stats.maxwell.pdf(x_vel, scale=mb_scale)
        ax2.plot(x_vel, mb_pdf, "darkred", lw=2,
                 label="Maxwell-Boltzmann (mode=200 km/s) [TdR p.24]")
        ax2.set_xlabel("Transversal Velocity v_perp (km/s)")
        ax2.set_ylabel("Probability Density")
        ax2.set_title("Lens Velocity v_perp [TdR p.24, Rahvar 2015]")
        ax2.legend()
        fig2.tight_layout()
        velocity_img = _fig_to_b64(fig2)

        ks_stat_v, ks_p_v = stats.kstest(v_perp, "maxwell", args=(0, mb_scale))
        # Judge on the KS statistic (effect size), not the p-value -- consistent with
        # every other check here. For a correctly sampled Maxwell the p-value is
        # uniform on [0,1], so a p<0.01 rule raises a false alarm ~1% of the time no
        # matter how good the generator is; and as n grows it collapses to 0 for any
        # negligible deviation. The statistic measures what actually matters: how far
        # apart the two cumulative distributions are.
        stats_out.append({
            "parameter": "Lens Velocity v_perp (km/s)",
            "reference": "TdR p.24 (Rahvar 2015): Maxwell-Boltzmann distribution, mode = 200 km/s",
            "observed": f"KS statistic = {ks_stat_v:.4f} (p = {ks_p_v:.3g}), median = {np.median(v_perp):.1f} km/s",
            "expected": "Maxwell-Boltzmann shape with mode near 200 km/s (KS stat < 0.05)",
            "status": "OK" if ks_stat_v < 0.05 else "CHECK",
        })

    # ── Binary parameters ───────────────────────────────────────────────────
    binary_img = None
    if data.get("n_binary", 0) > 0:
        binary_panels = []
        if "q_binary" in data and len(data["q_binary"]) > 1:
            binary_panels.append("q")
        if "a_pc_binary" in data and len(data["a_pc_binary"]) > 1:
            binary_panels.append("a")
        if "e_binary" in data and len(data["e_binary"]) > 1:
            binary_panels.append("e")
        if "alpha_ref_binary" in data and len(data["alpha_ref_binary"]) > 1:
            binary_panels.append("alpha")

        if binary_panels:
            n_b = len(binary_panels)
            ncols_b = min(2, n_b)
            nrows_b = (n_b + 1) // 2
            fig3, axes3 = plt.subplots(nrows_b, ncols_b, figsize=(11, 4.5 * nrows_b),
                                        squeeze=False)
            fig3.suptitle(
                f"Validation of Binary-Lens Parameters (n = {data['n_binary']:,})",
                fontsize=15, fontweight="bold",
            )
            ax_list3 = [axes3[r][c] for r in range(nrows_b) for c in range(ncols_b)]
            for ax in ax_list3[n_b:]:
                ax.axis("off")

            idx3 = 0

            if "q" in binary_panels:
                q_bin = data["q_binary"]
                ax = ax_list3[idx3]; idx3 += 1
                log10_q = np.log10(q_bin)
                ax.hist(log10_q, bins=40, density=True, alpha=0.6, color="mediumpurple",
                        edgecolor="white", label="Data")
                ax.axvline(np.log10(1.43e-3), color="red", ls="--", lw=1.5,
                           label="TdR Image 3 peak (q~1.43e-3)")
                ax.set_xlabel("log10(q)")
                ax.set_ylabel("Probability Density")
                ax.set_title("Mass Ratio q [Image 3, p.21]")
                ax.legend()
                median_q = float(np.median(q_bin))
                in_range = 1e-4 <= median_q <= 1e-2
                stats_out.append({
                    "parameter": "Mass Ratio q",
                    "reference": "TdR Image 3 (p.21): log-normal, median ~1.43e-3 (NASA Exoplanet Archive)",
                    "observed": f"median = {median_q:.2e}",
                    "expected": "median in [1e-4, 1e-2]",
                    "status": "OK" if in_range else "CHECK",
                })

            if "a" in binary_panels:
                a_bin = data["a_pc_binary"]
                ax = ax_list3[idx3]; idx3 += 1
                log10_a = np.log10(a_bin)
                ax.hist(log10_a, bins=40, density=True, alpha=0.6, color="goldenrod",
                        edgecolor="white", label="Data")
                ax.axvline(-6.4, color="red", ls="--", lw=1.5,
                           label="Dominant peak ~1e-6.4 pc [Image 8]")
                ax.axvline(-5.0, color="blue", ls="--", lw=1.5,
                           label="Secondary bump ~1e-5 pc [Image 8]")
                ax.set_xlabel("log10(a) [pc]")
                ax.set_ylabel("Probability Density")
                ax.set_title("Semi-major Axis a [Image 8, p.26]")
                ax.legend()
                median_a = float(np.median(a_bin))
                in_range = -7.5 <= np.log10(median_a) <= -5.0
                stats_out.append({
                    "parameter": "Semi-major Axis a (pc)",
                    "reference": "TdR Image 8 (p.26): bimodal in log-space, dominant peak ~1e-6.4 pc, secondary ~1e-5 pc",
                    "observed": f"median = {median_a:.2e} pc",
                    "expected": "median in [10^-7.5, 10^-5] pc",
                    "status": "OK" if in_range else "CHECK",
                })

            if "e" in binary_panels:
                e_bin = data["e_binary"]
                ax = ax_list3[idx3]; idx3 += 1
                ax.hist(e_bin, bins=40, density=True, alpha=0.6, color="cornflowerblue",
                        edgecolor="white", label="Data")
                x_ecc = np.linspace(0, 0.99, 300)
                beta_pdf = stats.beta.pdf(x_ecc, 1.5, 12.0)
                ax.plot(x_ecc, beta_pdf, "darkblue", lw=2, label="Beta(1.5, 12) [TdR Image 9]")
                ax.set_xlabel("Orbital Eccentricity e")
                ax.set_ylabel("Probability Density")
                ax.set_title("Eccentricity e [Image 9, p.26-27]")
                ax.legend()
                ks_stat_e, ks_p_e = stats.kstest(e_bin, "beta", args=(1.5, 12.0))
                stats_out.append({
                    "parameter": "Orbital Eccentricity e",
                    "reference": "TdR Image 9 (p.26-27): exponential-like decay, mode ~0.043, modeled as Beta(1.5, 12)",
                    "observed": f"KS statistic = {ks_stat_e:.4f} (p = {ks_p_e:.3g}), median = {np.median(e_bin):.3f}",
                    "expected": "Distribution concentrated near 0 with long tail toward 1",
                    "status": "OK" if ks_p_e > 0.01 else "CHECK",
                })

            if "alpha" in binary_panels:
                alpha_bin = data["alpha_ref_binary"]
                ax = ax_list3[idx3]; idx3 += 1
                ax.hist(alpha_bin, bins=40, density=True, alpha=0.6, color="lightcoral",
                        edgecolor="white", label="Data")
                ax.axhline(1.0 / (2 * np.pi), color="red", ls="--", lw=1.5,
                           label="Uniform(0, 2*pi) [TdR p.28]")
                ax.set_xlabel("Trajectory Angle alpha_ref (rad)")
                ax.set_ylabel("Probability Density")
                ax.set_title("Trajectory Angle alpha_ref [p.28]")
                ax.legend()
                ks_stat_a, ks_p_a = stats.kstest(alpha_bin, "uniform", args=(0, 2 * np.pi))
                stats_out.append({
                    "parameter": "Trajectory Angle alpha_ref (rad)",
                    "reference": "TdR p.28: theoretically completely random -> uniform(0, 2*pi)",
                    "observed": f"KS statistic = {ks_stat_a:.4f} (p = {ks_p_a:.3g})",
                    "expected": "Flat (uniform) distribution over [0, 2*pi]",
                    "status": "OK" if ks_p_a > 0.01 else "CHECK",
                })

            fig3.tight_layout()
            binary_img = _fig_to_b64(fig3)

    return common_img, velocity_img, binary_img, stats_out


def plot_ogle_validation(data):
    """Validate OGLE-IV noise and cadence imperfections.

    Panel A — noise level σ(I) distribution: compares the distribution of noise
    levels applied at observed time points vs the OGLE-IV reference (weighted by
    the empirical magnitude-bin counts from noise_model.npz).

    Panel B — cadence coverage: histogram of the fraction of time points that
    are non-NaN per event, showing how much of each light curve was "observed."

    Panel C — blend fraction f_s: generated vs the OGLE-IV catalogue it is
    bootstrapped from.

    Panel D — observed baseline I_base = I_s + 2.5*log10(f_s): an INDEPENDENT
    cross-check. I_s and f_s come from the fitted catalogue, while the reference
    here is measured straight from the raw photometry (baseline_model.npz). The
    two data products are unrelated, so agreement means the blending model really
    does reproduce the brightnesses OGLE records.

    Returns (b64_str, stats_list) or (None, []) if model files are missing.
    """
    try:
        nm = np.load(_NOISE_DIR / "noise_model.npz")
        cm = np.load(_NOISE_DIR / "cadence_model.npz")
    except FileNotFoundError:
        return None, []

    try:
        bm = np.load(_NOISE_DIR / "blend_model.npz")
        fs_ref, Is_ref = bm["fs"], bm["Is"]
    except (FileNotFoundError, KeyError):
        fs_ref, Is_ref = None, None

    try:
        base_ref = np.load(_NOISE_DIR / "baseline_model.npz")["baselines"]
    except (FileNotFoundError, KeyError):
        base_ref = None

    fit_params  = nm["fit_params"]
    I_ref_val   = float(nm["I_ref"][0])
    sig_floor   = float(fit_params[0])
    sig_phot0   = float(fit_params[1])
    bin_centers = nm["bin_centers"]
    n_per_bin   = nm["n_per_bin"].astype(float)

    def _sig(I):
        return np.sqrt(sig_floor ** 2 + sig_phot0 ** 2 * 10.0 ** (0.4 * (I - I_ref_val)))

    # Blending panels only exist if the dataset actually carries f_s.
    f_s_gen = data.get("f_s_blend")
    I_s_gen = data.get("I_s_mag")
    show_blend = f_s_gen is not None and fs_ref is not None

    if show_blend:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax3 = ax4 = None
    fig.suptitle("OGLE-IV Imperfections Validation", fontsize=15, fontweight="bold")

    # ── Panel A: σ(I) distribution — generated vs OGLE-IV reference ──────────
    time_cols = [c for c in data["df"].columns if c.startswith("t_") and c[2:].isdigit()]
    time_cols = sorted(time_cols, key=lambda c: int(c[2:]))
    lc_mat    = data["df"][time_cols].values
    obs_mask  = ~np.isnan(lc_mat)   # full mask (Panel B cadence coverage uses this)

    # The synthetic curves densely sample the magnified peak, which the
    # baseline-dominated OGLE reference does not. Pool only near-baseline points
    # (|tau| > 2) so the noise-level comparison is like-with-like.
    tau       = np.linspace(-3.0, 3.0, lc_mat.shape[1])
    base_cols = np.abs(tau) > 2.0
    lc_base   = lc_mat[:, base_cols]
    I_obs_all = lc_base[~np.isnan(lc_base)]
    rng_p     = np.random.default_rng(0)
    n_s       = min(len(I_obs_all), 100_000)
    I_obs     = rng_p.choice(I_obs_all, size=n_s, replace=False)
    sigma_gen = _sig(np.clip(I_obs, 12.0, 22.5))

    # Reference magnitudes. The generated dataset is a population of EVENTS (each
    # contributes equally), so the fair reference is the per-event baseline
    # distribution, not the observation-weighted pool -- the latter over-counts
    # heavily-monitored events and sits ~0.2 mag bright.
    if base_ref is not None:
        I_ref_s  = rng_p.choice(base_ref, size=50_000, replace=True)
        ref_label = "OGLE-IV reference (per-event baselines)"
    else:
        probs    = n_per_bin / n_per_bin.sum()
        I_ref_s  = rng_p.choice(bin_centers, size=50_000, p=probs)
        ref_label = "OGLE-IV reference (observation-weighted)"
    sigma_ref = _sig(np.clip(I_ref_s, 12.0, 22.5))

    log_bins = np.logspace(np.log10(0.001), np.log10(1.0), 80)
    ax1.hist(sigma_ref, bins=log_bins, density=True, alpha=0.40, color="orange",
             label=ref_label)
    ax1.hist(sigma_gen, bins=log_bins, density=True, alpha=0.55, color="#2563eb",
             label=f"Generated ({n_s:,} baseline pts, |tau|>2)")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\sigma_I$ [mag]")
    ax1.set_ylabel("Probability Density")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    n_ks_a    = min(n_s, 10_000)
    ks_a, p_a = stats.ks_2samp(
        rng_p.choice(sigma_gen, size=n_ks_a, replace=True),
        rng_p.choice(sigma_ref, size=n_ks_a, replace=True),
    )
    # Judge by the KS statistic (effect size), not the p-value: at ~10k samples
    # the p-value is ~0 for any real distribution pair, so it can never pass.
    # KS < 0.10 means the CDFs differ by < 10% -- a good distributional match.
    status_a = "OK" if ks_a < 0.10 else "CHECK"
    ax1.set_title(
        f"Noise Level Distribution\nKS stat={ks_a:.3f}, p={p_a:.3g}  [{status_a}]",
        fontsize=11,
    )

    # ── Panel B: cadence coverage per event ───────────────────────────────────
    obs_fracs = obs_mask.mean(axis=1)
    mean_frac = float(np.mean(obs_fracs))
    med_frac  = float(np.median(obs_fracs))

    ax2.hist(obs_fracs, bins=50, density=True, color="#2563eb", alpha=0.70,
             edgecolor="white", label=f"Per-event coverage (n={len(obs_fracs):,})")
    ax2.axvline(mean_frac, color="red",    ls="--", lw=1.5,
                label=f"Mean: {mean_frac:.1%}")
    ax2.axvline(med_frac,  color="orange", ls=":",  lw=1.5,
                label=f"Median: {med_frac:.1%}")
    ax2.set_xlabel("Observation fraction (non-NaN / total time points)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Cadence Coverage per Event", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    blend_stats = []
    if show_blend:
        f_s_gen = np.asarray(f_s_gen, dtype=float)

        # ── Panel C: blend fraction f_s — generated vs catalogue ──────────────
        fs_bins = np.linspace(0.0, 2.0, 60)
        ax3.hist(fs_ref, bins=fs_bins, density=True, alpha=0.40, color="orange",
                 label=f"OGLE-IV catalogue (n={len(fs_ref):,})")
        ax3.hist(f_s_gen, bins=fs_bins, density=True, alpha=0.55, color="#2563eb",
                 label=f"Generated (n={len(f_s_gen):,})")
        ax3.axvline(1.0, color="red", ls="--", lw=1.2, alpha=0.7)
        ax3.set_xlabel(r"Blend fraction  $f_s = F_{source}/F_{baseline}$")
        ax3.set_ylabel("Probability Density")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        rng_b = np.random.default_rng(0)
        n_fs = min(len(f_s_gen), 10_000)
        ks_fs, p_fs = stats.ks_2samp(
            rng_b.choice(f_s_gen, size=n_fs, replace=True), fs_ref
        )
        status_fs = "OK" if ks_fs < 0.05 else "CHECK"
        ax3.set_title(
            f"Blend Fraction f_s\nKS stat={ks_fs:.3f}  [{status_fs}]  median={np.median(f_s_gen):.3f}",
            fontsize=11,
        )
        blend_stats.append({
            "parameter": "Blend fraction f_s",
            "reference": (
                f"OGLE-IV event catalogue (fs_med), median {float(np.median(fs_ref)):.3f}; "
                "drawn PAIRED with I_s so their correlation is preserved"
            ),
            "observed": f"KS stat={ks_fs:.4f} vs catalogue, median={float(np.median(f_s_gen)):.3f}",
            "expected": "Bootstrapped from the catalogue (KS stat < 0.05)",
            "status": status_fs,
        })

        # ── Panel D: observed baseline I_base — INDEPENDENT cross-check ───────
        # I_base is derived from the fitted catalogue (I_s, f_s); the reference is
        # measured from the raw photometry. Two unrelated OGLE data products.
        if I_s_gen is not None and base_ref is not None:
            I_base_gen = np.asarray(I_s_gen, float) + 2.5 * np.log10(
                np.maximum(f_s_gen, 1e-10)
            )
            b_bins = np.linspace(12.0, 22.5, 60)
            ax4.hist(base_ref, bins=b_bins, density=True, alpha=0.40, color="orange",
                     label=f"OGLE-IV photometry (n={len(base_ref):,})")
            ax4.hist(I_base_gen, bins=b_bins, density=True, alpha=0.55, color="#2563eb",
                     label=f"Generated (n={len(I_base_gen):,})")
            ax4.axvline(float(np.median(I_base_gen)), color="navy", ls="--", lw=1.5,
                        label=f"Median: {float(np.median(I_base_gen)):.2f} mag")
            ax4.set_xlabel(r"Observed baseline  $I_{base} = I_s + 2.5\log_{10} f_s$  [mag]")
            ax4.set_ylabel("Probability Density")
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)

            n_bs = min(len(I_base_gen), 10_000)
            ks_b, p_b = stats.ks_2samp(
                rng_b.choice(I_base_gen, size=n_bs, replace=True), base_ref
            )
            d_med = float(np.median(I_base_gen)) - float(np.median(base_ref))
            # Loose threshold: this is a cross-check between two INDEPENDENT data
            # products (fitted catalogue vs raw photometry), not a bootstrap of one.
            status_b = "OK" if abs(d_med) <= 0.30 else "CHECK"
            ax4.set_title(
                f"Observed Baseline I_base — independent cross-check\n"
                f"KS stat={ks_b:.3f}, offset={d_med:+.2f} mag  [{status_b}]",
                fontsize=11,
            )
            blend_stats.append({
                "parameter": "Observed baseline I_base (cross-check)",
                "reference": (
                    f"OGLE-IV raw photometry, per-event baselines (baseline_model.npz), "
                    f"median {float(np.median(base_ref)):.2f} mag"
                ),
                "observed": (
                    f"median={float(np.median(I_base_gen)):.2f} mag "
                    f"(offset {d_med:+.2f}), KS stat={ks_b:.4f}"
                ),
                "expected": (
                    "I_base derived from the fitted catalogue should reproduce the "
                    "independently measured photometric baseline (offset < 0.30 mag)"
                ),
                "status": status_b,
            })
        elif ax4 is not None:
            ax4.axis("off")

    fig.tight_layout()
    b64 = _fig_to_b64(fig)

    stats_entries = [
        {
            "parameter": "Noise level sigma_I distribution",
            "reference": (
                f"OGLE-IV empirical noise (3 000 EWS events): "
                f"sigma_floor={sig_floor:.5f} mag, sigma_phot0={sig_phot0:.5f} mag at I=18"
            ),
            "observed": f"KS stat={ks_a:.4f} vs OGLE reference, {n_s:,} baseline pts (|tau|>2)",
            "expected": "Noise level distribution matches OGLE-IV reference (KS stat < 0.10)",
            "status": status_a,
        },
        {
            "parameter": "Cadence coverage fraction",
            "reference": "OGLE-IV within-season cadence: 76% intra-night (<0.5 d), 24% night-to-night",
            "observed": f"Mean observation fraction: {mean_frac:.1%},  median: {med_frac:.1%}",
            "expected": "Fraction < 100% (cadence gaps present in each light curve)",
            "status": "OK" if mean_frac < 0.99 else "CHECK",
        },
    ] + blend_stats

    return b64, stats_entries


def plot_distributions_ogle(data):
    """Informational figure shown after generation when ogle_noise=True.

    Panel A — the σ(I) noise model that was applied, overlaid on OGLE-IV reference.
    Panel B — the cadence Δt distribution that was bootstrap-resampled per event.
    Panel C — the blend fraction f_s applied, vs the OGLE-IV catalogue (if present).
    """
    try:
        nm = np.load(_NOISE_DIR / "noise_model.npz")
        cm = np.load(_NOISE_DIR / "cadence_model.npz")
    except FileNotFoundError:
        return None

    try:
        fs_ref = np.load(_NOISE_DIR / "blend_model.npz")["fs"]
    except (FileNotFoundError, KeyError):
        fs_ref = None

    f_s_gen = data.get("f_s_blend")
    show_blend = f_s_gen is not None and fs_ref is not None

    if show_blend:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax3 = None
    fig.suptitle(
        "OGLE-IV Imperfections Applied to Dataset",
        fontsize=15, fontweight="bold",
    )

    # ── Panel A: noise model σ(I) ─────────────────────────────────────────────
    I_curve     = nm["I_curve"]
    sigma_curve = nm["sigma_curve"]
    bin_centers = nm["bin_centers"]
    med_sigma   = nm["med_sigma"]
    pct16       = nm["pct16_sigma"]
    pct84       = nm["pct84_sigma"]
    fp          = nm["fit_params"]

    ax1.fill_between(bin_centers, pct16, pct84, alpha=0.25, color="orange",
                     label="16th–84th pct (OGLE-IV reference)")
    ax1.plot(bin_centers, med_sigma, "o", ms=4, color="orange", zorder=3,
             label="Median per bin (OGLE-IV reference)")
    ax1.plot(I_curve, sigma_curve, "r-", lw=2.5,
             label=fr"Applied: $\sqrt{{{fp[0]:.4f}^2+{fp[1]:.4f}^2\cdot10^{{0.4(I-18)}}}}$")
    ax1.set_yscale("log")
    ax1.set_xlim(12, 22.5)
    ax1.set_ylim(0.001, 1.0)
    ax1.set_xlabel("I-band magnitude [mag]")
    ax1.set_ylabel(r"$\sigma_I$ [mag]")
    ax1.set_title("Photometric Noise Model Applied\n(fitted to 3 000 OGLE-IV EWS events)")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # ── Panel B: cadence Δt distribution ─────────────────────────────────────
    dt_ref   = cm["dt_inseason"]
    st       = cm["stats"]
    med_dt   = float(st[0])
    vis_frac = float(st[6])

    dt_plot  = dt_ref[(dt_ref > 0.005) & (dt_ref < 100)]
    log_bins = np.logspace(np.log10(0.005), np.log10(100), 80)
    ax2.hist(dt_plot, bins=log_bins, density=True, color="#2563eb", alpha=0.60,
             label=f"OGLE-IV within-season Δt (n={len(dt_plot):,})")
    ax2.axvline(med_dt, color="red", ls="--", lw=1.5,
                label=f"Median: {med_dt * 24 * 60:.0f} min")
    ax2.set_xscale("log")
    ticks = [0.01, 0.1, 0.5, 1, 7, 30, 100]
    tlabs = ["15 min", "2.4 h", "0.5 d", "1 d", "1 wk", "1 mo", "100 d"]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tlabs, fontsize=8)
    ax2.set_xlim(0.005, 100)
    ax2.set_xlabel(r"$\Delta t$ between consecutive observations [days]")
    ax2.set_ylabel("Probability Density")
    ax2.set_title(
        f"Cadence Distribution Applied\n"
        f"(Galactic Bulge visible {vis_frac:.0%}/yr · bootstrap-resampled per event)"
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.25)

    # ── Panel C: blend fraction f_s applied ──────────────────────────────────
    if show_blend:
        f_s_gen = np.asarray(f_s_gen, dtype=float)
        fs_bins = np.linspace(0.0, 2.0, 60)
        ax3.hist(fs_ref, bins=fs_bins, density=True, alpha=0.40, color="orange",
                 label=f"OGLE-IV catalogue (n={len(fs_ref):,})")
        ax3.hist(f_s_gen, bins=fs_bins, density=True, alpha=0.60, color="#2563eb",
                 label=f"Applied to dataset (n={len(f_s_gen):,})")
        ax3.axvline(float(np.median(f_s_gen)), color="red", ls="--", lw=1.5,
                    label=f"Median: {float(np.median(f_s_gen)):.3f}")
        # f_s > 1 is not physical; it survives in the catalogue as fit scatter on
        # nearly-unblended events, and is kept so the distribution stays unbiased.
        ax3.axvspan(1.0, 2.0, color="red", alpha=0.06, zorder=0)
        ax3.set_xlim(0, 2)
        ax3.set_xlabel(r"Blend fraction  $f_s = F_{source}/F_{baseline}$")
        ax3.set_ylabel("Probability Density")
        ax3.set_title(
            "Blending Applied\n"
            "(drawn paired with I_s from the OGLE-IV catalogue)"
        )
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    return _fig_to_b64(fig)