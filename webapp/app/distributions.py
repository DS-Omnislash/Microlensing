"""Parameter sampling distributions for synthetic microlensing events.

All distributions are documented in ``TdR_RocRC.pdf`` (Roc Rubio,
"Gravitational Microlensing"), section "Parameters distributions" (pp. 20-29).
"""

import numpy as np
from scipy import stats

# --- Physical constants (SI units, float64 precision) ---
G = np.float64(6.67430e-11)            # Gravitational constant [m^3 kg^-1 s^-2]
C_LIGHT = np.float64(2.99792458e8)     # Speed of light [m/s]
M_SUN = np.float64(1.98892e30)         # Solar mass [kg]
M_JUP = np.float64(1.89813e27)         # Jupiter mass [kg]
PC_TO_M = np.float64(3.0856775814913673e16)  # 1 parsec in meters
KM_TO_M = np.float64(1.0e3)            # 1 km in meters
AU_TO_PC = np.float64(np.pi / 648_000.0)  # 1 AU in parsecs (IAU 2015 Res. B2)


def sample_lens_mass(n, rng):
    """Lens mass M_star [Msun] - bimodal Gaussian mixture (TdR Image 2, p.21).

    Component 1: low-mass peak ~0.10 Msun (M-dwarfs, ~25% weight)
    Component 2: main peak ~0.45 Msun (K/G-dwarfs, ~75% weight)
    Clipped to the physical range [0.01, 1.2] Msun.
    """
    w1, mu1, sig1 = 0.25, 0.10, 0.06
    w2, mu2, sig2 = 0.75, 0.45, 0.22

    component = rng.choice([0, 1], size=n, p=[w1, w2])
    masses = np.empty(n, dtype=np.float64)

    mask1 = component == 0
    mask2 = component == 1
    masses[mask1] = rng.normal(mu1, sig1, size=mask1.sum())
    masses[mask2] = rng.normal(mu2, sig2, size=mask2.sum())

    return np.clip(masses, 0.01, 1.2).astype(np.float64)


def sample_distance_to_lens(n, rng):
    """Distance to lens D_l [pc] - bulge-peaked mixture (TdR Image 5, p.24).

    Component 1: Galactic disk lenses (~20%), broad, centered ~1500 pc.
    Component 2: Galactic bulge lenses (~80%), peaked at ~6800 pc.
    Clipped to [300, 8500] pc.
    """
    w_disk, mu_disk, sig_disk = 0.20, 1500.0, 800.0
    w_bulge, mu_bulge, sig_bulge = 0.80, 6800.0, 750.0

    component = rng.choice([0, 1], size=n, p=[w_disk, w_bulge])
    distances = np.empty(n, dtype=np.float64)

    mask_disk = component == 0
    mask_bulge = component == 1
    distances[mask_disk] = rng.normal(mu_disk, sig_disk, size=mask_disk.sum())
    distances[mask_bulge] = rng.normal(mu_bulge, sig_bulge, size=mask_bulge.sum())

    return np.clip(distances, 300.0, 8500.0).astype(np.float64)


def sample_lens_source_distance(n, rng):
    """Lens-source distance D_ls [pc] - structured uniform (TdR Image 4, p.23).

    Base: uniform [100, 8000] pc (~70%)
    Peak 1: Gaussian at ~1000 pc, sigma=500 (~15%)
    Peak 2: Gaussian at ~5000 pc, sigma=800 (~15%)
    Clipped to [100, 8000] pc.
    """
    w_unif = 0.70
    w_peak1, mu1, sig1 = 0.15, 1000.0, 500.0
    w_peak2, mu2, sig2 = 0.15, 5000.0, 800.0

    component = rng.choice([0, 1, 2], size=n, p=[w_unif, w_peak1, w_peak2])
    distances = np.empty(n, dtype=np.float64)

    m0 = component == 0
    m1 = component == 1
    m2 = component == 2
    distances[m0] = rng.uniform(100.0, 8000.0, size=m0.sum())
    distances[m1] = rng.normal(mu1, sig1, size=m1.sum())
    distances[m2] = rng.normal(mu2, sig2, size=m2.sum())

    return np.clip(distances, 100.0, 8000.0).astype(np.float64)


def sample_lens_velocity(n, rng):
    """Transversal lens velocity v_perp [km/s] - Maxwell-Boltzmann (TdR p.25).

    Maxwell-Boltzmann distribution with mode = 200 km/s, i.e.
    sigma = 200 / sqrt(2) ~= 141.42 km/s.
    """
    mode_kms = np.float64(200.0)
    sigma = mode_kms / np.sqrt(np.float64(2.0))
    velocities = stats.maxwell.rvs(scale=sigma, size=n, random_state=rng.integers(2**31))
    return velocities.astype(np.float64)


def sample_impact_parameter(n, rng):
    """Impact parameter u0 - truncated exponential (TdR Image 7, p.26).

    Models the observed OGLE-IV telescope bias: a strong concentration near
    u0=0 decaying roughly exponentially toward u0=1 (decay rate lambda=3),
    rather than the theoretical uniform(0,1) expectation.
    """
    lam = np.float64(3.0)
    u_max = 1.0 - np.exp(-lam * 1.0)
    u_uniform = rng.uniform(0.0, u_max, size=n)
    u0 = -np.log(1.0 - u_uniform) / lam
    return np.clip(u0, 1e-6, 1.0).astype(np.float64)


def sample_mass_ratio(n, rng):
    """Mass ratio q = m_p / m_star - log-normal (TdR Image 3, p.22).

    q is the ratio of the two bodies' fractional masses,
    m_i = M_i / (M_p + M_star), so it reduces to q = M_planet / M_star --
    exactly the quantity ``distributions/q_mass_ratio.py`` extracts from the
    NASA Exoplanet Archive. Used directly as the companion mass m2 (in units
    of the primary) by the binary-lens solver.

    Binary-only. log10(q) ~ Normal(mu=-2.845, sigma=1.0), peak ~1.43e-3.
    """
    mu_log10 = np.float64(np.log10(1.43e-3))
    sigma_log10 = np.float64(1.0)
    log10_q = rng.normal(mu_log10, sigma_log10, size=n)
    q = np.power(np.float64(10.0), log10_q)
    return np.clip(q, 1e-6, 0.999).astype(np.float64)


# Semi-major axes [AU] of every planet in the NASA Exoplanet Archive that was
# actually DISCOVERED BY MICROLENSING (`ps` table, default_flag = 1,
# discoverymethod = 'Microlensing'; n = 274, retrieved 2026-07-14).
# Regenerate with distributions/a_pc_semi_major_axis.py, which prints this array.
#
# These are the real measured values; sample_semi_major_axis() resamples them
# directly rather than fitting an analytic form to them. See that function.
A_MICROLENSING_AU = np.array([
    0.24, 0.342, 0.39, 0.43, 0.48, 0.59, 0.62, 0.63, 0.681, 0.69, 0.702, 0.72, 0.73, 0.74,
    0.763, 0.78, 0.79, 0.8, 0.8, 0.83, 0.85, 0.85, 0.87, 0.87, 0.89, 0.9, 0.9, 0.92, 0.93,
    0.93, 0.94, 0.95, 0.95, 0.96, 0.97, 1.02, 1.064, 1.07, 1.08, 1.1, 1.1, 1.1, 1.11, 1.12,
    1.14, 1.15, 1.16, 1.16, 1.16, 1.18, 1.18, 1.21, 1.23, 1.26, 1.28, 1.305, 1.32, 1.36,
    1.38, 1.38, 1.39, 1.4, 1.41, 1.449, 1.5, 1.52, 1.54, 1.54, 1.6, 1.6, 1.6, 1.6, 1.62,
    1.63, 1.64, 1.68, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.72, 1.75, 1.75, 1.75, 1.78, 1.79, 1.8,
    1.8, 1.8, 1.8, 1.8, 1.8, 1.81, 1.84, 1.85, 1.85, 1.88, 1.887, 1.89, 1.9, 1.9, 1.9, 1.96,
    2, 2, 2, 2, 2.02, 2.02, 2.03, 2.03, 2.07, 2.07, 2.07, 2.09, 2.13, 2.14, 2.14, 2.15,
    2.15, 2.15, 2.17, 2.17, 2.18, 2.18, 2.2, 2.2, 2.22, 2.29, 2.29, 2.3, 2.3, 2.31, 2.34,
    2.36, 2.37, 2.41, 2.42, 2.424, 2.43, 2.44, 2.44, 2.46, 2.5, 2.5, 2.5, 2.51, 2.53, 2.55,
    2.59, 2.6, 2.6, 2.6, 2.6, 2.6, 2.62, 2.63, 2.63, 2.65, 2.68, 2.7, 2.7, 2.7, 2.72, 2.72,
    2.73, 2.73, 2.74, 2.77, 2.8, 2.8, 2.8, 2.8, 2.86, 2.86, 2.89, 2.92, 2.96, 2.96, 3, 3.02,
    3.02, 3.03, 3.03, 3.04, 3.05, 3.09, 3.1, 3.14, 3.14, 3.15, 3.2, 3.2, 3.2, 3.2, 3.23,
    3.29, 3.3, 3.31, 3.32, 3.4, 3.4, 3.44, 3.45, 3.45, 3.48, 3.49, 3.5, 3.5, 3.5, 3.5, 3.51,
    3.54, 3.54, 3.58, 3.59, 3.6, 3.6, 3.64, 3.7, 3.76, 3.79, 3.81, 3.9, 3.913, 3.92, 3.97,
    4, 4, 4.01, 4.06, 4.06, 4.14, 4.16, 4.17, 4.18, 4.27, 4.27, 4.3, 4.3, 4.39, 4.43, 4.5,
    4.5, 4.5, 4.51, 4.53, 4.8, 4.81, 5, 5.1, 5.19, 5.25, 5.4, 5.67, 5.67, 5.75, 5.9, 6.03,
    6.4, 6.46, 6.85, 6.99, 7.78, 8, 8.3, 10.1, 10.2, 12.6, 15, 15.91, 22.3,
], dtype=np.float64)

# Jitter applied to log10(a) when resampling, in dex. Chosen at 0.05, deliberately
# below Silverman's rule (0.073): Silverman is tuned for smooth density *estimation*
# and broadens the peak, whereas the real distribution is sharply peaked (excess
# kurtosis +0.85) and we want to preserve that shape.
_A_JITTER_DEX = 0.05


def sample_semi_major_axis(n, rng):
    """Planet semi-major axis a [pc] - resampled from the REAL microlensing planets.

    Binary-only. Rather than fitting an analytic form, this draws directly from the
    274 measured semi-major axes of microlensing-discovered planets
    (``A_MICROLENSING_AU``), adding a small Gaussian jitter in log10-space so the
    result is continuous instead of 274 discrete atoms. This is the same "bootstrap
    the real catalogue" approach already used for the (I_s, f_s) blending pairs.

    No analytic family was used because none fits: the real distribution is
    noticeably more peaked than a log-normal (excess kurtosis +0.85). Measured
    against the 274 real values, this sampler gives KS = 0.024, versus KS = 0.060
    for the best-fitting log-normal.

    NOTE: the archive as a whole must NOT be used here. Filtering only on
    default_flag=1 yields ~3 900 planets dominated by transit and radial-velocity
    discoveries, whose sensitivity peaks at a < 1 AU (median 0.12 AU). Microlensing
    is blind to those and is sensitive at 1-10 AU (median 2.37 AU). Using the
    unfiltered archive imports the transit selection function into a microlensing
    simulator: it drove the companion separation to d = a/r_E ~ 0.08 (vs ~1 for real
    events, since r_E ~ 2 AU) and shortened the orbital period so much that the
    companion completed ~2 full revolutions per event instead of being quasi-static.
    """
    log10_a_au = np.log10(rng.choice(A_MICROLENSING_AU, size=n))
    log10_a_au = log10_a_au + rng.normal(0.0, _A_JITTER_DEX, size=n)

    a_pc = np.power(np.float64(10.0), log10_a_au) * AU_TO_PC
    return np.clip(a_pc, 1e-8, 0.1).astype(np.float64)


def sample_eccentricity(n, rng):
    """Orbital eccentricity e - Beta distribution (TdR Image 9, p.27-28).

    Binary-only. Beta(alpha=1.5, beta=12), mode = (alpha-1)/(alpha+beta-2) ~= 0.043,
    matching the exponential-like decay observed from 0 to 1.
    """
    e = rng.beta(1.5, 12.0, size=n)
    return np.clip(e, 0.0, 0.99).astype(np.float64)


def sample_trajectory_angle(n, rng):
    """Trajectory angle alpha_ref [rad] - uniform (TdR p.29).

    Binary-only. "Theoretically completely random" -> uniform on [0, 2*pi].
    """
    return rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float64)


def sample_baseline_magnitude(n, rng):
    """Source baseline I-band magnitude I_s [mag] - scaled Beta (OGLE-IV, Mroz 2019).

    Beta(15, 6) scaled to [14, 22]: mode = 14 + 8*(15-1)/(15+6-2) = 19.89 mag.
    Matches the observed OGLE-IV distribution: left-skewed with a long tail toward
    bright (low-magnitude) stars and a steep drop after the peak at ~19.9 mag.
    """
    MAG_MIN, MAG_MAX = np.float64(14.0), np.float64(22.0)
    beta_samples = rng.beta(15.0, 6.0, size=n)
    return (MAG_MIN + beta_samples * (MAG_MAX - MAG_MIN)).astype(np.float64)
