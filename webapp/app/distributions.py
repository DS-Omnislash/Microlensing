"""Parameter sampling distributions for synthetic microlensing events.

All distributions are reproduced from ``Dataset1generator.ipynb`` and are
documented in ``TDR_ROC.pdf`` (Roc Rubio, "Gravitational Microlensing"),
section "Parameters distributions" (pp. 20-29).
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
    """Mass ratio q = M_p / (M_star + M_p) - log-normal (TdR Image 3, p.22).

    Binary-only. log10(q) ~ Normal(mu=-2.845, sigma=1.0), peak ~1.43e-3.
    """
    mu_log10 = np.float64(np.log10(1.43e-3))
    sigma_log10 = np.float64(1.0)
    log10_q = rng.normal(mu_log10, sigma_log10, size=n)
    q = np.power(np.float64(10.0), log10_q)
    return np.clip(q, 1e-6, 0.999).astype(np.float64)


def sample_semi_major_axis(n, rng):
    """Planet semi-major axis a [pc] - bimodal log-normal (TdR Image 8, p.27).

    Binary-only. Bimodal in log10-space:
      - Dominant peak at ~3-5e-7 pc (close-in orbits, ~70%)
      - Secondary bump at ~5-20e-6 pc (wider orbits, ~30%)
    """
    w1, mu1, sig1 = 0.70, -6.4, 0.35
    w2, mu2, sig2 = 0.30, -5.0, 0.40

    component = rng.choice([0, 1], size=n, p=[w1, w2])
    log10_a = np.empty(n, dtype=np.float64)

    m1 = component == 0
    m2 = component == 1
    log10_a[m1] = rng.normal(mu1, sig1, size=m1.sum())
    log10_a[m2] = rng.normal(mu2, sig2, size=m2.sum())

    a_pc = np.power(np.float64(10.0), log10_a)
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

    Beta(15, 6) scaled to [14, 22]: mode ~19.85 mag.
    Matches the observed OGLE-IV distribution: left-skewed with a long tail toward
    bright (low-magnitude) stars and a steep drop after the peak at ~19.86 mag.
    """
    MAG_MIN, MAG_MAX = np.float64(14.0), np.float64(22.0)
    beta_samples = rng.beta(15.0, 6.0, size=n)
    return (MAG_MIN + beta_samples * (MAG_MAX - MAG_MIN)).astype(np.float64)
