"""End-to-end synthetic microlensing dataset generation."""

import numpy as np
import pandas as pd

from . import distributions as dist
from .lightcurves import (
    compute_binary_lightcurves,
    compute_einstein_quantities,
    compute_single_lightcurves,
)
from .ogle_noise import apply_ogle_imperfections

SEED = 42


def generate_dataset(n_total: int, binary_fraction: float, n_time: int, seed: int = SEED, use_magnitudes: bool = False, ogle_noise: bool = False):
    """Generate a synthetic microlensing dataset.

    Parameters
    ----------
    n_total: total number of events to generate.
    binary_fraction: fraction (0-1) of events that are binary-lens events.
    n_time: number of time points sampled per light curve.
    seed: RNG seed for reproducibility.
    use_magnitudes: if True, convert A(t) light curves to I-band magnitudes
        using I(t) = I_s - 2.5*log10(A(t)), and add I_s_mag column.

    Returns a dict with the assembled DataFrame ``df`` plus the raw
    per-parameter arrays needed for plotting and validation.
    """
    rng = np.random.default_rng(seed)

    n_binary = int(round(n_total * binary_fraction))
    n_binary = max(0, min(n_binary, n_total))
    n_single = n_total - n_binary

    # --- Sample shared parameters for all events ---
    M_star_solar = dist.sample_lens_mass(n_total, rng)
    M_star_kg = M_star_solar * dist.M_SUN

    D_l_pc = dist.sample_distance_to_lens(n_total, rng)
    D_l_m = D_l_pc * dist.PC_TO_M

    D_ls_pc = dist.sample_lens_source_distance(n_total, rng)
    D_ls_m = D_ls_pc * dist.PC_TO_M
    D_s_pc = D_l_pc + D_ls_pc
    D_s_m = D_s_pc * dist.PC_TO_M

    v_perp_kms = dist.sample_lens_velocity(n_total, rng)
    v_perp_ms = v_perp_kms * dist.KM_TO_M

    u0_all = dist.sample_impact_parameter(n_total, rng)

    # --- Binary-only parameters ---
    q_binary = dist.sample_mass_ratio(n_binary, rng)
    a_pc_binary = dist.sample_semi_major_axis(n_binary, rng)
    a_m_binary = a_pc_binary * dist.PC_TO_M
    e_binary = dist.sample_eccentricity(n_binary, rng)
    alpha_ref_binary = dist.sample_trajectory_angle(n_binary, rng)

    # --- Derived quantities (TdR p.8-9) ---
    r_E_m, t_E_s = compute_einstein_quantities(
        M_star_kg, D_l_m, D_ls_m, D_s_m, v_perp_ms, dist.C_LIGHT
    )
    t_E_days = t_E_s / 86400.0

    # --- Light curves ---
    single_lightcurves = compute_single_lightcurves(u0_all[:n_single], n_time)

    if n_binary > 0:
        binary_lightcurves = compute_binary_lightcurves(
            n_time,
            M_star_kg=M_star_kg[n_single:],
            a_m=a_m_binary,
            e=e_binary,
            q=q_binary,
            u0=u0_all[n_single:],
            r_E_m=r_E_m[n_single:],
            t_E_s=t_E_s[n_single:],
            alpha_ref=alpha_ref_binary,
        )
    else:
        binary_lightcurves = np.empty((0, n_time), dtype=np.float64)

    all_lightcurves = np.vstack([single_lightcurves, binary_lightcurves])

    # --- Optional magnitude conversion ---
    I_s_mag = None
    if use_magnitudes:
        I_s_mag = dist.sample_baseline_magnitude(n_total, rng)
        # I(t) = I_s - 2.5 * log10(A(t)); guard against A<=0 edge cases
        all_lightcurves = (
            I_s_mag[:, np.newaxis]
            - 2.5 * np.log10(np.maximum(all_lightcurves, 1e-10))
        )

    # --- Optional OGLE-IV imperfections (noise + cadence gaps) ---
    if use_magnitudes and ogle_noise:
        all_lightcurves = apply_ogle_imperfections(all_lightcurves, t_E_days, rng)

    # --- Assemble DataFrame ---
    time_cols = [f"t_{j:03d}" for j in range(n_time)]

    event_lenses = np.concatenate([
        np.ones(n_single, dtype=np.int32),
        np.full(n_binary, 2, dtype=np.int32),
    ])

    q_full = np.full(n_total, np.nan, dtype=np.float64)
    q_full[n_single:] = q_binary

    a_pc_full = np.full(n_total, np.nan, dtype=np.float64)
    a_pc_full[n_single:] = a_pc_binary

    e_full = np.full(n_total, np.nan, dtype=np.float64)
    e_full[n_single:] = e_binary

    alpha_ref_full = np.full(n_total, np.nan, dtype=np.float64)
    alpha_ref_full[n_single:] = alpha_ref_binary

    params_dict = {
        "event_lenses": event_lenses,
        "M_star_solar": M_star_solar,
        "D_l_pc": D_l_pc,
        "D_ls_pc": D_ls_pc,
        "D_s_pc": D_s_pc,
        "v_perp_kms": v_perp_kms,
        "u0": u0_all,
        "r_E_m": r_E_m,
        "t_E_days": t_E_days,
        "q": q_full,
        "a_pc": a_pc_full,
        "eccentricity": e_full,
        "alpha_ref_rad": alpha_ref_full,
    }

    if use_magnitudes:
        params_dict["I_s_mag"] = I_s_mag

    df_params = pd.DataFrame(params_dict)
    df_lightcurves = pd.DataFrame(all_lightcurves, columns=time_cols, dtype=np.float64)
    df = pd.concat([df_params, df_lightcurves], axis=1)

    return {
        "df": df,
        "n_total": n_total,
        "n_single": n_single,
        "n_binary": n_binary,
        "n_time": n_time,
        "use_magnitudes": use_magnitudes,
        "M_star_solar": M_star_solar,
        "D_l_pc": D_l_pc,
        "D_ls_pc": D_ls_pc,
        "v_perp_kms": v_perp_kms,
        "u0_all": u0_all,
        "t_E_days": t_E_days,
        "q_binary": q_binary,
        "a_pc_binary": a_pc_binary,
        "e_binary": e_binary,
        "alpha_ref_binary": alpha_ref_binary,
        "single_lightcurves": single_lightcurves,
        "binary_lightcurves": binary_lightcurves,
        "I_s_mag": I_s_mag,
        "ogle_noise": ogle_noise,
    }