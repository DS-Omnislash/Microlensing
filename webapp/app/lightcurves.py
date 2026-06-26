"""Light curve and derived-quantity computations for synthetic events.

Reproduces the single-lens (Pacynski amplification) and binary-lens
(Jacobian inverse ray-shooting) light curve models from
``Dataset1generator.ipynb`` (TdR pp. 8-18).
"""

import numpy as np

from .distributions import G


def compute_einstein_quantities(M_star_kg, D_l_m, D_ls_m, D_s_m, v_perp_ms, c_light):
    """Einstein radius r_E [m] and Einstein time t_E [s] (TdR p.8-9)."""
    r_E_m = np.sqrt(
        4.0 * G * M_star_kg / (c_light ** 2) * (D_l_m * D_ls_m / D_s_m)
    ).astype(np.float64)
    t_E_s = (r_E_m / v_perp_ms).astype(np.float64)
    return r_E_m, t_E_s


def compute_single_lightcurves(u0, n_time):
    """Vectorized Pacynski amplification curves for single-lens events.

    A(u) = (u^2 + 2) / (u * sqrt(u^2 + 4)), with
    u(tau) = sqrt(u0^2 + tau^2), tau in linspace(-3, 3, n_time).
    """
    n = u0.shape[0]
    tau = np.linspace(-3.0, 3.0, n_time, dtype=np.float64)

    u0_2d = u0[:, np.newaxis]
    tau_2d = tau[np.newaxis, :]
    u = np.sqrt(u0_2d ** 2 + tau_2d ** 2)

    u_sq = u ** 2
    A = (u_sq + 2.0) / (u * np.sqrt(u_sq + 4.0))
    return A.astype(np.float64)


def compute_binary_lightcurves(
    n_time,
    M_star_kg,
    a_m,
    e,
    q,
    u0,
    r_E_m,
    t_E_s,
    alpha_ref,
):
    """Per-event Jacobian inverse ray-shooting for binary-lens events (TdR p.16-18).

    All inputs are 1-D arrays of length n_binary. Returns an
    (n_binary, n_time) array of amplifications, clipped to [1, 1e6].
    """
    n_binary = M_star_kg.shape[0]
    tau = np.linspace(-3.0, 3.0, n_time, dtype=np.float64)

    # Planet mass and mass fractions (TdR p.10)
    M_planet_kg = (q * M_star_kg / (1.0 - q)).astype(np.float64)
    M_total_kg = M_star_kg + M_planet_kg
    m_star_frac = (M_star_kg / M_total_kg).astype(np.float64)
    m_planet_frac = (M_planet_kg / M_total_kg).astype(np.float64)

    # Angular velocity / orbital phase (TdR p.14-15)
    omega = np.sqrt(G * M_star_kg / a_m ** 3).astype(np.float64)

    binary_lightcurves = np.empty((n_binary, n_time), dtype=np.float64)

    for i in range(n_binary):
        tE_i = t_E_s[i]
        u0_i = u0[i]
        rE_i = r_E_m[i]
        m_s_i = m_star_frac[i]
        m_p_i = m_planet_frac[i]
        omega_i = omega[i]
        alpha0_i = alpha_ref[i]
        a_i = a_m[i]
        e_i = e[i]

        t_phys = tau * tE_i

        # Source position (normalized by r_E_star) [TdR p.16]
        source_x = tau.copy()
        source_y = np.full(n_time, u0_i, dtype=np.float64)

        # Lens (star) fixed at origin [TdR p.16]
        lens_x = 0.0
        lens_y = 0.0

        # Planet position by time [TdR p.15-17]
        alpha_t = alpha0_i + omega_i * t_phys
        r_planet = a_i * (1.0 - e_i ** 2) / (1.0 + e_i * np.cos(alpha_t))
        d_t = r_planet / rE_i
        planet_x = d_t * np.cos(alpha_t)
        planet_y = d_t * np.sin(alpha_t)

        # Vectors from each mass to the source
        rx_star = source_x - lens_x
        ry_star = source_y - lens_y
        r4_star = np.maximum((rx_star ** 2 + ry_star ** 2) ** 2, 1e-30)

        rx_planet = source_x - planet_x
        ry_planet = source_y - planet_y
        r4_planet = np.maximum((rx_planet ** 2 + ry_planet ** 2) ** 2, 1e-30)

        # Jacobian J = I - sum_i (m_i/|r_i|^4) * [[y_i^2-x_i^2, -2x_i y_i], [-2x_i y_i, x_i^2-y_i^2]]
        J11_star = m_s_i / r4_star * (ry_star ** 2 - rx_star ** 2)
        J12_star = m_s_i / r4_star * (-2.0 * rx_star * ry_star)
        J22_star = m_s_i / r4_star * (rx_star ** 2 - ry_star ** 2)

        J11_planet = m_p_i / r4_planet * (ry_planet ** 2 - rx_planet ** 2)
        J12_planet = m_p_i / r4_planet * (-2.0 * rx_planet * ry_planet)
        J22_planet = m_p_i / r4_planet * (rx_planet ** 2 - ry_planet ** 2)

        J11 = 1.0 - J11_star - J11_planet
        J12 = 0.0 - J12_star - J12_planet
        J21 = J12
        J22 = 1.0 - J22_star - J22_planet

        det_J = J11 * J22 - J12 * J21
        A_binary = 1.0 / np.abs(det_J)
        binary_lightcurves[i, :] = np.clip(A_binary, 1.0, 1e6)

    return binary_lightcurves
