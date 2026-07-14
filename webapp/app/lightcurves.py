"""Light curve and derived-quantity computations for synthetic events.

Single-lens events use the Paczynski amplification. Binary-lens events are solved
properly in the image plane: the binary lens equation is recast as a 5th-order
complex polynomial whose roots are the image positions, and the magnification is
the sum of 1/|det J| over the images that actually satisfy the lens equation
(Witt & Mao 1995). See ``compute_binary_lightcurves``.

TdR pp. 8-18.
"""

import numpy as np

from .distributions import G

# Chunk size for the batched quintic solve, in (event, time) samples. Bounds peak
# memory: each sample needs a 5x5 complex companion matrix (400 bytes).
_ROOT_CHUNK = 200_000

# An image is accepted only if it maps back to the source position to within
# this tolerance (relative to the source distance from the origin, floored at 1).
# The quintic has 5 roots but only 3 or 5 are true images; the rest are spurious.
_IMAGE_TOL = 1e-6


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
    """Binary-lens magnification, solved in the image plane (TdR p.16-18).

    All inputs are 1-D arrays of length n_binary. Returns an
    (n_binary, n_time) array of amplifications, clipped to [1, 1e6].

    Method (Witt & Mao 1995)
    ------------------------
    The magnification is NOT 1/|det J| evaluated at the source position -- the
    source and its images are different points. It is

        A = sum over images of  1 / |det J(z_k)|

    so the images must be found first. Writing positions as complex numbers, with
    the primary at the origin and the companion at z2, the lens equation is

        zeta = z - m1 / conj(z)  -  m2 / (conj(z) - conj(z2))

    where lengths are in units of the primary's Einstein radius r_E_star, so
    m1 = 1 and m2 = q = M_planet / M_star (the ratio of the two bodies'
    fractional masses, matching distributions/q_mass_ratio.py). (Working in
    primary units rather than total-mass units keeps u0, t_E and tau meaning
    exactly what they mean everywhere else in this project.) Substituting the
    conjugate equation into
    itself clears the denominators and yields a 5th-order polynomial in z whose
    roots contain the images. A quintic has 5 roots but a binary lens has only 3
    or 5 real images, so each root is kept only if it maps back to the source.

    Then, with kappa = d(zeta)/d(conj z) = m1/conj(z)^2 + m2/(conj(z) - conj(z2))^2,

        det J = 1 - |kappa|^2

    which vanishes exactly on the caustics, giving the sharp caustic-crossing
    spikes. In the limit m2 -> 0 this reproduces Paczynski to ~1e-7 relative error.
    """
    n_binary = M_star_kg.shape[0]
    tau = np.linspace(-3.0, 3.0, n_time, dtype=np.float64)

    # Companion mass in units of the primary mass (TdR p.10). Lengths are in units
    # of the primary's Einstein radius, so the primary's own mass term is exactly 1
    # and the companion's is q itself: q = m_p / m_star, the ratio of the two
    # fractional masses m_i = M_i / (M_star + M_planet), which reduces to
    # M_planet / M_star -- the same quantity distributions/q_mass_ratio.py extracts.
    m1 = 1.0
    m2 = q.astype(np.float64)                        # = M_planet / M_star

    # Angular velocity / orbital phase (TdR p.14-15)
    omega = np.sqrt(G * M_star_kg / a_m ** 3).astype(np.float64)

    # --- Source and companion positions for every (event, time) sample ---------- #
    # Source travels along y = u0 at x = tau, in units of r_E_star [TdR p.16].
    zeta = tau[None, :] + 1j * u0[:, None]                       # (n_binary, n_time)

    t_phys = tau[None, :] * t_E_s[:, None]
    alpha_t = alpha_ref[:, None] + omega[:, None] * t_phys
    r_planet = a_m[:, None] * (1.0 - e[:, None] ** 2) / (
        1.0 + e[:, None] * np.cos(alpha_t)
    )
    d_t = r_planet / r_E_m[:, None]                              # separation in r_E_star
    z2 = d_t * np.exp(1j * alpha_t)                              # companion position

    m2_b = np.broadcast_to(m2[:, None], zeta.shape)

    amp = np.empty(zeta.shape, dtype=np.float64)

    # Flatten to a list of independent samples and solve in memory-bounded chunks.
    flat_zeta = zeta.ravel()
    flat_z2 = z2.ravel()
    flat_m2 = m2_b.ravel()

    for start in range(0, flat_zeta.size, _ROOT_CHUNK):
        sl = slice(start, start + _ROOT_CHUNK)
        amp.ravel()[sl] = _binary_amplification(
            flat_zeta[sl], flat_z2[sl], m1, flat_m2[sl]
        )

    return np.clip(amp, 1.0, 1e6)


def _binary_amplification(zeta, z2, m1, m2):
    """Point-source magnification for a 2-body lens: m1 at the origin, m2 at z2.

    zeta, z2, m2 : (N,) arrays -- source position, companion position, companion
                   mass (all in units of the primary's Einstein radius / mass).
    Returns (N,) float64 magnifications.
    """
    zeb = np.conj(zeta)
    z2b = np.conj(z2)

    # Coefficients of the 5th-order image polynomial, highest power first.
    # Derived symbolically from the lens equation with the primary at the origin;
    # valid for any m1, m2 (they need not sum to 1).
    c0 = -z2b * zeb + zeb ** 2
    c1 = (m1 * zeb + m2 * (-z2b + zeb) + z2 * (2 * z2b * zeb - 2 * zeb ** 2)
          + zeta * (z2b * zeb - zeb ** 2))
    c2 = (z2 * (-2 * m1 * zeb + m2 * z2b + z2 * (-z2b * zeb + zeb ** 2))
          + zeta * (m1 * (z2b - 2 * zeb) + m2 * (z2b - 2 * zeb)
                    + z2 * (-2 * z2b * zeb + 2 * zeb ** 2)))
    c3 = (z2 * (m1 * m2 + m2 ** 2 + z2 * (m1 * zeb - m2 * zeb))
          + zeta * (m1 * (-m1 - 2 * m2) - m2 ** 2
                    + z2 * (m1 * (-2 * z2b + 4 * zeb) + m2 * (-z2b + 2 * zeb)
                            + z2 * (z2b * zeb - zeb ** 2))))
    c4 = (-m1 * m2 * z2 ** 2
          + z2 * zeta * (m1 * z2 * (z2b - 2 * zeb) + m1 * (2 * m1 + 2 * m2)))
    c5 = -m1 ** 2 * z2 ** 2 * zeta

    coeffs = np.stack([c0, c1, c2, c3, c4, c5], axis=-1)         # (N, 6)

    # Roots via batched companion-matrix eigenvalues (what np.roots does, but for
    # all N quintics at once -- np.roots is scalar-only and far too slow here).
    n = coeffs.shape[0]
    lead = coeffs[:, 0]
    lead = np.where(np.abs(lead) < 1e-300, 1e-300 + 0j, lead)
    companion = np.zeros((n, 5, 5), dtype=np.complex128)
    companion[:, 0, :] = -coeffs[:, 1:] / lead[:, None]
    sub = np.arange(4)
    companion[:, sub + 1, sub] = 1.0
    roots = np.linalg.eigvals(companion)                         # (N, 5)

    zb = np.conj(roots)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Keep only roots that are genuine images (map back to the source).
        mapped = roots - m1 / zb - m2[:, None] / (zb - z2b[:, None])
        residual = np.abs(mapped - zeta[:, None])

        # det J = 1 - |d(zeta)/d(conj z)|^2 ; zero on the caustics.
        kappa = m1 / zb ** 2 + m2[:, None] / (zb - z2b[:, None]) ** 2
        image_amp = 1.0 / np.abs(1.0 - np.abs(kappa) ** 2)

    scale = np.maximum(np.abs(zeta), 1.0)[:, None]
    is_image = np.isfinite(residual) & (residual < _IMAGE_TOL * scale)
    return np.where(is_image & np.isfinite(image_amp), image_amp, 0.0).sum(axis=1)
