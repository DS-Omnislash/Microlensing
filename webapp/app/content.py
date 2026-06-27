"""Static explanatory content describing the distribution logic.

All references are to ``TDR_ROC.pdf`` (Roc Rubio, "Gravitational
Microlensing"), section "Parameters distributions" (pp. 20-29).
"""

# Recommended single/binary split, used as the default in
# Dataset1generator.ipynb (95,000 single + 5,000 binary = 100,000 events).
RECOMMENDED_BINARY_PERCENT = 5.0
RECOMMENDED_N_TIME = 400

EVENT_RATIO_EXPLANATION = (
    "Confirmed planetary (binary-lens) microlensing detections are rare "
    "compared to the total number of observed microlensing events: most "
    "alerts turn out to be single-lens (point-source point-lens) events. "
    "The reference dataset in this study used 95,000 single-lens events and "
    "5,000 binary-lens events (a 95% / 5% split), which keeps the binary "
    "class representative without overwhelming the dataset with the much "
    "more expensive per-event Jacobian ray-shooting computation."
)

N_TIME_EXPLANATION = (
    "Each light curve is sampled on a normalized time grid "
    "tau = (t - t0) / tE in the range [-3, 3], i.e. six Einstein "
    "timescales centered on the peak of the event. 400 points per curve "
    "(the value used in the reference dataset) gives a smooth light curve "
    "shape while keeping the resulting CSV a manageable size."
)

# Parameter cards shown in the "Distribution logic" section.
PARAMETER_INFO = [
    {
        "key": "M_star_solar",
        "name": "Lens Mass (M*)",
        "unit": "Solar masses (Msun)",
        "applies_to": "All events",
        "distribution": "Bimodal Gaussian mixture",
        "formula": "0.25 x N(0.10, 0.06^2) + 0.75 x N(0.45, 0.22^2), clipped to [0.01, 1.2]",
        "reference": "TdR Image 2, p.20 (NASA Exoplanet Archive)",
        "explanation": (
            "The lens mass directly sets the Einstein radius. Real "
            "microlensing lens masses, taken from the NASA Exoplanet "
            "Archive, peak around 0.45 Msun (typical K/G dwarfs) with a "
            "secondary low-mass population around 0.10 Msun (M dwarfs) and "
            "a tail extending to ~1.2 Msun."
        ),
    },
    {
        "key": "D_l_pc",
        "name": "Distance to Lens (D_l)",
        "unit": "parsecs (pc)",
        "applies_to": "All events",
        "distribution": "Bulge-peaked Gaussian mixture",
        "formula": "0.20 x N(1500, 800^2) + 0.80 x N(6800, 750^2), clipped to [300, 8500]",
        "reference": "TdR Image 5, p.23 (Paczynski 1991)",
        "explanation": (
            "Paczynski's analysis shows a prominent peak in the Galactic "
            "bulge regime (6-7 kpc), where most surveys point their "
            "telescopes because of the high stellar density. A smaller "
            "disk-lens population sits at 0.5-3 kpc."
        ),
    },
    {
        "key": "D_ls_pc",
        "name": "Lens-Source Distance (D_ls)",
        "unit": "parsecs (pc)",
        "applies_to": "All events",
        "distribution": "Structured uniform with mild peaks",
        "formula": "0.70 x U(100, 8000) + 0.15 x N(1000, 500^2) + 0.15 x N(5000, 800^2), clipped to [100, 8000]",
        "reference": "TdR Image 4, p.22 (NASA Exoplanet Archive)",
        "explanation": (
            "Few catalogued events report the source distance directly, so "
            "D_ls = D_s - D_l is reconstructed from limited data. The "
            "result is an irregular, roughly uniform distribution with mild "
            "peaks near 1 kpc and 5 kpc."
        ),
    },
    {
        "key": "v_perp_kms",
        "name": "Lens Transversal Velocity (v_perp)",
        "unit": "km/s",
        "applies_to": "All events",
        "distribution": "Maxwell-Boltzmann",
        "formula": "MaxwellBoltzmann(scale = 200/sqrt(2)), i.e. mode = 200 km/s",
        "reference": "TdR p.24 (Rahvar 2015)",
        "explanation": (
            "Transversal velocity cannot be measured directly from "
            "photometry alone, so a Maxwell-Boltzmann distribution with a "
            "characteristic (modal) speed of 200 km/s is assumed, "
            "consistent with Galactic kinematic models."
        ),
    },
    {
        "key": "u0",
        "name": "Impact Parameter (u0)",
        "unit": "dimensionless (Einstein radii)",
        "applies_to": "All events",
        "distribution": "Truncated exponential (telescope bias)",
        "formula": "TruncExp(lambda=3) on [0, 1]",
        "reference": "TdR Image 7, p.25 (OGLE-IV)",
        "explanation": (
            "Theoretically u0 should be uniform on (0,1) (Paczynski 1986), "
            "since the alignment probability is proportional to the "
            "projected area within the Einstein ring. However, real OGLE-IV "
            "survey data shows a strong concentration near u0=0: events "
            "with small impact parameters produce much higher peak "
            "amplification and are easier to detect, biasing the observed "
            "sample toward u0~0."
        ),
    },
    {
        "key": "t_E_days",
        "name": "Einstein Time (t_E)",
        "unit": "days",
        "applies_to": "All events (derived)",
        "distribution": "Derived quantity",
        "formula": "t_E = r_E / v_perp, where r_E = sqrt(4GM/c^2 * D_l*D_ls/D_s)",
        "reference": "TdR p.7-8",
        "explanation": (
            "Not sampled directly: t_E follows from the Einstein radius "
            "(set by lens mass and the three distances) and the transversal "
            "velocity. Its distribution is therefore a consequence of all "
            "the other sampled parameters."
        ),
    },
    {
        "key": "q",
        "name": "Mass Ratio (q)",
        "unit": "dimensionless",
        "applies_to": "Binary-lens events only",
        "distribution": "Log-normal",
        "formula": "log10(q) ~ N(log10(1.43e-3), 1.0^2), clipped to [1e-6, 0.999]",
        "reference": "TdR Image 3, p.21 (NASA Exoplanet Archive)",
        "explanation": (
            "q = M_planet / (M_star + M_planet) sets how strongly the "
            "secondary body perturbs the light curve. Computed from "
            "confirmed binary/planetary systems, it is well described by a "
            "log-normal distribution peaking near q ~ 1.43e-3."
        ),
    },
    {
        "key": "a_pc",
        "name": "Semi-major Axis (a)",
        "unit": "parsecs (pc)",
        "applies_to": "Binary-lens events only",
        "distribution": "Bimodal log-normal",
        "formula": "0.70 x LogN(10^-6.4, 0.35) + 0.30 x LogN(10^-5.0, 0.40), clipped to [1e-8, 0.1]",
        "reference": "TdR Image 8, p.26 (NASA Planetary Systems)",
        "explanation": (
            "The planet's orbital semi-major axis controls its angular "
            "separation from the host star. NASA Planetary Systems data "
            "shows a bimodal distribution: a dominant peak for close-in "
            "orbits (~1e-7 to 1e-6 pc) and a secondary bump for wider orbits "
            "(~1e-5 pc), with a long tail beyond."
        ),
    },
    {
        "key": "eccentricity",
        "name": "Orbital Eccentricity (e)",
        "unit": "dimensionless",
        "applies_to": "Binary-lens events only",
        "distribution": "Beta distribution",
        "formula": "Beta(alpha=1.5, beta=12), mode ~ 0.043",
        "reference": "TdR Image 9, p.26-27 (NASA Planetary Systems, van Eylen+ 2018)",
        "explanation": (
            "Together with the semi-major axis, eccentricity defines the "
            "planet's orbit shape. Real catalogues show many exactly-zero "
            "values from surveys that assume circular orbits, but the "
            "underlying distribution decays roughly exponentially from 0, "
            "well captured by Beta(1.5, 12)."
        ),
    },
    {
        "key": "alpha_ref_rad",
        "name": "Trajectory Angle (alpha_ref)",
        "unit": "radians",
        "applies_to": "Binary-lens events only",
        "distribution": "Uniform",
        "formula": "U(0, 2*pi)",
        "reference": "TdR p.28",
        "explanation": (
            "The orientation of the source trajectory relative to the "
            "binary axis at the reference time is theoretically completely "
            "random, so it is sampled uniformly over a full circle."
        ),
    },
    {
        "key": "I_s_mag",
        "name": "Source Baseline Magnitude (I_s)",
        "unit": "magnitudes (mag)",
        "applies_to": "All events (I(t) mode only)",
        "distribution": "Scaled Beta distribution",
        "formula": "Beta(15, 6) scaled to [14, 22], mode ~19.85 mag",
        "reference": "TdR Image 10, p.28 (OGLE-IV, Mroz et al. 2019)",
        "explanation": (
            "The apparent I-band brightness of the background source star at "
            "baseline (no lensing). Required to convert the dimensionless "
            "amplification A(t) into an observable magnitude light curve via "
            "I(t) = I_s - 2.5 log10 A(t). Derived from the OGLE-IV microlensing "
            "event catalogue: the distribution peaks near I_s ~ 19.9 mag, "
            "reflecting the stellar luminosity function of the Galactic Bulge "
            "as sampled by OGLE-IV detection efficiency."
        ),
    },
]