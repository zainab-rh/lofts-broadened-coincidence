"""Physical broadening and injection utilities for Stage 3.

It implements the Exo-IPM broadening model, peak-retention calculation,
Monte Carlo population sampling, and synthetic Lorentzian, box, and Gaussian
injections used by the training and evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import setigen as stg
    from astropy import units as u

    _HAVE_SETIGEN = True
except ImportError:  # allows the pure-physics parts to be unit-tested
    _HAVE_SETIGEN = False
# =============================================================================
# 0. LOFAR / LOFTS-PIPELINE STANDARD CONSTANTS (Johnson et al. 2023, Table 1)
# =============================================================================
LOFAR_DF_HZ = 2.980232239  # Hz / channel, fine-frequency (narrowband search) product
LOFAR_DT_S = 0.67108864  # s / time sample, fine-frequency (narrowband search) product
LOFAR_FCH1_MHZ = 150.0  # nominal band-centre used in quick demos
LOFAR_BAND_MIN_GHZ = 0.1099609375  # 109.9609375 MHz (Table 1)
LOFAR_BAND_MAX_GHZ = 0.1900390625  # 190.0390625 MHz (Table 1)
# The 2.98 Hz / 0.671 s values describe the fine-frequency ``0000`` product.
# The kHz-resolution ``0001`` product is not interchangeable for this
# broadening analysis. Runtime code reads df and dt from each filterbank
# header; these constants are documentation and demonstration defaults.
#
# Two distinct integration times are retained for clarity.
#   - ML_TILE_TOBS_S: the duration of a single (512, 128) high_time training
#     tile, i.e. 512 time samples * dt. This is the relevant T_obs for any
#     "drift resolvable within one training/evaluation tile" claim, and is
#     specific to the high_time
#     mode; use `tile_tobs_s()` below for the mode actually in use.
#   - FULL_SCAN_TOBS_S: the duration of one real LOFTS observing scan
#     (15 minutes, Johnson et al. 2023 Sec. 2.1). This is the relevant T_obs
#     for any "drift resolvable within a real observation" claim.
ML_TILE_TOBS_S = 512 * LOFAR_DT_S  # 343.5573... s for high_time mode
FULL_SCAN_TOBS_S = 900.0  # = 15 min


def tile_tobs_s(tchans: int, dt_s: float) -> float:
    """Mode-agnostic replacement for the hardcoded `ML_TILE_TOBS_S` constant
    above: the observed duration of a training/evaluation tile with
    `tchans` time samples at a per-sample cadence of `dt_s` seconds (read
    from the actual filterbank header, not assumed). Use this instead of
    `ML_TILE_TOBS_S` for any mode other than the original high_time
    (512, 128) configuration."""
    return float(tchans) * float(dt_s)


# Profile-shape indices are metadata for reporting, not supervised targets.
CLASS_NONE = 0
CLASS_NARROWBAND = 1
CLASS_LORENTZIAN = 2
CLASS_BOX = 3
CLASS_GAUSSIAN = 4
CLASS_NAMES = {
    CLASS_NONE: "none",
    CLASS_NARROWBAND: "narrowband",
    CLASS_LORENTZIAN: "lorentzian",
    CLASS_BOX: "box_decoy",
    CLASS_GAUSSIAN: "gaussian_decoy",
}
# =============================================================================
# 1. PEAK-NORMALISED SHAPE FUNCTIONS (pure morphology; carry NO energy scaling)
# =============================================================================


def lorentzian_shape(width_hz: float) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Peak-normalized Lorentzian (Cauchy) frequency profile for setigen's
    ``add_signal(f_profile=...)``.
        L(f) = gamma^2 / ((f - f_center)^2 + gamma^2),   gamma = width_hz / 2
    Peak is exactly 1.0 at f_center for any width_hz; this function carries
    morphology only. To inject a physically energy-conserving signal, scale
    the `level` argument of `t_profile` by `peak_retention_factor()` below;
    do not modify this function to "fix" the amplitude here.
    Parameters
    ----------
    width_hz : float
        Full width at half maximum (FWHM) = the physical spectral broadening
        Delta_nu_sb, in Hz (bare float; see module docstring's unit note).
    Returns
    -------
    callable(freqs_hz, f_center_hz) -> np.ndarray, same shape as freqs_hz.
    """
    gamma = max(float(width_hz), 1e-9) / 2.0

    def profile(freqs, f_center):
        delta_f = np.asarray(freqs, dtype=float) - np.asarray(f_center, dtype=float)
        return gamma**2 / (delta_f**2 + gamma**2)

    return profile


def gaussian_shape(width_hz: float) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Peak-normalized Gaussian profile (FWHM = width_hz). Used both as the
    standard "narrowband" injection class (matching the convention already
    used by the existing training pipeline's `f_profile_type='gaussian'`)
    and, at a broadened width, as a non-Lorentzian shape decoy for the
    match/mismatch discrimination stress-test (see train.py's
    'shape_decoy_match' case).
    """
    sigma = max(float(width_hz), 1e-9) / 2.3548200450309493  # FWHM -> sigma

    def profile(freqs, f_center):
        delta_f = np.asarray(freqs, dtype=float) - np.asarray(f_center, dtype=float)
        return np.exp(-0.5 * (delta_f / sigma) ** 2)

    return profile


def box_shape(width_hz: float) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Peak-normalized top-hat profile using the same FWHM convention as the
    Lorentzian profile. It provides a matched-width, matched-energy decoy for
    testing whether coincidence performance depends on the injected shape.
    """
    half_width = max(float(width_hz), 1e-9) / 2.0

    def profile(freqs, f_center):
        delta_f = np.asarray(freqs, dtype=float) - np.asarray(f_center, dtype=float)
        return np.where(np.abs(delta_f) <= half_width, 1.0, 0.0)

    return profile


SHAPE_REGISTRY = {
    "lorentzian": lorentzian_shape,
    "gaussian": gaussian_shape,
    "box": box_shape,
}
# =============================================================================
# 2. ENERGY-CONSERVING AMPLITUDE
# =============================================================================


def peak_retention_factor(
    delta_nu_sb_hz: float, intrinsic_width_hz: float = 1.0
) -> float:
    """
    Fraction of peak amplitude retained when an intrinsically W0-wide
    top-hat/delta-like transmission is broadened to FWHM = delta_nu_sb_hz by
    a Lorentzian scattering kernel.
    Gajjar & Brown (2026), Sections 2.1 and 7.1:
        retention = (2/pi) * arctan(W0 / Delta_nu_sb)
    For W0=1 Hz and Delta_nu_sb=10 Hz, the retained fraction is 0.06344,
    consistent with the approximately 6% worked example in that study.
    Parameters
    ----------
    delta_nu_sb_hz : float
        Spectral broadening Delta_nu_sb (Hz). Clipped to >= 1e-9 so that the
        zero-broadening limit correctly saturates retention -> 1.0.
    intrinsic_width_hz : float
        Intrinsic (transmitted) linewidth W0, in Hz. The paper adopts 1 Hz
        as its standard assumption for narrowband technosignature searches.
    Returns
    -------
    float in (0, 1].
    """
    delta_nu_sb_hz = max(float(delta_nu_sb_hz), 1e-9)
    return float((2.0 / np.pi) * np.arctan(intrinsic_width_hz / delta_nu_sb_hz))


def energy_conserving_level(
    base_level: float, delta_nu_sb_hz: float, intrinsic_width_hz: float = 1.0
) -> float:
    """
    The value to pass as the `t_profile` level so that a signal injected
    with `lorentzian_shape(delta_nu_sb_hz)` (or `box_shape`/`gaussian_shape`
    at the same width, for a matched-energy decoy) has a peak amplitude
    consistent with conserving the total transmitted power of an
    intrinsically `intrinsic_width_hz`-wide signal, instead of stamping a
    constant peak regardless of width.
    `base_level` should be the SNR-equivalent peak amplitude the signal
    would have if it arrived perfectly narrowband (i.e. the same `level`
    you would pass for a 1-channel-wide Gaussian narrowband signal at the
    desired SNR).
    """
    return float(base_level) * peak_retention_factor(delta_nu_sb_hz, intrinsic_width_hz)


def effective_post_broadening_snr(
    base_snr: float, delta_nu_sb_hz: float, intrinsic_width_hz: float = 1.0
) -> float:
    """Return the retained peak S/N after Lorentzian broadening.

    The calculation is
    ``base_snr * peak_retention_factor(delta_nu_sb_hz, intrinsic_width_hz)``.
    The conditioned training sampler uses this quantity to exclude examples
    whose peaks cannot reach its configured S/N floor.

    Parameters
    ----------
    base_snr : float
        S/N the signal would present if perfectly narrowband.
    delta_nu_sb_hz : float
        Spectral broadening, Hz.
    intrinsic_width_hz : float
        Intrinsic linewidth W0 in the retention law.
    Returns
    -------
    float : effective post-broadening S/N.
    """
    return float(base_snr) * peak_retention_factor(delta_nu_sb_hz, intrinsic_width_hz)


# =============================================================================
# 3. PHYSICAL BROADENING MODEL  (Gajjar & Brown 2026, Eq. 18, 19-21, 22)
# =============================================================================


def compute_delta_nu_sun(R_impact_stellar_radii, freq_GHz: float):
    """
    Spectral broadening for a Sun-like star, steady ambient wind (no CME).
    Gajjar & Brown (2026), Eq. (18) -- empirically anchored to spacecraft
    occultation data (Pioneer 6, Helios 1/2, Viking, Cassini, Mars/Venus
    Express, Rosetta) spanning ~1-200 R_sun and 0.4-8.4 GHz:
        Delta_nu_sb = [900.79*(R/R*)^-3.14 + 47.76*(R/R*)^-1.63] * nu_GHz^(-6/5)  [Hz]
    Here R is the line-of-sight impact distance (the output of Eq. 22), not
    the orbital semimajor axis. See `orbital_impact_distance()`.
    Hand-verified reference values used throughout this project:
        R_impact = 10 R*,  150 MHz -> 17.27 Hz
        R_impact = 217 R*, 150 MHz ->  0.073 Hz  (~Earth-Sun distance, negligible)
    Parameters
    ----------
    R_impact_stellar_radii : float or array
        LINE-OF-SIGHT IMPACT DISTANCE (perpendicular distance from the
        star to the Earth-transmitter sightline), in stellar radii.
    freq_GHz : float or array
        Observing frequency, GHz. LOFAR HBA: 0.110-0.190 GHz. May be an
        array (e.g. to sweep frequency at fixed R) as long as its shape
        broadcasts against R_impact_stellar_radii's shape.
    Returns
    -------
    float or array : Delta_nu_sb in Hz.
    """
    R = np.asarray(R_impact_stellar_radii, dtype=float)
    R = np.clip(R, 1.0, None)  # cannot be inside the stellar surface
    freq_factor = np.asarray(freq_GHz, dtype=float) ** (-6.0 / 5.0)
    term_inner = 900.79 * (R**-3.14)
    term_outer = 47.76 * (R**-1.63)
    return (term_inner + term_outer) * freq_factor


def compute_delta_nu_mdwarf(
    R_impact_stellar_radii,
    freq_GHz: float,
    wind_factor: float = 4.0,
    turbulence_factor: float = 30.0,
):
    """
    Spectral broadening for an M-dwarf star.
    Gajjar & Brown (2026), Eq. (19)-(21): the enhancement over the
    solar-calibrated radial law is LINEAR in the stellar-wind speed ratio
    and scales as the turbulence-integral ratio to the 3/5 power -- both
    exponents follow directly from Delta_nu_sb = V_perp / r_diff with
    r_diff ~ SM^(-3/5) (their Eq. 5-6), since Cn^2 is the SM integrand.
    R is the line-of-sight impact distance in M-dwarf stellar radii. Because
    M-dwarf radii are approximately 0.1--0.7 solar radii, the same R/R* ratio
    represents a smaller absolute distance than for a Sun-like star.
    Parameters
    ----------
    R_impact_stellar_radii : float or array
        Line-of-sight impact distance in M-dwarf stellar radii.
    freq_GHz : float
        Observing frequency, GHz.
    wind_factor : float
        V_perp,dM / V_perp,sun. Paper's adopted range [1, 8] (Eq. 19).
    turbulence_factor : float
        Cn2,dM / Cn2,sun. Paper's adopted range [0.3, 150] (Eq. 20).
    Returns
    -------
    float or array : Delta_nu_sb in Hz.
    """
    sun_broadening = compute_delta_nu_sun(R_impact_stellar_radii, freq_GHz)
    enhancement = float(wind_factor) * (float(turbulence_factor) ** (3.0 / 5.0))
    return sun_broadening * enhancement


def orbital_impact_distance(
    a_stellar_radii: float,
    eccentricity: float,
    inclination_rad: float,
    arg_periastron_rad: float,
    true_anomaly_rad: float,
) -> float:
    """
    Sky-projected LINE-OF-SIGHT IMPACT DISTANCE R from orbital elements.
    This is the quantity passed to `compute_delta_nu_sun` or
    `compute_delta_nu_mdwarf`, rather than the orbital separation alone.
    Gajjar & Brown (2026), Eq. (22):
        R_imp(f) = a(1-e^2)/(1+e*cos f) * sqrt(1 - sin^2(i)*sin^2(omega+f))
    The expression is symmetric in orbital phase. R_imp varies across an
    entire orbit: it is smallest (down to ~1 R*,
    superior conjunction, maximal broadening) near omega+f = +/-90 deg, and
    reaches its MAXIMUM (~a, minimal broadening) at quadrature
    (omega+f = 0 or 180 deg). A single frozen value of R (e.g. the orbital
    separation `a` itself) is only valid AT quadrature and is not a general
    substitute for R at other orbital phases.
    Parameters
    ----------
    a_stellar_radii : float
        Semimajor axis, in stellar radii.
    eccentricity : float
        Orbital eccentricity (0 = circular).
    inclination_rad : float
        Orbital inclination, radians. pi/2 = edge-on (transit geometry).
    arg_periastron_rad : float
        Argument of periastron, radians.
    true_anomaly_rad : float
        True anomaly (orbital phase), radians.
    Returns
    -------
    float : impact distance, stellar radii. Clipped to a minimum of 1.0
        (the stellar surface) to avoid a coordinate singularity.
    """
    e = float(eccentricity)
    r = a_stellar_radii * (1.0 - e**2) / (1.0 + e * np.cos(true_anomaly_rad))
    geom = np.sqrt(
        np.clip(
            1.0
            - (np.sin(inclination_rad) ** 2)
            * (np.sin(arg_periastron_rad + true_anomaly_rad) ** 2),
            0.0,
            None,
        )
    )
    return float(max(r * geom, 1.0))


def is_behind_star(
    inclination_rad: float, arg_periastron_rad: float, true_anomaly_rad: float
) -> bool:
    """
    True if the planet is on the far side of the star as seen from Earth
    (superior-conjunction-like geometry). The sign convention defines +z as
    pointing away from the observer. This diagnostic is not applied as an
    additional correction to the broadening law.
    """
    z_sign = np.sin(inclination_rad) * np.sin(arg_periastron_rad + true_anomaly_rad)
    return bool(z_sign > 0.0)


def broadening_curve_over_orbit(
    a_stellar_radii: float,
    eccentricity: float,
    inclination_rad: float,
    arg_periastron_rad: float,
    freq_GHz: float,
    stellar_type: str = "sunlike",
    wind_factor: float = 4.0,
    turbulence_factor: float = 30.0,
    n_phases: int = 360,
) -> tuple:
    """
    Full-orbit broadening curve for a system with known orbital elements.
    Use this rather than a single fixed impact distance because R_imp can
    vary by orders of magnitude over an orbit.
    Parameters
    ----------
    a_stellar_radii, eccentricity, inclination_rad, arg_periastron_rad :
        Orbital elements (true anomaly is swept internally).
    freq_GHz : float
    stellar_type : {'sunlike', 'mdwarf'}
    wind_factor, turbulence_factor : float
        Only used if stellar_type == 'mdwarf'.
    n_phases : int
        Number of true-anomaly samples across [0, 2*pi).
    Returns
    -------
    true_anomaly_rad : np.ndarray, shape (n_phases,)
    R_impact_rstar : np.ndarray, shape (n_phases,)
    delta_nu_sb_hz : np.ndarray, shape (n_phases,)
    """
    f_vals = np.linspace(0.0, 2 * np.pi, n_phases, endpoint=False)
    R_vals = np.array(
        [
            orbital_impact_distance(
                a_stellar_radii, eccentricity, inclination_rad, arg_periastron_rad, f
            )
            for f in f_vals
        ]
    )
    if stellar_type == "mdwarf":
        dnu_vals = compute_delta_nu_mdwarf(
            R_vals, freq_GHz, wind_factor, turbulence_factor
        )
    else:
        dnu_vals = compute_delta_nu_sun(R_vals, freq_GHz)
    return f_vals, R_vals, dnu_vals


# =============================================================================
# 4. MONTE CARLO POPULATION SAMPLER (joint width + drift, physically self-consistent)
# =============================================================================
# Fundamental constants (SI), needed for the joint orbital -> drift computation
_G_CONST = 6.67430e-11  # m^3 kg^-1 s^-2
_C_LIGHT = 2.99792458e8  # m/s
_M_SUN_KG = 1.98892e30
_R_SUN_M = 6.957e8
_AU_M = 1.495978707e11


@dataclass
class BroadeningDraw:
    """One draw from the Exo-IPM population model.

    Width and drift are derived from the same orbital geometry.
    """

    width_hz: float
    drift_hz_per_s: float
    stellar_type: str  # 'sunlike' or 'mdwarf'
    stellar_mass_msun: float
    impact_distance_rstar: float
    semimajor_axis_rstar: float
    eccentricity: float
    inclination_rad: float
    arg_periastron_rad: float
    true_anomaly_rad: float
    wind_factor: float
    turbulence_factor: float
    is_behind_star: bool


def _drift_rate_from_orbit(
    M_star_msun: float, r_m: float, inclination_rad: float, freq_hz: float
) -> float:
    """
    Doppler drift-rate magnitude implied by a specific orbital geometry,
    following Li et al. (2023), Eq. 1-2:
        nu_dot_norm [s^-1] = G * M_star * sin(i) / (c * r^2)
        nu_dot [Hz/s]      = nu_dot_norm * nu_obs [Hz]
    Parameters
    ----------
    M_star_msun : float
        Host star mass, solar masses.
    r_m : float
        Instantaneous star-planet separation, metres.
    inclination_rad : float
        Orbital inclination, radians.
    freq_hz : float
        Observing frequency, Hz.
    Returns
    -------
    float : unsigned drift rate magnitude (caller assigns a random sign,
        since prograde/retrograde acceleration phase is equally likely a
        priori), Hz/s.
    """
    M_kg = M_star_msun * _M_SUN_KG
    nu_dot_norm = (_G_CONST * M_kg * abs(np.sin(inclination_rad))) / (_C_LIGHT * r_m**2)
    return float(nu_dot_norm * freq_hz)


def sample_one_broadening(
    freq_GHz: float = 0.150,
    p_mdwarf: float = 0.75,
    rng: Optional[np.random.Generator] = None,
    min_width_hz: float = 0.05,
    max_width_hz: float = 20000.0,
) -> BroadeningDraw:
    """Draw coupled broadening and drift from the adopted population model.

    The model follows Gajjar & Brown (2026), Table 1 and Sections 6.1--6.3:
    the host is drawn from the configured M-dwarf/Sun-like mixture, orbital
    scale is log-uniform, inclination is isotropic, and phase and argument of
    periastron are uniform. M-dwarf wind and turbulence factors are sampled
    from the stated model ranges.

    Width is computed from the line-of-sight impact geometry; drift is derived
    from the same orbital draw using Li et al. (2023), Eqs. 1--2. The clipping
    bounds provide numerical safety and optional curriculum caps. For a named
    system, use `broadening_curve_over_orbit()` with measured orbital elements
    instead of this population sampler.

    Parameters
    ----------
    freq_GHz : float
        Observing frequency, GHz.
    p_mdwarf : float
        Probability of drawing an M-dwarf host (vs. Sun-like).
    rng : np.random.Generator, optional
    min_width_hz, max_width_hz : float
        Clipping bounds applied for numerical safety and optional curriculum
        learning.
    Returns
    -------
    BroadeningDraw
    """
    if rng is None:
        rng = np.random.default_rng()
    is_mdwarf = bool(rng.random() < p_mdwarf)
    e = float(rng.uniform(0.0, 0.9))
    inc = float(np.arccos(rng.uniform(-1.0, 1.0)))
    omega = float(rng.uniform(0.0, 2 * np.pi))
    f_true = float(rng.uniform(0.0, 2 * np.pi))
    if is_mdwarf:
        stellar_type = "mdwarf"
        M_star_msun = float(rng.uniform(0.1, 0.6))
        R_star_rsun = float(rng.uniform(0.08, 0.7))
        a_rstar = float(np.exp(rng.uniform(np.log(10.0), np.log(100.0))))
        wind_factor = float(rng.uniform(1.0, 8.0))
        turb_factor = float(np.exp(rng.uniform(np.log(0.3), np.log(150.0))))
    else:
        stellar_type = "sunlike"
        M_star_msun = float(rng.uniform(0.8, 1.2))
        R_star_rsun = float(rng.uniform(0.9, 1.1))
        a_rstar = float(np.exp(rng.uniform(np.log(10.0), np.log(217.0))))
        wind_factor = 1.0
        turb_factor = 1.0
    # Equation 22 converts orbital elements to line-of-sight impact distance.
    R_imp = orbital_impact_distance(a_rstar, e, inc, omega, f_true)
    behind = is_behind_star(inc, omega, f_true)
    if is_mdwarf:
        width = compute_delta_nu_mdwarf(R_imp, freq_GHz, wind_factor, turb_factor)
    else:
        width = compute_delta_nu_sun(R_imp, freq_GHz)
    width = float(np.clip(width, min_width_hz, max_width_hz))
    # Instantaneous star-planet separation (for the drift calculation),
    # from the same orbital draw.
    r_rstar = a_rstar * (1.0 - e**2) / (1.0 + e * np.cos(f_true))
    r_m = r_rstar * (R_star_rsun * _R_SUN_M)
    freq_hz = freq_GHz * 1e9
    drift_magnitude = _drift_rate_from_orbit(M_star_msun, r_m, inc, freq_hz)
    drift_signed = drift_magnitude * (1.0 if rng.random() < 0.5 else -1.0)
    return BroadeningDraw(
        width_hz=width,
        drift_hz_per_s=drift_signed,
        stellar_type=stellar_type,
        stellar_mass_msun=M_star_msun,
        impact_distance_rstar=R_imp,
        semimajor_axis_rstar=a_rstar,
        eccentricity=e,
        inclination_rad=inc,
        arg_periastron_rad=omega,
        true_anomaly_rad=f_true,
        wind_factor=wind_factor,
        turbulence_factor=turb_factor,
        is_behind_star=behind,
    )


def sample_one_broadening_conditioned(
    freq_GHz: float = 0.150,
    p_mdwarf: float = 0.75,
    rng: Optional[np.random.Generator] = None,
    snr_range: tuple = (5.0, 30.0),
    intrinsic_width_hz: float = 1.0,
    min_effective_snr: float = 3.0,
    max_attempts: int = 500,
    min_width_hz: float = 0.05,
    max_width_hz: float = 20000.0,
) -> tuple:
    """Draw a coupled sample subject to a best-case effective-S/N floor.

    Rejection sampling retains draws for which
    ``effective_post_broadening_snr(max(snr_range), width_hz)`` reaches
    `min_effective_snr`. This conditions training on plausibly detectable
    peaks without changing the underlying orbital population model. If the
    attempt budget is exhausted, the highest-effective-S/N draw is returned
    with `accepted=False`.

    Parameters
    ----------
    freq_GHz, p_mdwarf, rng, min_width_hz, max_width_hz :
        Passed through to `sample_one_broadening` unchanged.
    snr_range : tuple
        Minimum and maximum intrinsic (pre-broadening) S/N. Conditioning uses
        the upper bound.
    intrinsic_width_hz : float
        Intrinsic linewidth W0 used by the retention law.
    min_effective_snr : float
        Minimum best-case retained peak S/N required for acceptance.
    max_attempts : int
        Rejection-sampling attempt limit.
    Returns
    -------
    (BroadeningDraw, best_case_effective_snr: float, n_attempts: int,
     accepted: bool)
        `accepted` is false when the attempt budget is exhausted.
    """
    if rng is None:
        rng = np.random.default_rng()
    snr_ceiling = max(snr_range)
    best = None
    for attempt in range(1, max_attempts + 1):
        draw = sample_one_broadening(
            freq_GHz, p_mdwarf, rng, min_width_hz, max_width_hz
        )
        eff = effective_post_broadening_snr(
            snr_ceiling, draw.width_hz, intrinsic_width_hz
        )
        if best is None or eff > best[1]:
            best = (draw, eff, attempt)
        if eff >= min_effective_snr:
            return draw, eff, attempt, True
    return best[0], best[1], best[2], False


def sample_realistic_broadening(
    n_samples: int,
    freq_GHz: float = 0.150,
    p_mdwarf: float = 0.75,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Vectorised convenience wrapper around `sample_one_broadening` --
    returns only the width array (Hz), for quick survival-function-style
    population analyses."""
    if rng is None:
        rng = np.random.default_rng()
    return np.array(
        [
            sample_one_broadening(freq_GHz, p_mdwarf, rng).width_hz
            for _ in range(n_samples)
        ]
    )


# =============================================================================
# 5. SIGNAL INJECTION  (energy-conserving; for use inside setigen Frames)
# =============================================================================


def _resolve_drift_rate_hz_s(drift_rate) -> float:
    """Accept either a bare float (Hz/s) or an astropy Quantity and return a
    bare float in Hz/s, matching the bare-Hz convention used throughout."""
    if _HAVE_SETIGEN and isinstance(drift_rate, u.Quantity):
        return float(drift_rate.to(u.Hz / u.s).value)
    return float(drift_rate)


def add_narrowband_signal(
    frame,
    width_hz: float,
    drift_rate_hz_s,
    base_level: float,
    f_start_hz: Optional[float] = None,
) -> dict:
    """Inject a Gaussian narrowband signal using the shared call convention."""
    if not _HAVE_SETIGEN:
        raise ImportError("setigen is required for signal injection.")
    if f_start_hz is None:
        f_start_hz = frame.get_frequency(
            int(np.random.randint(int(0.2 * frame.fchans), int(0.8 * frame.fchans)))
        )
    drift_rate_hz_s = _resolve_drift_rate_hz_s(drift_rate_hz_s)
    frame.add_constant_signal(
        f_start=f_start_hz,
        drift_rate=drift_rate_hz_s * u.Hz / u.s,
        level=base_level,
        width=max(width_hz, frame.df),
        f_profile_type="gaussian",
    )
    return {
        "signal_type": "narrowband",
        "f_start_hz": f_start_hz,
        "drift_rate_hz_s": drift_rate_hz_s,
        "width_hz": width_hz,
        "injected_level": base_level,
        "profile_class": CLASS_NARROWBAND,
    }


def inject_lorentzian_signal(
    frame,
    f_start_hz: Optional[float],
    drift_rate_hz_s,
    base_level: float,
    delta_nu_sb_hz: float,
    intrinsic_width_hz: float = 1.0,
) -> dict:
    """
    Inject an energy-conserving Lorentzian-broadened signal.
    `base_level` is the SNR-equivalent level the signal would have if
    perfectly narrowband; the actual injected peak amplitude is
    `energy_conserving_level(base_level, delta_nu_sb_hz, intrinsic_width_hz)`,
    so the peak correctly falls following the (2/pi)*arctan(W0/Delta_nu_sb)
    law as `delta_nu_sb_hz` grows.
    Falls back to a clean 1-channel Gaussian narrowband injection when
    `delta_nu_sb_hz` is at or below the channel width (i.e. effectively
    unresolved / delta-like), matching the standard SETI-literature
    convention for an unresolved signal.
    Parameters
    ----------
    frame : setigen.Frame
    f_start_hz : float or None
        Injection frequency, Hz. Randomly chosen within the frame's central
        60% if None.
    drift_rate_hz_s : float or astropy Quantity
        Doppler drift rate.
    base_level : float
        SNR-equivalent unbroadened peak amplitude.
    delta_nu_sb_hz : float
        Target FWHM (the physical spectral broadening), Hz.
    intrinsic_width_hz : float
        W0 in the retention law, default 1 Hz.
    Returns
    -------
    dict of injection parameters, including the actual injected peak level
    and the retention fraction applied, for logging/verification.
    """
    if not _HAVE_SETIGEN:
        raise ImportError("setigen is required for signal injection.")
    if f_start_hz is None:
        f_start_hz = frame.get_frequency(
            int(np.random.randint(int(0.2 * frame.fchans), int(0.8 * frame.fchans)))
        )
    drift_rate_hz_s = _resolve_drift_rate_hz_s(drift_rate_hz_s)
    # Explicit, unambiguous branch on channel resolution.
    if delta_nu_sb_hz <= frame.df:
        level = base_level
        frame.add_constant_signal(
            f_start=f_start_hz,
            drift_rate=drift_rate_hz_s * u.Hz / u.s,
            level=level,
            width=frame.df,
            f_profile_type="gaussian",
        )
        return {
            "signal_type": "lorentzian",
            "f_start_hz": f_start_hz,
            "drift_rate_hz_s": drift_rate_hz_s,
            "width_hz": 0.0,
            "base_level": base_level,
            "injected_level": level,
            "retention": 1.0,
            "profile_class": CLASS_NARROWBAND,
        }
    injected_level = energy_conserving_level(
        base_level, delta_nu_sb_hz, intrinsic_width_hz
    )
    frame.add_signal(
        path=stg.constant_path(
            f_start=f_start_hz, drift_rate=drift_rate_hz_s * u.Hz / u.s
        ),
        t_profile=stg.constant_t_profile(level=injected_level),
        f_profile=lorentzian_shape(delta_nu_sb_hz),
    )
    return {
        "signal_type": "lorentzian",
        "f_start_hz": f_start_hz,
        "drift_rate_hz_s": drift_rate_hz_s,
        "width_hz": delta_nu_sb_hz,
        "base_level": base_level,
        "injected_level": injected_level,
        "retention": peak_retention_factor(delta_nu_sb_hz, intrinsic_width_hz),
        "profile_class": CLASS_LORENTZIAN,
    }


def inject_box_decoy_signal(
    frame,
    f_start_hz: Optional[float],
    drift_rate_hz_s,
    base_level: float,
    width_hz: float,
    intrinsic_width_hz: float = 1.0,
) -> dict:
    """
    Inject a box-shaped decoy with matched width and energy budget.

    The FWHM, drift rate, and post-retention peak amplitude match the
    corresponding Lorentzian injection; only the profile shape differs.
    """
    if not _HAVE_SETIGEN:
        raise ImportError("setigen is required for signal injection.")
    if f_start_hz is None:
        f_start_hz = frame.get_frequency(
            int(np.random.randint(int(0.2 * frame.fchans), int(0.8 * frame.fchans)))
        )
    drift_rate_hz_s = _resolve_drift_rate_hz_s(drift_rate_hz_s)
    injected_level = energy_conserving_level(base_level, width_hz, intrinsic_width_hz)
    frame.add_signal(
        path=stg.constant_path(
            f_start=f_start_hz, drift_rate=drift_rate_hz_s * u.Hz / u.s
        ),
        t_profile=stg.constant_t_profile(level=injected_level),
        f_profile=box_shape(width_hz),
    )
    return {
        "signal_type": "box_decoy",
        "f_start_hz": f_start_hz,
        "drift_rate_hz_s": drift_rate_hz_s,
        "width_hz": width_hz,
        "base_level": base_level,
        "injected_level": injected_level,
        "retention": peak_retention_factor(width_hz, intrinsic_width_hz),
        "profile_class": CLASS_BOX,
    }


def inject_gaussian_decoy_signal(
    frame,
    f_start_hz: Optional[float],
    drift_rate_hz_s,
    base_level: float,
    width_hz: float,
    intrinsic_width_hz: float = 1.0,
) -> dict:
    """Same convention as `inject_box_decoy_signal`, but Gaussian-shaped."""
    if not _HAVE_SETIGEN:
        raise ImportError("setigen is required for signal injection.")
    if f_start_hz is None:
        f_start_hz = frame.get_frequency(
            int(np.random.randint(int(0.2 * frame.fchans), int(0.8 * frame.fchans)))
        )
    drift_rate_hz_s = _resolve_drift_rate_hz_s(drift_rate_hz_s)
    injected_level = energy_conserving_level(base_level, width_hz, intrinsic_width_hz)
    frame.add_signal(
        path=stg.constant_path(
            f_start=f_start_hz, drift_rate=drift_rate_hz_s * u.Hz / u.s
        ),
        t_profile=stg.constant_t_profile(level=injected_level),
        f_profile=gaussian_shape(width_hz),
    )
    return {
        "signal_type": "gaussian_decoy",
        "f_start_hz": f_start_hz,
        "drift_rate_hz_s": drift_rate_hz_s,
        "width_hz": width_hz,
        "base_level": base_level,
        "injected_level": injected_level,
        "retention": peak_retention_factor(width_hz, intrinsic_width_hz),
        "profile_class": CLASS_GAUSSIAN,
    }


def make_lofar_frame(
    fchans: int = 128,
    tchans: int = 512,
    df_hz: float = LOFAR_DF_HZ,
    dt_s: float = LOFAR_DT_S,
    fch1_mhz: float = LOFAR_FCH1_MHZ,
    noise_std: float = 1.0,
    noise_mean: float = 0.0,
):
    """Convenience constructor for a LOFAR-shaped setigen Frame with
    Gaussian noise already added, matching Johnson et al. (2023) Table 1's
    fine-frequency (narrowband-search) product by default."""
    if not _HAVE_SETIGEN:
        raise ImportError("setigen is required to build Frames.")
    frame = stg.Frame(
        fchans=fchans,
        tchans=tchans,
        df=df_hz * u.Hz,
        dt=dt_s * u.s,
        fch1=fch1_mhz * u.MHz,
        ascending=True,
    )
    frame.add_noise(x_mean=noise_mean, x_std=noise_std, noise_type="gaussian")
    return frame


# =============================================================================
# 6. VISUAL PHYSICS VALIDATION
# =============================================================================


def plot_physics_validation(
    save_path: str = "physics_validation.png", freq_ghz: float = 0.150, seed: int = 42
) -> None:
    """
    Create a six-panel visual validation of the implemented physics:
      (a) Peak retention factor vs. Delta_nu_sb, with the published
          worked example (10 Hz -> ~6%) marked explicitly.
      (b) Sun-like vs. M-dwarf broadening vs. impact distance R_imp, at
          LOFAR's 150 MHz, showing the ~20x M-dwarf enhancement.
      (c) Full-orbit broadening curve for a representative close-in,
          edge-on system (TRAPPIST-1b-like elements), visually
          demonstrating why R_imp cannot be represented by a single value.
      (d) Monte Carlo population: width distribution (log-x histogram),
          from the joint orbital sampler.
      (e) Monte Carlo population: |drift rate| distribution.
      (f) Frequency-scaling cross-check: Delta_nu_sb ratio between LOFAR
          (150 MHz) and GBT L-band (1.4 GHz), confirming the nu^(-6/5) law.
    Parameters
    ----------
    save_path : str
        Output PNG path.
    freq_ghz : float
        Observing frequency for panels (a), (b), (d), (e). Default 150 MHz
        (LOFAR HBA band centre).
    seed : int
        RNG seed for the Monte Carlo panels, for reproducibility.
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes
    # --- (a) Peak retention curve -------------------------------------------
    widths = np.logspace(np.log10(0.5), np.log10(3000), 200)
    retention_pct = np.array([peak_retention_factor(w) * 100 for w in widths])
    ax_a.plot(widths, retention_pct, color="cyan", lw=2)
    ref_retention = peak_retention_factor(10.0) * 100
    ax_a.scatter(
        [10.0],
        [ref_retention],
        color="yellow",
        s=80,
        zorder=5,
        label=f"Paper's example: 10 Hz -> {ref_retention:.2f}%",
    )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Δνsb (Hz)")
    ax_a.set_ylabel("Peak retention (%)")
    ax_a.set_title("(a) Peak retention law\n(2/π)·arctan(W₀/Δνsb), W₀=1 Hz")
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3, which="both")
    # --- (b) Sun-like vs M-dwarf broadening vs impact distance --------------
    R_range = np.linspace(1, 220, 200)
    dnu_sun = compute_delta_nu_sun(R_range, freq_ghz)
    dnu_mdwarf = compute_delta_nu_mdwarf(
        R_range, freq_ghz, wind_factor=4, turbulence_factor=30
    )
    ax_b.plot(R_range, dnu_sun, color="orange", lw=2, label="Sun-like")
    ax_b.plot(
        R_range, dnu_mdwarf, color="magenta", lw=2, label="M-dwarf (wind=4, turb=30)"
    )
    ax_b.axhline(
        2.980232239, color="gray", ls=":", lw=1, label="1 LOFAR channel (2.98 Hz)"
    )
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Impact distance R_imp (stellar radii)")
    ax_b.set_ylabel("Δνsb (Hz)")
    ax_b.set_title(f"(b) Broadening vs. impact distance @ {freq_ghz*1000:.0f} MHz")
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3, which="both")
    # --- (c) Full-orbit curve for a TRAPPIST-1b-like system ------------------
    a_au, R_star_sun = 0.01154, 0.1192
    a_rstar = a_au * (_AU_M / _R_SUN_M) / R_star_sun
    f_vals, R_vals, dnu_vals = broadening_curve_over_orbit(
        a_stellar_radii=a_rstar,
        eccentricity=0.00622,
        inclination_rad=np.radians(89.65),
        arg_periastron_rad=0.0,
        freq_GHz=freq_ghz,
        stellar_type="mdwarf",
        wind_factor=4.0,
        turbulence_factor=30.0,
        n_phases=720,
    )
    ax_c.plot(np.degrees(f_vals), dnu_vals, color="lightgreen", lw=2)
    ax_c.axhline(
        dnu_vals.min(),
        color="cyan",
        ls=":",
        lw=1,
        label=f"Quadrature min: {dnu_vals.min():.1f} Hz",
    )
    ax_c.axhline(
        dnu_vals.max(),
        color="salmon",
        ls=":",
        lw=1,
        label=f"Conjunction max: {dnu_vals.max():.1f} Hz",
    )
    ax_c.set_yscale("log")
    ax_c.set_xlabel("True anomaly (deg)")
    ax_c.set_ylabel("Δνsb (Hz)")
    ax_c.set_title(
        "(c) Full-orbit broadening, TRAPPIST-1b-like elements\n"
        "(impact distance varies with orbital phase)"
    )
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.3, which="both")
    # --- (d) & (e) Monte Carlo population ------------------------------------
    rng = np.random.default_rng(seed)
    draws = [sample_one_broadening(freq_ghz, 0.75, rng) for _ in range(20000)]
    widths_mc = np.array([d.width_hz for d in draws])
    drifts_mc = np.array([abs(d.drift_hz_per_s) for d in draws])
    ax_d.hist(np.log10(widths_mc), bins=60, color="deepskyblue", alpha=0.85)
    ax_d.axvline(
        np.log10(np.median(widths_mc)),
        color="yellow",
        ls="--",
        label=f"Median: {np.median(widths_mc):.1f} Hz",
    )
    ax_d.set_xlabel("log10(Δνsb / Hz)")
    ax_d.set_ylabel("Count")
    ax_d.set_title(
        "(d) Monte Carlo population: width distribution\n"
        "(75% M-dwarf, joint orbital sampler, n=20,000)"
    )
    ax_d.legend(fontsize=8)
    ax_d.grid(alpha=0.3)
    ax_e.hist(np.log10(drifts_mc + 1e-9), bins=60, color="violet", alpha=0.85)
    ax_e.axvline(
        np.log10(np.median(drifts_mc)),
        color="yellow",
        ls="--",
        label=f"Median: {np.median(drifts_mc):.4f} Hz/s",
    )
    ax_e.set_xlabel("log10(|drift| / Hz s⁻¹)")
    ax_e.set_ylabel("Count")
    ax_e.set_title(
        "(e) Monte Carlo population: |drift rate| distribution\n"
        "(same draws, jointly derived from orbital geometry)"
    )
    ax_e.legend(fontsize=8)
    ax_e.grid(alpha=0.3)
    # --- (f) Frequency-scaling cross-check -----------------------------------
    freqs_ghz = np.logspace(np.log10(0.11), np.log10(2.0), 100)
    dnu_at_freq = compute_delta_nu_sun(50, freqs_ghz)
    ax_f.plot(freqs_ghz * 1000, dnu_at_freq, color="gold", lw=2)
    ax_f.scatter(
        [150],
        [compute_delta_nu_sun(50, 0.150)],
        color="cyan",
        s=70,
        zorder=5,
        label=f"LOFAR 150 MHz: {compute_delta_nu_sun(50, 0.150):.2f} Hz",
    )
    ax_f.scatter(
        [1400],
        [compute_delta_nu_sun(50, 1.400)],
        color="orange",
        s=70,
        zorder=5,
        label=f"GBT L-band 1.4 GHz: {compute_delta_nu_sun(50, 1.400):.4f} Hz",
    )
    ratio = compute_delta_nu_sun(50, 0.150) / compute_delta_nu_sun(50, 1.400)
    ax_f.set_xscale("log")
    ax_f.set_yscale("log")
    ax_f.set_xlabel("Observing frequency (MHz)")
    ax_f.set_ylabel("Δνsb (Hz), R=50 R★")
    ax_f.set_title(
        f"(f) ν^(-6/5) scaling cross-check\n"
        f"LOFAR sees {ratio:.1f}× more broadening than GBT L-band"
    )
    ax_f.legend(fontsize=8)
    ax_f.grid(alpha=0.3, which="both")
    fig.suptitle("Broadening-model physics validation", fontsize=15, y=1.00)
    plt.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved physics validation figure to: {save_path}")


# =============================================================================
# 7. COMMAND-LINE NUMERICAL CHECKS AND PHYSICS FIGURE
# =============================================================================


if __name__ == "__main__":
    print("=" * 72)
    print("lorentzian_signals.py self-test")
    print("=" * 72)
    print("\n[1] Peak retention law (published example: ~94% loss at 10 Hz):")
    for dnu in [3, 10, 30, 100, 300, 1000]:
        r = peak_retention_factor(dnu, 1.0)
        print(
            f"    Delta_nu_sb={dnu:>5} Hz -> retained={r*100:7.3f}%  loss={(1-r)*100:6.2f}%"
        )
    _r10 = peak_retention_factor(10.0, 1.0)
    assert abs(_r10 - 0.0634) < 0.001, "Retention formula failed its reference check."
    print("    PASS: retention(10 Hz, W0=1 Hz) matches paper's stated ~6% ('94% loss')")
    print("\n[1b] effective_post_broadening_snr () at base_snr=10:")
    for dnu in [1, 3, 10, 30, 100]:
        eff = effective_post_broadening_snr(10.0, dnu)
        print(
            f"    Delta_nu_sb={dnu:>4} Hz -> effective SNR = {eff:6.3f}"
            f"{'  (below SNmin=10 search threshold)' if eff < 10 else ''}"
        )
    assert abs(effective_post_broadening_snr(10.0, 10.0) - 0.6345) < 0.001
    print(
        "\n[2] Broadening tables at LOFAR 150 MHz (R = LINE-OF-SIGHT IMPACT DISTANCE):"
    )
    for R in [10, 20, 50, 100, 150, 217]:
        print(
            f"    Sun-like  R_imp={R:>4} R*  -> {compute_delta_nu_sun(R, 0.150):9.4f} Hz"
        )
    for R in [5, 10, 20, 30, 50, 100]:
        print(
            f"    M-dwarf   R_imp={R:>4} R*  -> {compute_delta_nu_mdwarf(R, 0.150, 4, 30):9.2f} Hz"
        )
    print("\n[3] Frequency-scaling cross-check (Delta_nu_sb ~ nu^(-6/5)):")
    ratio_1ghz = compute_delta_nu_sun(50, 0.150) / compute_delta_nu_sun(50, 1.000)
    ratio_gbt = compute_delta_nu_sun(50, 0.150) / compute_delta_nu_sun(50, 1.400)
    print(f"    150 MHz vs 1.0 GHz  -> {ratio_1ghz:.2f}x more broadening at LOFAR")
    print(
        f"    150 MHz vs 1.4 GHz  -> {ratio_gbt:.2f}x more broadening at LOFAR (GBT L-band)"
    )
    expected_1ghz = (0.150 / 1.000) ** (-6.0 / 5.0)
    assert abs(ratio_1ghz - expected_1ghz) < 1e-6
    print(
        "\n[4] Monte Carlo population sample (n=20000, 75% M-dwarf, joint width+drift):"
    )
    rng = np.random.default_rng(42)
    draws = [sample_one_broadening(0.150, 0.75, rng) for _ in range(20000)]
    widths = np.array([d.width_hz for d in draws])
    drifts = np.array([abs(d.drift_hz_per_s) for d in draws])
    print(f"    median width  = {np.median(widths):.2f} Hz")
    print(f"    P(width > 10 Hz)  = {np.mean(widths > 10) * 100:.1f}%")
    print(f"    P(width > 100 Hz) = {np.mean(widths > 100) * 100:.1f}%")
    print(f"    median |drift| = {np.median(drifts):.5f} Hz/s")
    print("\n[4b] curriculum-cap population check (max_width_hz=30.0):")
    rng2 = np.random.default_rng(42)
    draws_capped = [
        sample_one_broadening(0.150, 0.75, rng2, max_width_hz=30.0)
        for _ in range(20000)
    ]
    widths_capped = np.array([d.width_hz for d in draws_capped])
    print(
        "    fraction of draws pinned at the 30 Hz cap = "
        f"{np.mean(np.isclose(widths_capped, 30.0)) * 100:.1f}%"
    )
    print(
        "    (this is the mass that a curriculum sub-stage effectively 'defers' "
        "to a later, wider-cap sub-stage)"
    )
    print("\n[4c] Conditioned broadening sampler:")
    print("     The raw sampler's median width (34 Hz) gives an")
    print("     effective SNR of only 30*0.019=0.57 at the BEST-case intrinsic SNR=30.")
    rng3 = np.random.default_rng(42)
    cond_widths, cond_effs, cond_attempts, cond_accepted = [], [], [], []
    for _ in range(3000):
        draw, eff, n_att, ok = sample_one_broadening_conditioned(
            0.150, 0.75, rng3, snr_range=(5.0, 30.0), min_effective_snr=3.0
        )
        cond_widths.append(draw.width_hz)
        cond_effs.append(eff)
        cond_attempts.append(n_att)
        cond_accepted.append(ok)
    cond_widths = np.array(cond_widths)
    print(
        f"    median width AFTER conditioning  = {np.median(cond_widths):.2f} Hz "
        f"(raw sampler: {np.median(widths):.2f} Hz)"
    )
    print(
        f"    P(width > 10 Hz) after conditioning  = {np.mean(cond_widths > 10) * 100:.1f}% "
        f"(raw sampler: {np.mean(widths > 10) * 100:.1f}%)"
    )
    print(
        f"    mean attempts to accept = {np.mean(cond_attempts):.1f}  "
        f"(max 500); acceptance rate = {np.mean(cond_accepted) * 100:.1f}%"
    )
    assert np.median(cond_widths) < np.median(
        widths
    ), "Conditioning should shift the population toward narrower, more detectable widths."
    print("\n[5] TRAPPIST-1b: WHY a single frozen R is not a valid example")
    a_au, R_star_sun, M_star_sun = 0.01154, 0.1192, 0.089
    a_rstar = a_au * (_AU_M / _R_SUN_M) / R_star_sun
    print(f"    TRAPPIST-1b semimajor axis a = {a_rstar:.1f} R* (a/R*, not R_imp)")
    f_vals, R_vals, dnu_vals = broadening_curve_over_orbit(
        a_stellar_radii=a_rstar,
        eccentricity=0.00622,
        inclination_rad=np.radians(89.65),
        arg_periastron_rad=0.0,
        freq_GHz=0.150,
        stellar_type="mdwarf",
        wind_factor=4.0,
        turbulence_factor=30.0,
        n_phases=720,
    )
    print(
        "    Over one full orbit, Delta_nu_sb (M-dwarf-enhanced) ranges from "
        f"{dnu_vals.min():.1f} Hz (quadrature, R_imp~=a) up to {dnu_vals.max():.1f} Hz "
        "(near conjunction, R_imp->1 R*)"
    )
    print("    -> NO SINGLE FROZEN NUMBER represents this system's broadening.")
    print("\n[6] Generating visual physics validation figure...")
    plot_physics_validation(save_path="physics_validation.png", freq_ghz=0.150, seed=42)
    print("\nAll self-tests completed.")
