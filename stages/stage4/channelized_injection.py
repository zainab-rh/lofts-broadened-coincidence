"""Energy-conserving injections on the *channelized* LOFAR grid.

The analytic peak-retention relation in Gajjar & Brown (2026) describes
continuous spectral flux density for an intrinsic linewidth W0.  A SIGPROC
waterfall, however, stores power in finite ~2.98 Hz channels.  For training a
post-detection classifier, the safest numerical convention is therefore to
integrate the chosen profile over each channel and explicitly normalise the
in-tile weights.  This module does that and exposes two distinct experiments:

``snr_mode='power'``
    Hold the unbroadened, full-tile matched-filter S/N fixed and redistribute
    the same signal power as width increases.  This is the physically hard
    stress test; broad-line S/N necessarily declines.

``snr_mode='detected'``
    Hold the *broadened-template* matched-filter S/N fixed.  This represents
    the population conditional on a width-aware upstream detector
    having already reported a candidate.  It is the primary distribution for
    evaluating the downstream coincidence classifier used for the downstream coincidence experiment.

Keeping both modes prevents an apparent ML improvement from being produced by
silently changing the injected signal population.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from candidate_preprocessing import CandidateParams, robust_sigma, track_channels
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, correlation_lags
from scipy.special import ndtr


class ProfileTruncationError(ValueError):
    """Raised when a requested profile is not sufficiently contained in a tile."""


@dataclass(frozen=True)
class InjectionResult:
    shape: str
    width_hz: float
    drift_hz_per_s: float
    center_channel: float
    target_snr: float
    snr_mode: str
    amplitude: float
    noise_projection_sigma: float
    template_l2: float
    expected_broadened_snr: float
    in_tile_energy_fraction_min: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _profile_cdf(x_hz: np.ndarray, width_hz: float, shape: str) -> np.ndarray:
    """CDF of a unit-area profile centred at zero."""

    shape = str(shape).lower()
    width_hz = float(width_hz)
    if width_hz <= 0 or not np.isfinite(width_hz):
        raise ValueError("width_hz must be finite and positive")
    x = np.asarray(x_hz, dtype=np.float64)
    if shape == "lorentzian":
        # gamma = FWHM/2; CDF = 1/2 + atan(x/gamma)/pi.
        return 0.5 + np.arctan(2.0 * x / width_hz) / np.pi
    if shape == "gaussian":
        sigma = width_hz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return ndtr(x / sigma)
    if shape == "box":
        return np.clip(x / width_hz + 0.5, 0.0, 1.0)
    raise ValueError("shape must be 'lorentzian', 'gaussian', or 'box'")


def safe_center_channel_interval(
    n_channels: int,
    width_hz: float,
    df_hz: float,
    shape: str = "lorentzian",
    minimum_captured_fraction: float = 0.95,
    safety_fraction: float = 5e-4,
) -> Tuple[float, float]:
    """Return the exact centre interval that keeps a profile inside a cutout.

    The calculation uses the same analytic, channel-edge-integrated CDF as the
    injector.  It therefore handles Lorentzian wings correctly and avoids an
    unreliable fixed-number-of-channels edge guard.  ``safety_fraction`` plans
    to a slightly stricter captured-mass target than the injector's hard
    threshold, protecting against floating-point equality at the boundary.

    The returned limits apply to a *single time row*.  A drifting track must
    keep every row centre inside this interval; :mod:`stage4_data` enforces that
    additional geometric constraint from the actual ``foff`` and ``tsamp``.
    """

    if n_channels < 2:
        raise ValueError("n_channels must be at least 2")
    if not np.isfinite(df_hz) or df_hz <= 0:
        raise ValueError("df_hz must be finite and positive")
    if not (0 < minimum_captured_fraction <= 1):
        raise ValueError("minimum_captured_fraction must be in (0, 1]")
    if not np.isfinite(safety_fraction) or safety_fraction < 0:
        raise ValueError("safety_fraction must be finite and non-negative")

    # Validate width/shape once and retain float64 arithmetic throughout.
    _profile_cdf(np.asarray([0.0]), width_hz, shape)
    midpoint = (float(n_channels) - 1.0) / 2.0

    def captured(center_channel: float) -> float:
        left_hz = (-0.5 - center_channel) * df_hz
        right_hz = (n_channels - 0.5 - center_channel) * df_hz
        values = _profile_cdf(
            np.asarray([left_hz, right_hz], dtype=np.float64),
            width_hz,
            shape,
        )
        return float(values[1] - values[0])

    maximum = captured(midpoint)
    if maximum + 1e-12 < minimum_captured_fraction:
        raise ValueError(
            "profile cannot satisfy %.4f captured mass in %d channels: "
            "maximum is %.4f" % (minimum_captured_fraction, n_channels, maximum)
        )

    # Do not make an otherwise feasible configuration impossible solely to
    # obtain the numerical buffer when its centre mass is close to the limit.
    target = min(
        minimum_captured_fraction + safety_fraction,
        minimum_captured_fraction + 0.5 * max(maximum - minimum_captured_fraction, 0.0),
    )
    edge_capture = captured(0.0)
    if edge_capture >= target:
        lower = 0.0
    else:
        # Captured mass is monotone on [0, midpoint] for all supported symmetric
        # profiles.  Bisection finds the first admissible centre without a
        # Lorentzian-specific approximation.
        rejected = 0.0
        accepted = midpoint
        for _ in range(80):
            trial = 0.5 * (rejected + accepted)
            if captured(trial) >= target:
                accepted = trial
            else:
                rejected = trial
        lower = accepted
    upper = (float(n_channels) - 1.0) - lower
    if lower > upper + 1e-10:
        raise ValueError("no safe profile-centre interval exists")
    return float(lower), float(upper)


def channel_integrated_weights(
    n_channels: int,
    center_channel: float,
    width_hz: float,
    df_hz: float,
    shape: str = "lorentzian",
    renormalize_in_tile: bool = True,
) -> Tuple[np.ndarray, float]:
    """Integrate a continuous profile over every channel boundary.

    Returns ``(weights, captured_fraction)``.  Before optional
    renormalisation, ``captured_fraction`` is the analytic profile mass inside
    this cutout.  Candidate generation rejects severely truncated profiles.
    """

    if n_channels < 2:
        raise ValueError("n_channels must be at least 2")
    if not np.isfinite(center_channel):
        raise ValueError("center_channel must be finite")
    if not np.isfinite(df_hz) or df_hz <= 0:
        raise ValueError("df_hz must be finite and positive")
    edges = (np.arange(n_channels + 1, dtype=np.float64) - 0.5 - center_channel) * df_hz
    cdf = _profile_cdf(edges, width_hz, shape)
    weights = np.maximum(np.diff(cdf), 0.0)
    captured = float(np.sum(weights))
    if not np.isfinite(captured) or captured <= 0:
        raise ProfileTruncationError("profile has no mass inside the requested cutout")
    if renormalize_in_tile:
        weights = weights / captured
    return weights.astype(np.float64, copy=False), captured


def delta_template(shape: Tuple[int, int], params: CandidateParams) -> np.ndarray:
    """Unit-energy unresolved track with sub-channel linear deposition."""

    params.validate(shape)
    out = np.zeros(shape, dtype=np.float64)
    for row, centre in enumerate(track_channels(params, shape[0])):
        lo = int(np.floor(centre))
        frac = float(centre - lo)
        if 0 <= lo < shape[1]:
            out[row, lo] += 1.0 - frac
        if 0 <= lo + 1 < shape[1]:
            out[row, lo + 1] += frac
    return out


def broadened_template(
    shape: Tuple[int, int],
    params: CandidateParams,
    profile_shape: str = "lorentzian",
    temporal_envelope: Optional[np.ndarray] = None,
    minimum_captured_fraction: float = 0.95,
) -> Tuple[np.ndarray, float]:
    """Create a per-row unit-energy broadened drifting template."""

    params.validate(shape)
    if not (0 < minimum_captured_fraction <= 1):
        raise ValueError("minimum_captured_fraction must be in (0, 1]")
    if temporal_envelope is None:
        envelope = np.ones(shape[0], dtype=np.float64)
    else:
        envelope = np.asarray(temporal_envelope, dtype=np.float64)
        if envelope.shape != (shape[0],):
            raise ValueError("temporal_envelope must have one value per time row")
        if np.any(~np.isfinite(envelope)) or np.any(envelope < 0):
            raise ValueError("temporal_envelope must be finite and non-negative")
        mean = float(np.mean(envelope))
        if mean <= 0:
            raise ValueError("temporal_envelope must have positive mean")
        envelope = envelope / mean

    out = np.zeros(shape, dtype=np.float64)
    fractions = []
    for row, centre in enumerate(track_channels(params, shape[0])):
        weights, captured = channel_integrated_weights(
            shape[1],
            centre,
            params.width_hz,
            abs(params.channel_step_hz),
            shape=profile_shape,
            renormalize_in_tile=True,
        )
        fractions.append(captured)
        out[row] = envelope[row] * weights
    minimum = float(np.min(fractions))
    if minimum < minimum_captured_fraction:
        raise ProfileTruncationError(
            "candidate profile is truncated by the cutout: minimum captured "
            "fraction %.4f < %.4f" % (minimum, minimum_captured_fraction)
        )
    return out, minimum


def make_temporal_envelope(
    n_rows: int,
    rng: np.random.Generator,
    log_sigma: float = 0.12,
    correlation_rows: float = 2.0,
) -> np.ndarray:
    """Smooth positive modulation for mild scintillation/amplitude variation."""

    if log_sigma <= 0:
        return np.ones(n_rows, dtype=np.float64)
    white = rng.normal(0.0, log_sigma, size=n_rows)
    if correlation_rows > 0:
        white = gaussian_filter1d(white, correlation_rows, mode="reflect")
    env = np.exp(white)
    return env / np.mean(env)


def _centred_background(background: np.ndarray) -> np.ndarray:
    x = np.asarray(background, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("background must be time x frequency")
    finite = np.isfinite(x)
    fill = float(np.median(x[finite])) if np.any(finite) else 0.0
    x = np.where(finite, x, fill)
    return x - np.median(x, axis=1, keepdims=True)


def estimate_noise_projection_sigma(
    background: np.ndarray,
    params: CandidateParams,
    profile_shape: str,
    rng: np.random.Generator,
    n_offsource: int = 48,
    temporal_envelope: Optional[np.ndarray] = None,
    template: Optional[np.ndarray] = None,
) -> float:
    """Estimate matched-filter noise from off-candidate frequency shifts.

    This uses the real tile rather than assuming independent Gaussian pixels,
    so stationary bandpass/RFI correlations enter the injection calibration.
    A MAD estimate limits the influence of a few bright RFI shifts.
    """

    x = _centred_background(background)
    n_rows, n_cols = x.shape
    # The projections are computed by translating one already-valid template,
    # so a two-FWHM edge guard is sufficient.  The former six-FWHM guard plus
    # a twelve-FWHM exclusion left *no* off-source samples at 100 Hz in a
    # 1024-channel tile and silently fell back to a raw-pixel noise scale.
    half_guard = max(
        8,
        int(np.ceil(2.0 * params.width_hz / abs(params.channel_step_hz))),
    )
    drift_span = (
        abs(params.drift_hz_per_s)
        * params.dt_s
        * (n_rows - 1)
        / abs(params.channel_step_hz)
    )
    edge = int(np.ceil(half_guard + drift_span + 2))
    lo = edge
    hi = n_cols - edge
    if hi <= lo:
        return robust_sigma(x)

    candidates = np.arange(lo, hi, dtype=int)
    candidates = candidates[np.abs(candidates - params.center_channel) > 2 * half_guard]
    if candidates.size < 8:
        return robust_sigma(x)
    if candidates.size > n_offsource:
        candidates = rng.choice(candidates, size=n_offsource, replace=False)

    if template is None:
        template, _ = broadened_template(
            x.shape,
            params,
            profile_shape=profile_shape,
            temporal_envelope=temporal_envelope,
            minimum_captured_fraction=0.90,
        )
    scores = _frequency_shift_projection_scores(
        x, template, params.center_channel, candidates
    )
    if scores.size < 4:
        return robust_sigma(x)
    return robust_sigma(scores)


def _frequency_shift_projection_scores(
    background: np.ndarray,
    template: np.ndarray,
    template_center_channel: float,
    requested_centers: np.ndarray,
) -> np.ndarray:
    """Project all integer frequency translations of one track efficiently."""

    x = np.asarray(background, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    if x.shape != t.shape or x.ndim != 2:
        raise ValueError("background and template must be matching 2D arrays")
    norm = float(np.linalg.norm(t))
    if norm <= 0:
        return np.empty(0, dtype=np.float64)
    summed = np.zeros(x.shape[1], dtype=np.float64)
    for row in range(x.shape[0]):
        summed += correlate(x[row], t[row], mode="same", method="fft")
    lags = correlation_lags(x.shape[1], x.shape[1], mode="same")
    desired_lags = np.rint(
        np.asarray(requested_centers, dtype=np.float64) - template_center_channel
    ).astype(int)
    indices = np.searchsorted(lags, desired_lags)
    good = (
        (indices >= 0)
        & (indices < lags.size)
        & (lags[np.clip(indices, 0, lags.size - 1)] == desired_lags)
    )
    return summed[indices[good]] / norm


def estimate_delta_projection_sigma(
    background: np.ndarray,
    params: CandidateParams,
    rng: np.random.Generator,
    n_offsource: int = 64,
) -> float:
    """Estimate unresolved-track noise for the fixed-power experiment.

    The returned scale is the standard deviation of *unit-normalised matched
    filter projections*, not a raw-pixel standard deviation.  This matters
    when the real filterbank background is correlated in time or frequency.
    """

    x = _centred_background(background)
    n_rows, n_cols = x.shape
    drift_span = (
        abs(params.drift_hz_per_s)
        * params.dt_s
        * (n_rows - 1)
        / abs(params.channel_step_hz)
    )
    edge = int(np.ceil(drift_span + 3.0))
    candidates = np.arange(edge, n_cols - edge, dtype=int)
    candidates = candidates[np.abs(candidates - params.center_channel) > 8]
    if candidates.size < 8:
        return robust_sigma(x)
    if candidates.size > n_offsource:
        candidates = rng.choice(candidates, size=n_offsource, replace=False)

    template = delta_template(x.shape, params)
    scores = _frequency_shift_projection_scores(
        x, template, params.center_channel, candidates
    )
    if scores.size < 4:
        return robust_sigma(x)
    return robust_sigma(scores)


def inject_broadened_signal(
    background: np.ndarray,
    params: CandidateParams,
    target_snr: float,
    profile_shape: str = "lorentzian",
    snr_mode: str = "detected",
    rng: Optional[np.random.Generator] = None,
    temporal_envelope: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, InjectionResult]:
    """Inject a channel-integrated signal with an explicit S/N definition.

    Returns ``(data_with_signal, signal_only, metadata)``.
    """

    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(background, dtype=np.float64)
    params.validate(x.shape)
    if not np.isfinite(target_snr) or target_snr <= 0:
        raise ValueError("target_snr must be finite and positive")
    snr_mode = str(snr_mode).lower()
    if snr_mode not in ("detected", "power"):
        raise ValueError("snr_mode must be 'detected' or 'power'")

    template, captured = broadened_template(
        x.shape,
        params,
        profile_shape=profile_shape,
        temporal_envelope=temporal_envelope,
    )
    broad_norm = float(np.linalg.norm(template))

    if snr_mode == "detected":
        calibration_template = template
        calibration_shape = profile_shape
    else:
        calibration_template = delta_template(x.shape, params)
        calibration_shape = "lorentzian"

    calibration_norm = float(np.linalg.norm(calibration_template))
    if calibration_norm <= 0 or broad_norm <= 0:
        raise ValueError("signal template is degenerate")

    if snr_mode == "detected":
        noise_sigma = estimate_noise_projection_sigma(
            x,
            params,
            profile_shape=calibration_shape,
            rng=rng,
            temporal_envelope=temporal_envelope,
            template=template,
        )
    else:
        noise_sigma = estimate_delta_projection_sigma(x, params, rng=rng)

    amplitude = float(target_snr * noise_sigma / calibration_norm)
    signal = amplitude * template
    expected_broad_snr = float(amplitude * broad_norm / noise_sigma)
    result = InjectionResult(
        shape=str(profile_shape).lower(),
        width_hz=float(params.width_hz),
        drift_hz_per_s=float(params.drift_hz_per_s),
        center_channel=float(params.center_channel),
        target_snr=float(target_snr),
        snr_mode=snr_mode,
        amplitude=amplitude,
        noise_projection_sigma=float(noise_sigma),
        template_l2=broad_norm,
        expected_broadened_snr=expected_broad_snr,
        in_tile_energy_fraction_min=captured,
    )
    return (x + signal).astype(np.float32), signal.astype(np.float32), result


def _self_test() -> None:
    rng = np.random.default_rng(8)
    shape = (16, 1024)
    step = -2.980232239
    background = rng.normal(0.0, 1.0, size=shape)
    print("channelized_injection.py self-test")

    for profile in ("lorentzian", "gaussian", "box"):
        weights, fraction = channel_integrated_weights(
            1024, 511.5, 30.0, abs(step), profile
        )
        if not np.isclose(weights.sum(), 1.0, atol=1e-12):
            raise AssertionError("channel weights do not conserve in-tile energy")
        if fraction < 0.98:
            raise AssertionError("unexpectedly truncated central profile")
        print("  PASS channel integration/energy: %s" % profile)

    params_10 = CandidateParams(511.5, -0.4, 10.0, step, 0.67108864)
    params_100 = CandidateParams(511.5, -0.4, 100.0, step, 0.67108864)
    _, _, detected_10 = inject_broadened_signal(
        background, params_10, 12.0, snr_mode="detected", rng=rng
    )
    _, _, detected_100 = inject_broadened_signal(
        background, params_100, 12.0, snr_mode="detected", rng=rng
    )
    if not np.isclose(detected_10.expected_broadened_snr, 12.0, rtol=1e-6):
        raise AssertionError("detected-SNR calibration failed at 10 Hz")
    if not np.isclose(detected_100.expected_broadened_snr, 12.0, rtol=1e-6):
        raise AssertionError("detected-SNR calibration failed at 100 Hz")
    if detected_100.amplitude <= detected_10.amplitude:
        raise AssertionError(
            "wider detected-conditioned signal should require more power"
        )
    print("  PASS detected-conditioned S/N calibration")

    _, _, power_10 = inject_broadened_signal(
        background, params_10, 12.0, snr_mode="power", rng=rng
    )
    _, _, power_100 = inject_broadened_signal(
        background, params_100, 12.0, snr_mode="power", rng=rng
    )
    if power_100.expected_broadened_snr >= power_10.expected_broadened_snr:
        raise AssertionError("power-controlled S/N should decline with width")
    print(
        "  PASS power-controlled degradation: 10 Hz %.3f -> 100 Hz %.3f"
        % (power_10.expected_broadened_snr, power_100.expected_broadened_snr)
    )
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
