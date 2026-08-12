"""Detector-informed preprocessing for broadened LOFAR candidates.

This module implements the two candidate-informed operations used after the
BLISS search:

1. de-chirp a candidate using its reported drift rate;
2. integrate ("frequency scrunch") over its reported width.

The implementation addresses three requirements that are important for real
LOFAR products:

* ``channel_step_hz`` is *signed*.  LOFAR ``0000.fil`` products normally have
  descending frequency (negative ``foff``), so using ``abs(foff)`` reverses
  the de-chirp direction;
* shifted edges are zero/validity masked instead of being filled by repeated
  edge pixels, which otherwise turns edge RFI into artificial streaks;
* scrunching is an L2-normalised frequency filter.  For whitened independent
  noise its output variance remains approximately one, while signal power
  spread over N channels gains approximately sqrt(N) in S/N.  The default is
  a boxcar is the default, with a Lorentzian filter available as a
  physics-informed sensitivity analysis.

All arrays are time x frequency.  Frequencies may increase or decrease with
column index; the sign is carried exclusively by ``channel_step_hz``.

The functions are NumPy/SciPy-only, so ``python candidate_preprocessing.py``
can be run on a login node without PyTorch, blimpy, or setigen.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import shift as ndi_shift
from scipy.signal import convolve
from scipy.special import ndtr

_EPS = 1e-6


@dataclass(frozen=True)
class CandidateParams:
    """Candidate coordinates in a waterfall cutout.

    Parameters
    ----------
    center_channel
        Signal-centre column at ``reference_row``.
    drift_hz_per_s
        Physical drift rate.  Positive means increasing physical frequency.
    width_hz
        Reported FWHM.  BLISS width must be confirmed to use this definition.
    channel_step_hz
        Signed frequency increment per array column (``foff * 1e6`` for a
        SIGPROC header whose ``foff`` is in MHz).  Never take its absolute
        value before de-chirping.
    dt_s
        Time cadence per row.
    reference_row
        Row at which ``center_channel`` is defined.  BLISS usually reports a
        start frequency, for which this is zero.
    """

    center_channel: float
    drift_hz_per_s: float
    width_hz: float
    channel_step_hz: float
    dt_s: float
    reference_row: float = 0.0

    def validate(self, shape: Tuple[int, int]) -> None:
        if len(shape) != 2:
            raise ValueError("candidate data must be two-dimensional")
        if not np.isfinite(self.center_channel):
            raise ValueError("center_channel must be finite")
        if not np.isfinite(self.drift_hz_per_s):
            raise ValueError("drift_hz_per_s must be finite")
        if not np.isfinite(self.width_hz) or self.width_hz <= 0:
            raise ValueError("width_hz must be finite and positive")
        if not np.isfinite(self.channel_step_hz) or self.channel_step_hz == 0:
            raise ValueError("channel_step_hz must be finite and non-zero")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        if not (0 <= self.center_channel <= shape[1] - 1):
            raise ValueError(
                "center_channel %.3f is outside [0, %d]"
                % (self.center_channel, shape[1] - 1)
            )


def robust_sigma(values: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Return a finite MAD-based Gaussian sigma estimate.

    ``mask=True`` marks values to exclude.  A conventional standard deviation
    is used only if the MAD degenerates; the final floor prevents division by
    zero on pathological tiles.
    """

    x = np.asarray(values, dtype=np.float64)
    good = np.isfinite(x)
    if mask is not None:
        if mask.shape != x.shape:
            raise ValueError("mask shape does not match values")
        good &= ~mask
    x = x[good]
    if x.size == 0:
        return 1.0
    med = float(np.median(x))
    sigma = 1.482602218505602 * float(np.median(np.abs(x - med)))
    if not np.isfinite(sigma) or sigma < _EPS:
        sigma = float(np.std(x))
    return sigma if np.isfinite(sigma) and sigma >= _EPS else 1.0


def track_channels(params: CandidateParams, n_rows: int) -> np.ndarray:
    """Candidate centre in channel coordinates for every time row."""

    rows = np.arange(n_rows, dtype=np.float64)
    elapsed_s = (rows - params.reference_row) * params.dt_s
    return params.center_channel + (
        params.drift_hz_per_s * elapsed_s / params.channel_step_hz
    )


def candidate_mask(
    shape: Tuple[int, int],
    params: CandidateParams,
    half_width_factor: float = 2.5,
    minimum_half_width_channels: int = 2,
) -> np.ndarray:
    """Mask the candidate corridor while estimating background statistics."""

    params.validate(shape)
    n_rows, n_cols = shape
    half_width = max(
        int(minimum_half_width_channels),
        int(np.ceil(half_width_factor * params.width_hz / abs(params.channel_step_hz))),
    )
    centers = track_channels(params, n_rows)
    cols = np.arange(n_cols, dtype=np.float64)[None, :]
    return np.abs(cols - centers[:, None]) <= half_width


def _interpolate_masked_vector(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly fill invalid entries; fall back to zero if none are valid."""

    idx = np.arange(values.size)
    if not np.any(valid):
        return np.zeros_like(values, dtype=np.float64)
    if np.count_nonzero(valid) == 1:
        return np.full_like(values, float(values[valid][0]), dtype=np.float64)
    return np.interp(idx, idx[valid], values[valid]).astype(np.float64, copy=False)


def robust_whiten(
    data: np.ndarray,
    params: Optional[CandidateParams] = None,
    clip_sigma: float = 8.0,
    remove_static_bandpass: bool = True,
    bandpass_smooth_channels: int = 0,
) -> np.ndarray:
    """Robustly centre, flatten, and scale a candidate tile.

    Candidate pixels are excluded from all nuisance estimates.  When enabled,
    the time-median bandpass is subtracted before row-wise centring.  This is
    useful for the strong fixed-channel structure visible in the supplied
    station comparison, while masking prevents a low-drift candidate from
    being treated as bandpass.

    ``bandpass_smooth_channels=0`` subtracts the raw robust time median.
    A positive odd value median-smooths that estimate first.
    """

    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("robust_whiten expects time x frequency data")
    if clip_sigma <= 0:
        raise ValueError("clip_sigma must be positive")
    finite = np.isfinite(x)
    finite_values = x[finite]
    fill = float(np.median(finite_values)) if finite_values.size else 0.0
    x = np.where(finite, x, fill)

    mask = np.zeros_like(x, dtype=bool)
    if params is not None:
        mask = candidate_mask(x.shape, params)

    if remove_static_bandpass:
        masked = np.where(mask, np.nan, x)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            bandpass = np.nanmedian(masked, axis=0)
        valid = np.isfinite(bandpass)
        bandpass = _interpolate_masked_vector(bandpass, valid)
        if bandpass_smooth_channels:
            size = int(bandpass_smooth_channels)
            if size < 3:
                raise ValueError("bandpass_smooth_channels must be 0 or >= 3")
            if size % 2 == 0:
                size += 1
            bandpass = median_filter(bandpass, size=size, mode="reflect")
        x = x - bandpass[None, :]

    out = np.empty_like(x)
    for row in range(x.shape[0]):
        good = ~mask[row]
        vals = x[row, good]
        if vals.size == 0:
            vals = x[row]
        centre = float(np.median(vals))
        scale = robust_sigma(vals)
        out[row] = (x[row] - centre) / scale

    out = np.clip(out, -clip_sigma, clip_sigma)
    return out.astype(np.float32, copy=False)


def dechirp(
    data: np.ndarray,
    drift_hz_per_s: float,
    channel_step_hz: float,
    dt_s: float,
    reference_row: float = 0.0,
    order: int = 1,
    return_validity: bool = False,
):
    """Straighten a known drifting signal on either frequency ordering.

    Row ``i`` is shifted by

    ``-(drift * (i-reference_row) * dt) / channel_step``.

    Retaining the signed channel step is essential for descending-frequency
    data; replacing it with ``abs(foff)`` reverses the correction direction.
    """

    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("dechirp expects time x frequency data")
    if not np.isfinite(drift_hz_per_s):
        raise ValueError("drift_hz_per_s must be finite")
    if not np.isfinite(channel_step_hz) or channel_step_hz == 0:
        raise ValueError("channel_step_hz must be finite and non-zero")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")

    out = np.empty_like(x)
    valid = np.empty_like(x, dtype=np.float32)
    ones = np.ones(x.shape[1], dtype=np.float64)
    for row in range(x.shape[0]):
        elapsed_s = (row - reference_row) * dt_s
        shift_channels = -(drift_hz_per_s * elapsed_s) / channel_step_hz
        out[row] = ndi_shift(
            x[row],
            shift=shift_channels,
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        valid[row] = ndi_shift(
            ones,
            shift=shift_channels,
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    out[valid < 0.5] = 0.0
    out = out.astype(np.float32, copy=False)
    if return_validity:
        return out, valid
    return out


def recenter_frequency(
    data: np.ndarray,
    source_channel: float,
    target_channel: Optional[float] = None,
    order: int = 1,
) -> np.ndarray:
    """Shift all rows so a de-chirped candidate sits at a fixed column."""

    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("recenter_frequency expects time x frequency data")
    if target_channel is None:
        target_channel = (x.shape[1] - 1) / 2.0
    shift_channels = float(target_channel) - float(source_channel)
    out = np.empty_like(x)
    for row in range(x.shape[0]):
        out[row] = ndi_shift(
            x[row],
            shift=shift_channels,
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    return out.astype(np.float32, copy=False)


def _frequency_kernel(width_hz: float, df_hz: float, kind: str) -> np.ndarray:
    """Return an odd, symmetric, channel-integrated, unit-L2 kernel.

    A common implementation mistake is to round ``width_hz / df_hz`` and
    fill that many whole channels with equal weights.  At LOFAR's ~2.98 Hz
    resolution this turns, for example, a 10 Hz filter into either 8.94 or
    14.90 Hz depending on the rounding rule.  Here each coefficient is the
    exact overlap of a finite channel with the requested continuous profile.
    Fractional edge channels therefore carry fractional weight, and the
    filter remains centred without requiring an ambiguous even-length shift.

    L2 normalisation makes the output noise variance approximately invariant
    for independent unit-variance channels; it is the matched-filter rather
    than flux-conserving normalisation.
    """

    if not np.isfinite(width_hz) or width_hz <= 0:
        raise ValueError("width_hz must be finite and positive")
    if not np.isfinite(df_hz) or df_hz <= 0:
        raise ValueError("df_hz must be finite and positive")
    kind = str(kind).lower()
    n_fwhm = width_hz / df_hz
    if kind == "boxcar":
        half = max(1, int(np.ceil(0.5 * n_fwhm + 0.5)))
    elif kind == "lorentzian":
        # +/- 8 FWHM contains about 96% of a Lorentzian.  Truncating only the
        # filter (not the injected profile) is an explicit speed/robustness
        # trade-off and is harmless after L2 normalisation.
        half = max(2, int(np.ceil(8.0 * n_fwhm + 0.5)))
    elif kind == "gaussian":
        half = max(2, int(np.ceil(3.0 * n_fwhm + 0.5)))
    else:
        raise ValueError("kind must be 'boxcar', 'lorentzian', or 'gaussian'")

    centres = np.arange(-half, half + 1, dtype=np.float64) * df_hz
    lower = centres - 0.5 * df_hz
    upper = centres + 0.5 * df_hz
    if kind == "boxcar":
        cdf_lo = np.clip(lower / width_hz + 0.5, 0.0, 1.0)
        cdf_hi = np.clip(upper / width_hz + 0.5, 0.0, 1.0)
    elif kind == "lorentzian":
        cdf_lo = 0.5 + np.arctan(2.0 * lower / width_hz) / np.pi
        cdf_hi = 0.5 + np.arctan(2.0 * upper / width_hz) / np.pi
    else:  # gaussian
        sigma = width_hz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        cdf_lo = ndtr(lower / sigma)
        cdf_hi = ndtr(upper / sigma)
    kernel = np.maximum(cdf_hi - cdf_lo, 0.0)

    norm = float(np.linalg.norm(kernel))
    if not np.isfinite(norm) or norm < _EPS:
        raise ValueError("frequency kernel is degenerate")
    return kernel / norm


def frequency_integrate(
    data: np.ndarray,
    width_hz: float,
    df_hz: float,
    kind: str = "boxcar",
) -> np.ndarray:
    """Apply noise-normalised frequency scrunching without edge wrapping.

    The output has the same shape as the input for checkpoint compatibility.
    ``scipy.signal.convolve(..., mode='same')`` uses zero padding, appropriate
    after whitening because zero is the background mean.  It does not repeat
    edge RFI.
    """

    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("frequency_integrate expects time x frequency data")
    kernel = _frequency_kernel(width_hz, df_hz, kind)[None, :]
    out = convolve(x, kernel, mode="same", method="auto")
    return out.astype(np.float32, copy=False)


def make_candidate_view(
    data: np.ndarray,
    params: CandidateParams,
    integration: str = "boxcar",
    dechirp_order: int = 1,
    recenter: bool = True,
    clip_sigma: float = 8.0,
    remove_static_bandpass: bool = True,
) -> np.ndarray:
    """Create the fixed-shape view consumed by the Stage-4 coincidence model.

    ``integration='none'`` performs whitening, de-chirping, and centring only.
    ``'boxcar'`` is the primary candidate-informed experiment;
    ``'lorentzian'`` and ``'gaussian'`` are controlled ablations.
    """

    x = np.asarray(data)
    params.validate(x.shape)
    whitened = robust_whiten(
        x,
        params=params,
        clip_sigma=clip_sigma,
        remove_static_bandpass=remove_static_bandpass,
    )
    straight = dechirp(
        whitened,
        drift_hz_per_s=params.drift_hz_per_s,
        channel_step_hz=params.channel_step_hz,
        dt_s=params.dt_s,
        reference_row=params.reference_row,
        order=dechirp_order,
    )
    if recenter:
        straight = recenter_frequency(
            straight,
            source_channel=params.center_channel,
            target_channel=(straight.shape[1] - 1) / 2.0,
            order=dechirp_order,
        )
    integration = str(integration).lower()
    if integration in ("none", "raw", "dechirp_only"):
        out = straight
    else:
        out = frequency_integrate(
            straight,
            width_hz=params.width_hz,
            df_hz=abs(params.channel_step_hz),
            kind=integration,
        )
    # Preserve the physically meaningful filter gain: clip only; do not divide
    # by the post-filter tile standard deviation here.  The filter is unit-L2.
    return np.clip(out, -clip_sigma, clip_sigma).astype(np.float32, copy=False)


def centre_column_statistic(view: np.ndarray) -> float:
    """Time-summed central response, with correct even-grid centring.

    A 1024-column view is re-centred at 511.5, so the candidate lies equally
    between columns 511 and 512.  Summing that pair and dividing by
    ``sqrt(2 * n_time)`` is the unit-noise statistic; selecting only column
    512 would throw away half of a perfectly centred unresolved response.
    """

    x = np.asarray(view, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("view must be two-dimensional")
    centre = x.shape[1] // 2
    if x.shape[1] % 2 == 0:
        response = np.sum(x[:, centre - 1 : centre + 1])
        variance_pixels = 2 * x.shape[0]
    else:
        response = np.sum(x[:, centre])
        variance_pixels = x.shape[0]
    return float(response / np.sqrt(max(variance_pixels, 1)))


def _inject_test_track(
    shape: Tuple[int, int],
    params: CandidateParams,
    amplitude: float = 12.0,
) -> np.ndarray:
    """Small synthetic drifting line used only by the self-test."""

    out = np.zeros(shape, dtype=np.float64)
    for row, centre in enumerate(track_channels(params, shape[0])):
        lo = int(np.floor(centre))
        frac = centre - lo
        if 0 <= lo < shape[1]:
            out[row, lo] += amplitude * (1.0 - frac)
        if 0 <= lo + 1 < shape[1]:
            out[row, lo + 1] += amplitude * frac
    return out


def _self_test() -> None:
    rng = np.random.default_rng(1234)
    shape = (32, 256)
    dt_s = 0.67108864
    drift = 2.5

    print("candidate_preprocessing.py self-test")
    for step in (2.980232239, -2.980232239):
        params = CandidateParams(110.0, drift, 3.0, step, dt_s)
        signal = _inject_test_track(shape, params)
        corrected = dechirp(signal, drift, step, dt_s)
        peaks = np.argmax(corrected, axis=1)
        if np.std(peaks) >= 0.75:
            raise AssertionError(
                "de-chirp failed for channel_step_hz=%s (peak std %.3f)"
                % (step, np.std(peaks))
            )
        wrong = dechirp(signal, drift, -step, dt_s)
        if np.std(np.argmax(wrong, axis=1)) <= 2.0:
            raise AssertionError(
                "wrong frequency ordering spuriously straightened track"
            )
        print("  PASS signed de-chirp: channel_step_hz=%+.6f" % step)

    # Frequency integration gain on a width-matched top-hat in white noise.
    n = 11
    raw_stats = []
    filtered_stats = []
    for _ in range(300):
        noise = rng.normal(size=shape)
        signal = np.zeros(shape)
        c = shape[1] // 2
        signal[:, c - n // 2 : c + n // 2 + 1] = 0.35
        raw_stats.append(np.mean((noise + signal)[:, c]))
        filt = frequency_integrate(
            noise + signal, n * abs(2.980232239), abs(2.980232239), kind="boxcar"
        )
        filtered_stats.append(np.mean(filt[:, c]))
    gain = np.mean(filtered_stats) / max(np.mean(raw_stats), _EPS)
    if not (2.5 < gain < 4.2):
        raise AssertionError("unexpected boxcar gain %.3f; expected ~sqrt(11)" % gain)
    print(
        "  PASS L2-normalised scrunch gain: %.3fx (sqrt(11)=%.3fx)"
        % (gain, np.sqrt(11.0))
    )
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
