"""Reproducible candidate-pair generation for Stage-4 fine-tuning.

The factory uses real memory-mapped LOFAR backgrounds but performs injection
and S/N calibration with :mod:`channelized_injection`.  It deliberately keeps
the post-detection (``detected``) and fixed-power (``power``) populations
separate.  Frequency partitions are disjoint for train/validation/test to
reduce background leakage even when only one Sweden-accessible file is
available.

No Irish compute node is required.  A second accessible filterbank may be
provided as an independent-noise proxy; if omitted, station B is sampled from
a different location in the same Sweden file.  Set ``station_b_is_proxy=False``
only for a genuinely different telescope station. Proxy runs must be described
as synthetic dual-station coincidence on real Sweden backgrounds, not as real
Ireland--Sweden validation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from candidate_preprocessing import CandidateParams, make_candidate_view
from channelized_injection import (
    ProfileTruncationError,
    inject_broadened_signal,
    make_temporal_envelope,
    safe_center_channel_interval,
)

try:  # Allows pure data helpers to be imported on a login node without torch.
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised only in lightweight QA envs.
    torch = None
    Dataset = object


SPLIT_FRACTIONS = {
    "train": (0.00, 0.70),
    "val": (0.72, 0.85),
    "test": (0.87, 1.00),
}

DEFAULT_CASE_PROBS = {
    "match": 0.50,
    "onesided": 0.19,
    "independent": 0.19,
    # Retain substantial exposure to the difficult no-injection/RFI negative.
    "noise": 0.12,
}

DEFAULT_SHAPE_PROBS = {
    "lorentzian": 0.60,
    "box": 0.20,
    "gaussian": 0.20,
}


@dataclass
class PairRecord:
    seed: int
    split: str
    case: str
    label: int
    true_shape: str
    true_width_hz: float
    true_drift_hz_per_s: float
    true_center_channel_a: float
    true_reference_row: float
    true_shape_b: Optional[str]
    true_width_hz_b: Optional[float]
    true_drift_hz_per_s_b: Optional[float]
    true_center_channel_b: Optional[float]
    snr_mode: str
    target_snr_a: float
    target_snr_b: Optional[float]
    expected_broadened_snr_a: Optional[float]
    expected_broadened_snr_b: Optional[float]
    in_tile_energy_fraction_min_a: Optional[float]
    in_tile_energy_fraction_min_b: Optional[float]
    reported_width_hz_a: float
    reported_width_hz_b: float
    reported_drift_hz_per_s_a: float
    reported_drift_hz_per_s_b: float
    reported_center_channel_a: float
    reported_center_channel_b: float
    background_start_a: Tuple[int, int]
    background_start_b: Tuple[int, int]
    station_b_proxy: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _normalise_probs(
    probs: Dict[str, float], allowed: Sequence[str]
) -> Tuple[list, np.ndarray]:
    unknown = set(probs) - set(allowed)
    if unknown:
        raise ValueError("unknown probability keys: %s" % sorted(unknown))
    keys = [key for key in allowed if probs.get(key, 0.0) > 0]
    if not keys:
        raise ValueError("at least one probability must be positive")
    values = np.asarray([float(probs[key]) for key in keys], dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probabilities must be finite and non-negative")
    return keys, values / values.sum()


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    if low <= 0 or high < low:
        raise ValueError("log-uniform bounds must satisfy 0 < low <= high")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _drift_interval_for_track(
    center_channel: float,
    n_rows: int,
    reference_row: float,
    dt_s: float,
    channel_step_hz: float,
    safe_center_lo: float,
    safe_center_hi: float,
    max_abs_drift_hz_s: float,
) -> Tuple[float, float]:
    """Physical-drift interval whose complete track remains profile-safe."""

    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if dt_s <= 0 or not np.isfinite(dt_s):
        raise ValueError("dt_s must be finite and positive")
    if channel_step_hz == 0 or not np.isfinite(channel_step_hz):
        raise ValueError("channel_step_hz must be finite and non-zero")
    if max_abs_drift_hz_s < 0 or not np.isfinite(max_abs_drift_hz_s):
        raise ValueError("max_abs_drift_hz_s must be finite and non-negative")
    if not (safe_center_lo <= center_channel <= safe_center_hi):
        raise ValueError("center_channel is outside the safe profile interval")

    lower = -float(max_abs_drift_hz_s)
    upper = float(max_abs_drift_hz_s)
    scales = (
        (np.arange(n_rows, dtype=np.float64) - float(reference_row))
        * float(dt_s)
        / float(channel_step_hz)
    )
    # The centre is affine in row, so constraints at the extrema suffice.
    for scale in (float(np.min(scales)), float(np.max(scales))):
        if abs(scale) < 1e-15:
            continue
        bound_a = (safe_center_lo - center_channel) / scale
        bound_b = (safe_center_hi - center_channel) / scale
        lower = max(lower, min(bound_a, bound_b))
        upper = min(upper, max(bound_a, bound_b))
    if lower > upper + 1e-12:
        raise ValueError(
            "no drift can keep the complete candidate track inside the cutout"
        )
    # Collapse sub-ulp inversions to their common boundary.
    if lower > upper:
        lower = upper = 0.5 * (lower + upper)
    return float(lower), float(upper)


def _joint_drift_interval(
    center_channel: float,
    track_specs: Sequence[Tuple[int, float, float, float, float, float]],
    max_abs_drift_hz_s: float,
) -> Tuple[float, float]:
    """Intersect containment-compatible drift intervals for all stations."""

    lower = -float(max_abs_drift_hz_s)
    upper = float(max_abs_drift_hz_s)
    for n_rows, reference_row, dt_s, channel_step_hz, safe_lo, safe_hi in track_specs:
        station_lo, station_hi = _drift_interval_for_track(
            center_channel,
            int(n_rows),
            reference_row,
            dt_s,
            channel_step_hz,
            safe_lo,
            safe_hi,
            max_abs_drift_hz_s,
        )
        lower = max(lower, station_lo)
        upper = min(upper, station_hi)
    if lower > upper + 1e-12:
        raise ValueError(
            "station geometries have no common containment-compatible drift"
        )
    if lower > upper:
        lower = upper = 0.5 * (lower + upper)
    return float(lower), float(upper)


def _full_drift_center_interval(
    safe_lo: float,
    safe_hi: float,
    n_rows: int,
    reference_row: float,
    dt_s: float,
    channel_step_hz: float,
    max_abs_drift_hz_s: float,
) -> Tuple[float, float]:
    """Centre range supporting the entire configured symmetric drift range."""

    scales = (
        (np.arange(n_rows, dtype=np.float64) - float(reference_row))
        * float(dt_s)
        / float(channel_step_hz)
    )
    displacement = float(np.max(np.abs(scales))) * float(max_abs_drift_hz_s)
    return float(safe_lo + displacement), float(safe_hi - displacement)


def _sample_separated_center(
    rng: np.random.Generator,
    anchor: float,
    safe_lo: float,
    safe_hi: float,
    minimum_offset_channels: float,
    n_channels: int,
) -> float:
    """Sample a valid unrelated-event centre away from the union anchor."""

    if safe_lo > safe_hi:
        raise ValueError("safe centre interval is empty")
    if minimum_offset_channels < 0:
        raise ValueError("minimum_offset_channels must be non-negative")

    def segments(lo: float, hi: float):
        result = []
        left_hi = min(hi, anchor - minimum_offset_channels)
        if left_hi >= lo:
            result.append((lo, left_hi))
        right_lo = max(lo, anchor + minimum_offset_channels)
        if hi >= right_lo:
            result.append((right_lo, hi))
        return result

    # Preserve the former central 30--70% prior when geometry permits, but
    # expand to the exact safe interval rather than clipping an unsafe draw.
    preferred_lo = max(safe_lo, 0.30 * (n_channels - 1.0))
    preferred_hi = min(safe_hi, 0.70 * (n_channels - 1.0))
    choices = segments(preferred_lo, preferred_hi)
    if not choices:
        choices = segments(safe_lo, safe_hi)
    if not choices:
        raise ValueError(
            "cutout is too narrow to place an independent event at the "
            "requested minimum offset"
        )
    lengths = np.asarray([max(hi - lo, 1e-12) for lo, hi in choices], dtype=np.float64)
    selected = int(rng.choice(len(choices), p=lengths / np.sum(lengths)))
    lo, hi = choices[selected]
    return float(lo if hi <= lo else rng.uniform(lo, hi))


class CandidatePairFactory:
    """Generate one labelled, candidate-conditioned pair at a time."""

    def __init__(
        self,
        mode: str,
        station_a_filterbank: str,
        station_b_filterbank: Optional[str] = None,
        station_b_is_proxy: bool = True,
        split: str = "train",
        widths_hz: Tuple[float, float] = (10.0, 100.0),
        target_snr_range: Tuple[float, float] = (8.0, 30.0),
        snr_mode: str = "detected",
        integration: str = "boxcar",
        case_probs: Optional[Dict[str, float]] = None,
        shape_probs: Optional[Dict[str, float]] = None,
        max_abs_drift_hz_s: float = 4.0,
        width_log_error_sigma: float = 0.15,
        drift_error_channels_per_tile: float = 0.35,
        center_error_channels: float = 0.35,
        station_snr_ratio_range: Tuple[float, float] = (0.5, 2.0),
        scintillation_log_sigma: float = 0.12,
        station_modulation_log_sigma: float = 0.06,
        remove_static_bandpass: bool = True,
    ):
        if split not in SPLIT_FRACTIONS:
            raise ValueError("split must be one of %s" % sorted(SPLIT_FRACTIONS))
        if snr_mode not in ("detected", "power"):
            raise ValueError("snr_mode must be 'detected' or 'power'")
        if not (0 < widths_hz[0] <= widths_hz[1]):
            raise ValueError("invalid widths_hz")
        if not (0 < target_snr_range[0] <= target_snr_range[1]):
            raise ValueError("invalid target_snr_range")
        if not (0 < station_snr_ratio_range[0] <= station_snr_ratio_range[1]):
            raise ValueError("invalid station_snr_ratio_range")
        if width_log_error_sigma < 0 or drift_error_channels_per_tile < 0:
            raise ValueError("metadata-error scales must be non-negative")
        if center_error_channels < 0:
            raise ValueError("center_error_channels must be non-negative")
        if scintillation_log_sigma < 0 or station_modulation_log_sigma < 0:
            raise ValueError("modulation scales must be non-negative")
        if not np.isfinite(max_abs_drift_hz_s) or max_abs_drift_hz_s < 0:
            raise ValueError("max_abs_drift_hz_s must be finite and non-negative")

        # Importing train loads torch/setigen/blimpy; defer it until a real
        # factory is constructed so preprocessing self-tests stay lightweight.
        from train import CONFIGS, load_background

        if mode not in CONFIGS:
            raise ValueError("unknown mode %r" % mode)
        self.mode = mode
        self.tile_shape = tuple(CONFIGS[mode]["frame_shape"])
        self.split = split
        self.widths_hz = tuple(map(float, widths_hz))
        self.target_snr_range = tuple(map(float, target_snr_range))
        self.snr_mode = snr_mode
        self.integration = integration
        self.max_abs_drift_hz_s = float(max_abs_drift_hz_s)
        self.width_log_error_sigma = float(width_log_error_sigma)
        self.drift_error_channels_per_tile = float(drift_error_channels_per_tile)
        self.center_error_channels = float(center_error_channels)
        self.station_snr_ratio_range = tuple(map(float, station_snr_ratio_range))
        self.scintillation_log_sigma = float(scintillation_log_sigma)
        self.station_modulation_log_sigma = float(station_modulation_log_sigma)
        self.remove_static_bandpass = bool(remove_static_bandpass)

        self.case_names, self.case_p = _normalise_probs(
            case_probs or DEFAULT_CASE_PROBS,
            ("match", "onesided", "independent", "noise"),
        )
        self.shape_names, self.shape_p = _normalise_probs(
            shape_probs or DEFAULT_SHAPE_PROBS,
            ("lorentzian", "box", "gaussian"),
        )

        self.real_a, self.header_a = load_background(station_a_filterbank)
        if station_b_filterbank:
            self.real_b, self.header_b = load_background(station_b_filterbank)
            self.station_b_proxy = bool(station_b_is_proxy)
        else:
            self.real_b, self.header_b = self.real_a, self.header_a
            self.station_b_proxy = True

        self.step_a = float(self.header_a["foff"]) * 1e6
        self.step_b = float(self.header_b["foff"]) * 1e6
        self.dt_a = float(self.header_a["tsamp"])
        self.dt_b = float(self.header_b["tsamp"])
        if self.step_a == 0 or self.step_b == 0:
            raise ValueError("filterbank foff must be non-zero")
        if self.dt_a <= 0 or self.dt_b <= 0:
            raise ValueError("filterbank tsamp must be positive")
        for name, data in (("station A", self.real_a), ("station B", self.real_b)):
            if data.ndim != 2:
                raise ValueError("%s background must be 2D after load" % name)
            if data.shape[0] < self.tile_shape[0] or data.shape[1] < self.tile_shape[1]:
                raise ValueError("%s background is smaller than frame_shape" % name)

    def _sample_background(
        self, data: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        t_size, f_size = self.tile_shape
        t_start = int(rng.integers(0, data.shape[0] - t_size + 1))
        frac_lo, frac_hi = SPLIT_FRACTIONS[self.split]
        f_lo = int(math.floor(frac_lo * data.shape[1]))
        f_hi = int(math.floor(frac_hi * data.shape[1]))
        f_lo = min(max(f_lo, 0), data.shape[1] - f_size)
        f_hi = min(max(f_hi, f_lo + f_size), data.shape[1])
        max_start = f_hi - f_size
        if max_start < f_lo:
            raise ValueError("frequency partition is too small for frame_shape")
        f_start = int(rng.integers(f_lo, max_start + 1))
        tile = np.asarray(
            data[t_start : t_start + t_size, f_start : f_start + f_size],
            dtype=np.float32,
        ).copy()
        return tile, (t_start, f_start)

    def _reported_params(
        self,
        true_params: CandidateParams,
        rng: np.random.Generator,
        dt_s: float,
    ) -> CandidateParams:
        width = true_params.width_hz * float(
            np.exp(rng.normal(0.0, self.width_log_error_sigma))
        )
        tile_duration = max((self.tile_shape[0] - 1) * dt_s, dt_s)
        drift_sigma = (
            self.drift_error_channels_per_tile
            * abs(true_params.channel_step_hz)
            / tile_duration
        )
        drift = true_params.drift_hz_per_s + float(rng.normal(0.0, drift_sigma))
        centre = true_params.center_channel + float(
            rng.normal(0.0, self.center_error_channels)
        )
        centre = float(np.clip(centre, 0.0, self.tile_shape[1] - 1.0))
        return CandidateParams(
            center_channel=centre,
            drift_hz_per_s=drift,
            width_hz=max(width, 0.25 * abs(true_params.channel_step_hz)),
            channel_step_hz=true_params.channel_step_hz,
            dt_s=dt_s,
            reference_row=true_params.reference_row,
        )

    def sample(
        self,
        seed: int,
        fixed_width_hz: Optional[float] = None,
        fixed_shape: Optional[str] = None,
        fixed_case: Optional[str] = None,
        return_raw: bool = False,
    ) -> Dict[str, object]:
        rng = np.random.default_rng(int(seed))
        tile_a, start_a = self._sample_background(self.real_a, rng)
        tile_b, start_b = self._sample_background(self.real_b, rng)

        case = fixed_case or str(rng.choice(self.case_names, p=self.case_p))
        if case not in ("match", "onesided", "independent", "noise"):
            raise ValueError("unknown fixed_case %r" % case)
        label = 1 if case == "match" else 0
        profile_shape = fixed_shape or str(rng.choice(self.shape_names, p=self.shape_p))
        if profile_shape not in ("lorentzian", "box", "gaussian"):
            raise ValueError("unknown fixed_shape %r" % profile_shape)
        width = (
            float(fixed_width_hz)
            if fixed_width_hz is not None
            else _log_uniform(rng, self.widths_hz[0], self.widths_hz[1])
        )
        if not np.isfinite(width) or width <= 0:
            raise ValueError("fixed_width_hz must be finite and positive")

        # Define the injected centre at mid-observation.  Relative to a row-zero
        # reference this halves the maximum excursion inside a fixed cutout,
        # while representing exactly the same physical linear drift.  The
        # actual safe interval is then solved from the analytic profile CDF and
        # each station's signed foff/tsamp, rather than assumed from mock data.
        reference_row = (self.tile_shape[0] - 1.0) / 2.0
        safe_a = safe_center_channel_interval(
            self.tile_shape[1], width, abs(self.step_a), profile_shape
        )
        safe_b = safe_center_channel_interval(
            self.tile_shape[1], width, abs(self.step_b), profile_shape
        )
        base_center_lo = max(safe_a[0], safe_b[0])
        base_center_hi = min(safe_a[1], safe_b[1])
        if base_center_lo > base_center_hi:
            raise ValueError(
                "station cutouts have no common safe centre for width %.3f Hz" % width
            )

        full_a = _full_drift_center_interval(
            safe_a[0],
            safe_a[1],
            self.tile_shape[0],
            reference_row,
            self.dt_a,
            self.step_a,
            self.max_abs_drift_hz_s,
        )
        full_b = _full_drift_center_interval(
            safe_b[0],
            safe_b[1],
            self.tile_shape[0],
            reference_row,
            self.dt_b,
            self.step_b,
            self.max_abs_drift_hz_s,
        )
        full_center_lo = max(full_a[0], full_b[0])
        full_center_hi = min(full_a[1], full_b[1])
        if full_center_lo <= full_center_hi:
            center_lo, center_hi = full_center_lo, full_center_hi
        else:
            # Extremely coarse/narrow cutouts may not support the complete
            # configured drift prior.  Keep the profile valid and explicitly
            # condition the drift draw on what this geometry can represent.
            center_lo, center_hi = base_center_lo, base_center_hi
        nominal_center = (self.tile_shape[1] - 1.0) / 2.0 + float(
            rng.uniform(-0.45, 0.45)
        )
        centre = float(np.clip(nominal_center, center_lo, center_hi))
        drift_lo, drift_hi = _joint_drift_interval(
            centre,
            (
                (
                    self.tile_shape[0],
                    reference_row,
                    self.dt_a,
                    self.step_a,
                    safe_a[0],
                    safe_a[1],
                ),
                (
                    self.tile_shape[0],
                    reference_row,
                    self.dt_b,
                    self.step_b,
                    safe_b[0],
                    safe_b[1],
                ),
            ),
            self.max_abs_drift_hz_s,
        )
        drift = float(
            drift_lo if drift_hi <= drift_lo else rng.uniform(drift_lo, drift_hi)
        )
        target_a = float(rng.uniform(*self.target_snr_range))
        ratio = _log_uniform(rng, *self.station_snr_ratio_range)
        target_b = float(np.clip(target_a * ratio, *self.target_snr_range))

        true_a = CandidateParams(
            centre,
            drift,
            width,
            self.step_a,
            self.dt_a,
            reference_row=reference_row,
        )
        true_b = CandidateParams(
            centre,
            drift,
            width,
            self.step_b,
            self.dt_b,
            reference_row=reference_row,
        )
        shared_env = make_temporal_envelope(
            self.tile_shape[0], rng, log_sigma=self.scintillation_log_sigma
        )
        # The astronomical modulation is shared, but independent receiver /
        # propagation fluctuations prevent the positive class from becoming
        # an unrealistically pixel-identical template-matching problem.
        station_env_a = shared_env * make_temporal_envelope(
            self.tile_shape[0], rng, log_sigma=self.station_modulation_log_sigma
        )
        station_env_b = shared_env * make_temporal_envelope(
            self.tile_shape[0], rng, log_sigma=self.station_modulation_log_sigma
        )
        station_env_a = station_env_a / np.mean(station_env_a)
        station_env_b = station_env_b / np.mean(station_env_b)

        inj_a = inj_b = None
        if case in ("match", "onesided", "independent"):
            tile_a, _, inj_a = inject_broadened_signal(
                tile_a,
                true_a,
                target_a,
                profile_shape=profile_shape,
                snr_mode=self.snr_mode,
                rng=rng,
                temporal_envelope=station_env_a,
            )
        if case == "match":
            tile_b, _, inj_b = inject_broadened_signal(
                tile_b,
                true_b,
                target_b,
                profile_shape=profile_shape,
                snr_mode=self.snr_mode,
                rng=rng,
                temporal_envelope=station_env_b,
            )
        elif case == "independent":
            # An unrelated local event: different location, drift, width, and
            # temporal modulation.  It is deliberately preprocessed using the
            # station-A union candidate below, so it should not align.
            independent_width = _log_uniform(rng, self.widths_hz[0], self.widths_hz[1])
            independent_shape = str(rng.choice(self.shape_names, p=self.shape_p))
            independent_safe = safe_center_channel_interval(
                self.tile_shape[1],
                independent_width,
                abs(self.step_b),
                independent_shape,
            )
            min_offset = max(12.0, 4.0 * width / abs(self.step_b))

            # Prefer centres that support the full drift prior.  If the cutout
            # cannot satisfy both that and the hard-negative frequency offset,
            # fall back to the exact profile-safe interval and restrict only the
            # independent event's drift to the representable range.
            independent_full = _full_drift_center_interval(
                independent_safe[0],
                independent_safe[1],
                self.tile_shape[0],
                reference_row,
                self.dt_b,
                self.step_b,
                self.max_abs_drift_hz_s,
            )
            try:
                independent_centre = _sample_separated_center(
                    rng,
                    centre,
                    independent_full[0],
                    independent_full[1],
                    min_offset,
                    self.tile_shape[1],
                )
            except ValueError:
                independent_centre = _sample_separated_center(
                    rng,
                    centre,
                    independent_safe[0],
                    independent_safe[1],
                    min_offset,
                    self.tile_shape[1],
                )
            independent_drift_lo, independent_drift_hi = _drift_interval_for_track(
                independent_centre,
                self.tile_shape[0],
                reference_row,
                self.dt_b,
                self.step_b,
                independent_safe[0],
                independent_safe[1],
                self.max_abs_drift_hz_s,
            )
            independent_drift = float(
                independent_drift_lo
                if independent_drift_hi <= independent_drift_lo
                else rng.uniform(independent_drift_lo, independent_drift_hi)
            )
            independent_params = CandidateParams(
                independent_centre,
                independent_drift,
                independent_width,
                self.step_b,
                self.dt_b,
                reference_row=reference_row,
            )
            independent_env = make_temporal_envelope(
                self.tile_shape[0], rng, log_sigma=self.scintillation_log_sigma
            )
            tile_b, _, inj_b = inject_broadened_signal(
                tile_b,
                independent_params,
                target_b,
                profile_shape=independent_shape,
                snr_mode=self.snr_mode,
                rng=rng,
                temporal_envelope=independent_env,
            )

        reported_a = self._reported_params(true_a, rng, self.dt_a)
        # In the real union pipeline station B may not have its own BLISS hit;
        # use the union anchor (same physical hypothesis) with independent
        # measurement error, not the unrelated event's true parameters.
        reported_b_anchor = CandidateParams(
            centre,
            drift,
            width,
            self.step_b,
            self.dt_b,
            reference_row=reference_row,
        )
        reported_b = self._reported_params(reported_b_anchor, rng, self.dt_b)

        view_a = make_candidate_view(
            tile_a,
            reported_a,
            integration=self.integration,
            remove_static_bandpass=self.remove_static_bandpass,
        )
        view_b = make_candidate_view(
            tile_b,
            reported_b,
            integration=self.integration,
            remove_static_bandpass=self.remove_static_bandpass,
        )

        record = PairRecord(
            seed=int(seed),
            split=self.split,
            case=case,
            label=label,
            true_shape=profile_shape,
            true_width_hz=width,
            true_drift_hz_per_s=drift,
            true_center_channel_a=centre,
            true_reference_row=reference_row,
            true_shape_b=(str(inj_b.shape) if inj_b is not None else None),
            true_width_hz_b=(float(inj_b.width_hz) if inj_b is not None else None),
            true_drift_hz_per_s_b=(
                float(inj_b.drift_hz_per_s) if inj_b is not None else None
            ),
            true_center_channel_b=(
                float(inj_b.center_channel) if inj_b is not None else None
            ),
            snr_mode=self.snr_mode,
            target_snr_a=target_a,
            target_snr_b=target_b if inj_b is not None else None,
            expected_broadened_snr_a=(
                float(inj_a.expected_broadened_snr) if inj_a is not None else None
            ),
            expected_broadened_snr_b=(
                float(inj_b.expected_broadened_snr) if inj_b is not None else None
            ),
            in_tile_energy_fraction_min_a=(
                float(inj_a.in_tile_energy_fraction_min) if inj_a is not None else None
            ),
            in_tile_energy_fraction_min_b=(
                float(inj_b.in_tile_energy_fraction_min) if inj_b is not None else None
            ),
            reported_width_hz_a=reported_a.width_hz,
            reported_width_hz_b=reported_b.width_hz,
            reported_drift_hz_per_s_a=reported_a.drift_hz_per_s,
            reported_drift_hz_per_s_b=reported_b.drift_hz_per_s,
            reported_center_channel_a=reported_a.center_channel,
            reported_center_channel_b=reported_b.center_channel,
            background_start_a=start_a,
            background_start_b=start_b,
            station_b_proxy=self.station_b_proxy,
        )
        result = {
            "view_a": view_a,
            "view_b": view_b,
            "label": label,
            "record": record,
            "params_a": reported_a,
            "params_b": reported_b,
        }
        if return_raw:
            result["raw_a"] = tile_a.astype(np.float32, copy=False)
            result["raw_b"] = tile_b.astype(np.float32, copy=False)
        return result


class Stage4PairDataset(Dataset):
    """Torch Dataset wrapper with deterministic, epoch-varying samples."""

    def __init__(
        self,
        factory: CandidatePairFactory,
        n_samples: int,
        seed: int,
        max_generation_attempts: int = 8,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for Stage4PairDataset")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if max_generation_attempts <= 0:
            raise ValueError("max_generation_attempts must be positive")
        self.factory = factory
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.max_generation_attempts = int(max_generation_attempts)
        self.epoch = 0

    def __len__(self) -> int:
        return self.n_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        # Large coprime strides make (epoch, index, split) map to stable,
        # non-overlapping generator seeds without relying on global RNG state.
        split_code = {"train": 11, "val": 23, "test": 37}[self.factory.split]
        sample_seed = (
            self.seed + 1_000_003 * self.epoch + 97_409 * int(index) + split_code
        ) % (2**63 - 1)
        item = None
        last_error = None
        # Geometry-aware sampling should make the first attempt valid.  This
        # narrow retry guard protects long runs from a future profile/numerical
        # edge case without hiding unrelated data or programming errors.
        for attempt in range(self.max_generation_attempts):
            attempt_seed = (sample_seed + 1_299_709 * attempt) % (2**63 - 1)
            try:
                item = self.factory.sample(attempt_seed)
                break
            except ProfileTruncationError as exc:
                last_error = exc
        if item is None:
            raise RuntimeError(
                "failed to generate a contained Stage-4 sample after %d "
                "deterministic attempts; check width/drift/cutout geometry"
                % self.max_generation_attempts
            ) from last_error
        view_a = torch.from_numpy(item["view_a"][None, ...])
        view_b = torch.from_numpy(item["view_b"][None, ...])
        label = torch.tensor(float(item["label"]), dtype=torch.float32)
        return view_a, view_b, label
