#!/usr/bin/env python3
"""Physical-coverage checks for real two-station BLISS candidate unions.

The absence of a hit at one station is informative only when that station had
data *and* its BLISS search examined the projected track.  Naoise's blind hit
finder flags a configurable rolloff fraction at both edges of every coarse
channel.  Those gaps are therefore distinguished from genuine one-station
detections throughout the real-pair workflow.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from lofts_bliss_schema import SECONDS_PER_DAY, CandidateRecord, ObservationRecord


def observation_frequency_bounds_hz(
    observation: ObservationRecord,
) -> Tuple[float, float]:
    """Return inclusive channel-centre bounds in ascending physical order."""

    first = observation.frequency_hz_for_channel(0.0)
    last = observation.frequency_hz_for_channel(float(observation.n_channels - 1))
    return min(first, last), max(first, last)


def common_frequency_bounds_hz(
    observations: Sequence[ObservationRecord],
) -> Tuple[float, float]:
    if len(observations) != 2:
        raise ValueError("exactly two observations are required")
    bounds = [observation_frequency_bounds_hz(item) for item in observations]
    low = max(item[0] for item in bounds)
    high = min(item[1] for item in bounds)
    if not low < high:
        raise ValueError("the two observations have no common frequency coverage")
    return float(low), float(high)


def _candidate_frequency_at_observation_row(
    candidate: CandidateRecord,
    observation: ObservationRecord,
    row: float,
) -> float:
    if candidate.frequency_ref_mjd is not None:
        mjd = observation.start_mjd + float(row) * observation.tsamp_s / SECONDS_PER_DAY
        return candidate.frequency_at_mjd(mjd)
    offset_s = float(row) * observation.tsamp_s
    return candidate.frequency_at_offset_s(offset_s)


def candidate_track_channels(
    candidate: CandidateRecord,
    observation: ObservationRecord,
    rows: Optional[Iterable[float]] = None,
) -> np.ndarray:
    if rows is None:
        rows = range(observation.n_time)
    frequencies = [
        _candidate_frequency_at_observation_row(candidate, observation, row)
        for row in rows
    ]
    return np.asarray(
        [observation.channel_for_frequency_hz(value) for value in frequencies],
        dtype=float,
    )


def _coarse_clean_margin(
    channel: float,
    observation: ObservationRecord,
) -> Tuple[float, int, float, float]:
    """Return clean-band margin, coarse index, clean low and clean high.

    A positive margin means the channel centre is inside the BLISS clean
    region.  The clean-high coordinate is exclusive, matching Python/BLISS
    slicing conventions.
    """

    fpc = int(observation.search_fine_channels_per_coarse)
    if fpc <= 0:
        raise ValueError("search_fine_channels_per_coarse is not recorded")
    coarse_index = int(math.floor(float(channel) / fpc))
    if coarse_index < 0:
        return -math.inf, coarse_index, math.nan, math.nan
    coarse_start = coarse_index * fpc
    coarse_width = min(fpc, observation.n_channels - coarse_start)
    if coarse_width <= 0:
        return -math.inf, coarse_index, math.nan, math.nan
    flag = int(round(coarse_width * observation.search_rolloff_fraction))
    clean_low = float(coarse_start + flag)
    clean_high = float(coarse_start + coarse_width - flag)
    # ``clean_high`` is exclusive. Candidate frequencies are channel-centre
    # coordinates, so the final searched centre is ``clean_high - 1``. Without
    # this subtraction a zero-width coverage audit would incorrectly classify
    # the first rolloff channel as searched.
    margin = min(
        float(channel) - clean_low,
        (clean_high - 1.0) - float(channel),
    )
    return float(margin), coarse_index, clean_low, clean_high


def search_coverage_audit(
    candidate: CandidateRecord,
    observation: ObservationRecord,
    guard_fwhm_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Audit whether an entire projected candidate track was actually searched.

    The guard is expressed as a fraction of the candidate FWHM on each side of
    the centre.  This is stricter than a centre-only check and prevents a
    counterpart near a rolloff boundary from being called a meaningful BLISS
    non-detection.
    """

    if guard_fwhm_fraction < 0:
        raise ValueError("guard_fwhm_fraction must be non-negative")
    channels = candidate_track_channels(candidate, observation)
    half_guard_channels = (
        float(guard_fwhm_fraction)
        * float(candidate.width_hz)
        / abs(float(observation.signed_foff_hz))
    )
    full_margin = float(
        min(np.min(channels), (observation.n_channels - 1) - np.max(channels))
    )
    full_data_covered = bool(full_margin >= half_guard_channels)

    drift_in_search = True
    if observation.search_drift_min_hz_s is not None:
        drift_in_search = bool(
            float(observation.search_drift_min_hz_s)
            <= float(candidate.drift_hz_s)
            <= float(observation.search_drift_max_hz_s)
        )

    margins = []
    coarse_indices = []
    clean_bounds = []
    search_geometry_recorded = bool(
        observation.search_fine_channels_per_coarse > 0
        and observation.search_rolloff_fraction >= 0
    )
    if search_geometry_recorded:
        for channel in channels:
            margin, coarse_index, clean_low, clean_high = _coarse_clean_margin(
                float(channel), observation
            )
            margins.append(margin)
            coarse_indices.append(coarse_index)
            clean_bounds.append((clean_low, clean_high))
        minimum_clean_margin = float(min(margins))
        clean_band_covered = bool(
            full_data_covered
            and drift_in_search
            and minimum_clean_margin >= half_guard_channels
        )
    else:
        minimum_clean_margin = math.nan
        clean_band_covered = False

    if not full_data_covered:
        reason = "outside_filterbank_coverage"
    elif not search_geometry_recorded:
        reason = "search_geometry_not_recorded"
    elif not drift_in_search:
        reason = "drift_outside_search_range"
    elif minimum_clean_margin < half_guard_channels:
        reason = "track_intersects_coarse_channel_rolloff"
    else:
        reason = "searched_clean_band"

    return {
        "full_data_covered": full_data_covered,
        "search_geometry_recorded": search_geometry_recorded,
        "drift_in_search_range": drift_in_search,
        "searched_clean_band_covered": clean_band_covered,
        "reason": reason,
        "track_first_channel": float(channels[0]),
        "track_last_channel": float(channels[-1]),
        "track_min_channel": float(np.min(channels)),
        "track_max_channel": float(np.max(channels)),
        "full_data_minimum_margin_channels": full_margin,
        "clean_band_minimum_margin_channels": minimum_clean_margin,
        "required_half_guard_channels": float(half_guard_channels),
        "coarse_channels_touched": sorted(set(int(value) for value in coarse_indices)),
        "rolloff_fraction": float(observation.search_rolloff_fraction),
        "fine_channels_per_coarse": int(observation.search_fine_channels_per_coarse),
    }


def stage4_width_from_candidate(
    candidate: CandidateRecord,
    mode: str = "native",
) -> Tuple[float, str]:
    """Return the preprocessing width and an explicit provenance label.

    ``native`` is the primary mode and uses Naoise's selected winning template.
    ``restricted_subbank`` is a preregistered sensitivity analysis that uses
    the best 10--100 Hz per-template response, never injected truth.
    """

    if mode == "native":
        return float(candidate.width_hz), "naoise_native_winning_template"
    if mode != "restricted_subbank":
        raise ValueError("width mode must be native or restricted_subbank")
    value = candidate.extras.get("stage4_restricted_width_hz")
    above = bool(candidate.extras.get("stage4_restricted_above_floor", False))
    if value in (None, "") or not above:
        raise ValueError("candidate has no above-floor 10--100 Hz subbank response")
    return float(value), "naoise_per_template_restricted_argmax"
