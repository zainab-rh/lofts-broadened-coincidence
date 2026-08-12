#!/usr/bin/env python3
"""Bounded, index-audited filterbank reads for candidate extraction.

Only the requested time/frequency rectangle is loaded through blimpy.  The
reader converts requested channel indices to physical frequency bounds using
the *signed* SIGPROC ``foff`` and verifies blimpy's selected channel indices
and returned shape.  No wraparound or edge padding is permitted.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
from lofts_bliss_schema import ObservationRecord

_HDF5_HANDLES: Dict[str, Any] = {}


def close_cached_filterbanks() -> None:
    """Close process-local read-only HDF5 handles."""

    for handle in list(_HDF5_HANDLES.values()):
        try:
            handle.close()
        except Exception:
            pass
    _HDF5_HANDLES.clear()


atexit.register(close_cached_filterbanks)


def _cached_hdf5_file(path: Path):
    """Open each large read-only HDF5 observation once per process."""

    try:
        # Importing hdf5plugin registers the bitshuffle/LZ4 filters used by
        # BLISS-compatible HDF5 products. Some cluster installations expose
        # these through HDF5_PLUGIN_PATH, so absence is not itself fatal.
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for exact HDF5 extraction") from exc
    key = str(path.resolve())
    handle = _HDF5_HANDLES.get(key)
    if handle is None or not bool(handle.id.valid):
        handle = h5py.File(key, "r")
        _HDF5_HANDLES[key] = handle
    return handle


def validate_window_bounds(
    observation: ObservationRecord,
    t_start: int,
    f_start: int,
    n_rows: int,
    n_cols: int,
) -> None:
    values = (int(t_start), int(f_start), int(n_rows), int(n_cols))
    t_start, f_start, n_rows, n_cols = values
    if t_start < 0 or f_start < 0 or n_rows <= 0 or n_cols <= 0:
        raise ValueError("window indices must be non-negative and dimensions positive")
    if t_start + n_rows > observation.n_time:
        raise ValueError(
            "time window [%d,%d) exceeds observation with %d rows"
            % (t_start, t_start + n_rows, observation.n_time)
        )
    if f_start + n_cols > observation.n_channels:
        raise ValueError(
            "frequency window [%d,%d) exceeds observation with %d channels"
            % (f_start, f_start + n_cols, observation.n_channels)
        )


def frequency_bounds_mhz(
    observation: ObservationRecord, f_start: int, n_cols: int
) -> Tuple[float, float]:
    """Return ascending physical bounds selecting exactly ``n_cols`` channels."""

    first = observation.frequency_hz_for_channel(float(f_start)) / 1e6
    exclusive = observation.frequency_hz_for_channel(float(f_start + n_cols)) / 1e6
    return min(first, exclusive), max(first, exclusive)


def read_filterbank_window(
    observation: ObservationRecord,
    t_start: int,
    f_start: int,
    n_rows: int,
    n_cols: int,
    if_index: int = 0,
    max_load_gb: float = 0.1,
    fail_on_masked: bool = True,
    return_audit: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Load one exact ``time x frequency`` candidate cutout.

    The returned column ordering is the native file ordering.  Consequently
    ``observation.signed_foff_hz`` remains the correct de-chirp denominator.
    """

    validate_window_bounds(observation, t_start, f_start, n_rows, n_cols)
    if not (0 <= int(if_index) < int(observation.n_ifs)):
        raise ValueError("if_index is outside the observation IF axis")
    target = Path(observation.filterbank_path)
    if target.suffix.lower() in {".h5", ".hdf5"}:
        handle = _cached_hdf5_file(target)
        if "data" not in handle:
            raise ValueError("HDF5 filterbank has no /data dataset")
        dataset = handle["data"]
        try:
            if dataset.ndim == 3:
                selected = dataset[
                    int(t_start) : int(t_start + n_rows),
                    int(if_index),
                    int(f_start) : int(f_start + n_cols),
                ]
            elif dataset.ndim == 2 and int(if_index) == 0:
                selected = dataset[
                    int(t_start) : int(t_start + n_rows),
                    int(f_start) : int(f_start + n_cols),
                ]
            else:
                raise ValueError("unsupported HDF5 /data shape %r" % (dataset.shape,))
        except OSError as exc:
            if "filter" in str(exc).lower() or "plugin" in str(exc).lower():
                raise RuntimeError(
                    "HDF5 cutout decompression failed. Install/import "
                    "hdf5plugin in the Stage-4 environment or configure the "
                    "cluster's HDF5_PLUGIN_PATH."
                ) from exc
            raise
        masked_count = 0
        mask_present = "mask" in handle
        if mask_present:
            mask_dataset = handle["mask"]
            if mask_dataset.ndim == 3:
                mask = mask_dataset[
                    int(t_start) : int(t_start + n_rows),
                    int(if_index),
                    int(f_start) : int(f_start + n_cols),
                ]
            elif mask_dataset.ndim == 2 and int(if_index) == 0:
                mask = mask_dataset[
                    int(t_start) : int(t_start + n_rows),
                    int(f_start) : int(f_start + n_cols),
                ]
            else:
                raise ValueError(
                    "unsupported HDF5 /mask shape %r" % (mask_dataset.shape,)
                )
            masked_count = int(np.count_nonzero(mask))
            if masked_count and fail_on_masked:
                raise ValueError(
                    "requested HDF5 cutout contains %d masked samples; "
                    "no imputation policy has been registered" % masked_count
                )
        result = np.asarray(selected, dtype=np.float32).copy()
        expected_shape = (int(n_rows), int(n_cols))
        if tuple(result.shape) != expected_shape:
            raise RuntimeError(
                "HDF5 extraction returned shape %r, expected %r"
                % (tuple(result.shape), expected_shape)
            )
        if not np.isfinite(result).all():
            raise ValueError("filterbank cutout contains NaN or infinite values")
        audit = {
            "reader": "h5py_exact_slice_cached_read_only_handle",
            "mask_present": bool(mask_present),
            "masked_sample_count": masked_count,
            "masked_fraction": masked_count / float(result.size),
            "mask_policy": "fail" if fail_on_masked else "preserve_data_values",
        }
        return (result, audit) if return_audit else result

    try:
        import blimpy as bl
    except ImportError as exc:
        raise ImportError("blimpy is required for filterbank extraction") from exc

    f_low_mhz, f_high_mhz = frequency_bounds_mhz(observation, f_start, n_cols)
    waterfall = bl.Waterfall(
        observation.filterbank_path,
        f_start=f_low_mhz,
        f_stop=f_high_mhz,
        t_start=int(t_start),
        t_stop=int(t_start + n_rows),
        load_data=True,
        max_load=float(max_load_gb),
    )
    data = np.asarray(waterfall.data)
    if data.ndim == 2:
        selected = data
    elif data.ndim == 3:
        if data.shape[1] <= int(if_index):
            raise RuntimeError("blimpy returned fewer IFs than declared")
        selected = data[:, int(if_index), :]
    else:
        raise RuntimeError("blimpy returned unsupported data shape %r" % (data.shape,))

    selected_start = getattr(waterfall.container, "chan_start_idx", None)
    selected_stop = getattr(waterfall.container, "chan_stop_idx", None)
    if selected_start is not None and int(selected_start) != int(f_start):
        raise RuntimeError(
            "blimpy selected channel start %d, requested %d"
            % (int(selected_start), int(f_start))
        )
    if selected_stop is not None and int(selected_stop) != int(f_start + n_cols):
        raise RuntimeError(
            "blimpy selected channel stop %d, requested %d"
            % (int(selected_stop), int(f_start + n_cols))
        )
    expected_shape = (int(n_rows), int(n_cols))
    if tuple(selected.shape) != expected_shape:
        raise RuntimeError(
            "filterbank extraction returned shape %r, expected %r; refusing to pad/crop"
            % (tuple(selected.shape), expected_shape)
        )
    result = np.asarray(selected, dtype=np.float32).copy()
    if not np.isfinite(result).all():
        raise ValueError("filterbank cutout contains NaN or infinite values")
    audit = {
        "reader": "blimpy_bounded_slice",
        "mask_present": False,
        "masked_sample_count": 0,
        "masked_fraction": 0.0,
        "mask_policy": "not_available_through_reader",
    }
    return (result, audit) if return_audit else result
