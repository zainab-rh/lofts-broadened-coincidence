"""Helpers for standards-compliant JSON output."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def json_safe(value):
    """Recursively convert NumPy objects and non-finite floats for JSON.

    Missing or undefined numeric results are written as ``null`` rather than
    the non-standard JavaScript tokens ``NaN`` and ``Infinity``.
    """

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
