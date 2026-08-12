"""Lightweight tests for Stage 3 numerical utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

STAGE3_DIR = Path(__file__).resolve().parents[1] / "stages" / "stage3"
sys.path.insert(0, str(STAGE3_DIR))

import threshold_utils as tu  # noqa: E402
from json_utils import json_safe  # noqa: E402


class ThresholdTests(unittest.TestCase):
    def test_best_f1_threshold_separates_simple_populations(self):
        distances = {
            "broadened_match": np.array([0.05, 0.10, 0.15]),
            "rfi_only_mismatch": np.array([0.70, 0.80, 0.90]),
        }

        result = tu.best_f1_threshold(
            distances,
            candidates=[0.1, 0.2, 0.8],
        )

        self.assertEqual(result["threshold"], 0.2)
        self.assertEqual(result["f1"], 1.0)

    def test_json_safe_replaces_nonfinite_values(self):
        converted = json_safe(
            {
                "nan": np.nan,
                "inf": np.inf,
                "x": np.int64(3),
            }
        )

        self.assertEqual(
            converted,
            {
                "nan": None,
                "inf": None,
                "x": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()