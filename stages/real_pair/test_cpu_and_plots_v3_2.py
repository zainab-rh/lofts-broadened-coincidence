#!/usr/bin/env python3
"""Regression tests for explicit CPU execution and roadmap visualisation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lofts_bliss_schema import CandidateRecord, write_jsonl
from make_observation_manifest import normalize_hdf5_attribute
from plot_real_pair_roadmap import main as plot_main
from plot_stage4_frozen_results import (
    configure_style,
    load_paired_filter_comparison,
    plot_paired_filter_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent


class CpuAndPlotV32Tests(unittest.TestCase):
    def test_hdf5_attribute_normalizer_accepts_vector_labels(self):
        labels = np.asarray([b"time", b"feed_id", b"frequency"], dtype=object)
        self.assertEqual(
            normalize_hdf5_attribute(labels),
            ["time", "feed_id", "frequency"],
        )
        self.assertEqual(normalize_hdf5_attribute(np.int64(1)), 1)
        self.assertEqual(normalize_hdf5_attribute(b"LOFTS0050"), "LOFTS0050")

    def test_cpu_launcher_blocks_importable_cupy_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cupy = root / "cupy"
            cupy.mkdir()
            (cupy / "__init__.py").write_text("MARKER = 'imported'\n", encoding="utf-8")
            source = root / "fake_blind_search.py"
            source.write_text(
                """import json, sys
try:
    import cupy
    backend = "cuda"
except ImportError:
    backend = "cpu"
print(json.dumps({"backend": backend, "argv": sys.argv[1:]}))
""",
                encoding="utf-8",
            )
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            provenance = root / "runtime.json"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = (
                str(root) + os.pathsep + environment.get("PYTHONPATH", "")
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "run_pinned_naoise_search.py"),
                    "--script",
                    str(source),
                    "--expected-script-sha256",
                    digest,
                    "--backend",
                    "cpu",
                    "--runtime-provenance",
                    str(provenance),
                    "--",
                    "input.h5",
                    "--floor",
                    "20",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [
                json.loads(line)
                for line in completed.stdout.splitlines()
                if line.startswith("{")
            ]
            self.assertEqual(records[-1]["backend"], "cpu")
            self.assertEqual(records[-1]["argv"], ["input.h5", "--floor", "20"])
            runtime = json.loads(provenance.read_text(encoding="utf-8"))
            self.assertEqual(runtime["requested_backend"], "cpu")
            self.assertEqual(runtime["effective_backend"], "cpu")
            self.assertEqual(runtime["pinned_script_sha256"], digest)
            self.assertEqual(runtime["exit_status"], 0)
            self.assertTrue(runtime["completed"])

    def test_runner_defaults_to_cpu_without_cuda_gate(self):
        text = (SCRIPT_DIR / "run_lofts0050_real_pair.sh").read_text(encoding="utf-8")
        self.assertIn("BLIND_SEARCH_BACKEND=${BLIND_SEARCH_BACKEND:-cpu}", text)
        self.assertIn("STAGE4_DEVICE=${STAGE4_DEVICE:-cpu}", text)
        self.assertNotIn("verify_cuda_for_bliss", text)
        self.assertIn("run_pinned_naoise_search.py", text)
        self.assertIn("plot-pair-example)", text)
        self.assertIn("plot-presentation)", text)

    def test_locked_paired_filter_result_generates_academic_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "paired_filter.json"
            result_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "analysis_role": "locked_posthoc_incremental_value_analysis",
                        "regime": "detected",
                        "width_interval_hz": [10.0, 100.0],
                        "n_pairs": 10800,
                        "stage4_auc": 0.9931742455418381,
                        "matched_filter_auc": 0.9747044581618656,
                        "stage4_minus_matched_filter": {
                            "delta_auc": 0.018469787379972513,
                            "ci_lo": 0.015609117798353958,
                            "ci_hi": 0.02138732596021954,
                        },
                        "bootstrap": {
                            "replicates": 2000,
                            "ci_level": 0.95,
                            "paired": True,
                            "stratified_by": "shape|width|label|negative_case",
                        },
                        "pair_export_validation": {"validated": True},
                        "model_or_threshold_tuning_performed": False,
                    }
                ),
                encoding="utf-8",
            )
            comparison = load_paired_filter_comparison(result_path)
            self.assertAlmostEqual(comparison["delta_auc"], 0.018469787379972513)
            evaluation = {
                "primary_detected_conditioned": {
                    "methods": {
                        "matched_filter": {
                            "auc": 0.9747044581618656,
                            "ci_lo": 0.971,
                            "ci_hi": 0.978,
                        },
                        "stage4": {
                            "auc": 0.9931742455418381,
                            "ci_lo": 0.991865,
                            "ci_hi": 0.994309,
                        },
                    }
                }
            }
            configure_style()
            outputs = plot_paired_filter_summary(
                comparison, evaluation, root, ("png",), 72
            )
            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].is_file())
            self.assertGreater(outputs[0].stat().st_size, 0)

    def test_representative_pair_plot_uses_existing_archive_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage4 = root / "stage4"
            stage4.mkdir()
            (stage4 / "candidate_preprocessing.py").write_text(
                """import numpy as np
class CandidateParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
def make_candidate_view(raw, params, integration, remove_static_bandpass):
    value = np.asarray(raw, dtype=np.float32)
    scale = float(np.std(value)) or 1.0
    return (value - float(np.mean(value))) / scale
""",
                encoding="utf-8",
            )
            raw_a = np.arange(128, dtype=np.float32).reshape(8, 16)
            raw_b = np.flip(raw_a, axis=1).copy()
            archive = root / "pair.npz"
            np.savez(
                archive,
                raw_a=raw_a,
                raw_b=raw_b,
                union_id=np.asarray("U1"),
            )
            archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            station = {
                "candidate_center_channel": 7.5,
                "reported_drift_hz_s": 0.01,
                "reported_width_fwhm_hz": 24.0,
                "signed_foff_hz": 3.0,
                "tsamp_s": 18.0,
                "candidate_reference_row": 3.5,
            }
            manifest = root / "pairs.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "pair_id": "U1",
                        "union_id": "U1",
                        "detection_state": "SWE_only",
                        "array_file": str(archive),
                        "array_sha256": archive_digest,
                        "station_order": ["IRL", "SWE"],
                        "stations": {"IRL": dict(station), "SWE": dict(station)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = root / "summary.json"
            summary.write_text(json.dumps({"integration": "boxcar"}), encoding="utf-8")
            out_dir = root / "figures"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "plot_representative_pair.py"),
                    "--pair-manifest",
                    str(manifest),
                    "--stage4-code-dir",
                    str(stage4),
                    "--inference-summary",
                    str(summary),
                    "--out-dir",
                    str(out_dir),
                    "--formats",
                    "png",
                    "--dpi",
                    "72",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            figure = out_dir / "14_representative_pair_preprocessing.png"
            metadata = out_dir / "14_representative_pair_preprocessing.metadata.json"
            self.assertTrue(figure.is_file())
            audit = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(audit["selection_rule"], "first_manifest_record")
            self.assertFalse(audit["inference_was_rerun"])

    def test_complete_synthetic_roadmap_plot_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations = root / "observations.summary.json"
            bank = root / "bank.json"
            union_path = root / "union.jsonl"
            primary_path = root / "primary.jsonl"
            controls_path = root / "controls.jsonl"
            catalog_paths = []

            observations.write_text(
                json.dumps(
                    {
                        "observations": [
                            {
                                "station_id": "IRL",
                                "fch1_hz": 100_000_000.0,
                                "signed_foff_hz": 3.0,
                                "n_channels": 1000,
                                "start_mjd": 60_000.0,
                                "n_time": 26,
                                "tsamp_s": 18.0,
                            },
                            {
                                "station_id": "SWE",
                                "fch1_hz": 100_001_000.0,
                                "signed_foff_hz": 2.9,
                                "n_channels": 1000,
                                "start_mjd": 60_000.00001,
                                "n_time": 26,
                                "tsamp_s": 18.0,
                            },
                        ],
                        "group_geometry": {
                            "G": {
                                "common_frequency_low_hz": 100_001_000.0,
                                "common_frequency_high_hz": 100_002_897.1,
                                "time_overlap_start_mjd": 60_000.00001,
                                "integration_overlap_end_mjd_exclusive": 60_000.0054,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            station_audit = {}
            for station, channel in (("IRL", 3.0), ("SWE", 2.9)):
                station_audit[station] = {
                    "native_bank_hz": [
                        value * channel for value in (1, 3, 8, 20, 50, 120)
                    ],
                    "targets": [
                        {"multiplicative_mismatch": 1.0 + index / 20.0}
                        for index in range(6)
                    ],
                }
            bank.write_text(
                json.dumps(
                    {
                        "stations": station_audit,
                        "target_widths_hz": [10, 20, 30, 50, 75, 100],
                    }
                ),
                encoding="utf-8",
            )

            for station_index, station in enumerate(("IRL", "SWE")):
                path = root / (station.lower() + ".jsonl")
                records = []
                for index in range(4):
                    item = CandidateRecord(
                        candidate_id="%s_%d" % (station, index),
                        observation_id="OBS_" + station,
                        simultaneous_group_id="G",
                        station_id=station,
                        frequency_hz=100_001_200.0 + 100 * index + station_index,
                        drift_hz_s=-0.02 + 0.01 * index,
                        width_hz=20.0 + 10 * index,
                        width_definition="selected nominal Lorentzian FWHM",
                        snr=30.0 + index,
                        snr_definition="unit-L2 bank response / row sigma",
                        frequency_ref_mjd=60_000.0,
                        frequency_ref_offset_s=None,
                        extras={
                            "native_width_channels": (1, 3, 8, 20)[index],
                            "bank_snr": 30.0 + index,
                            "standard_snr": 10.0 + index,
                            "bank_standard_ratio": (30.0 + index) / (10.0 + index),
                            "source_flag": "zero_drift" if index == 2 else "",
                            "broadband_rfi_like": index == 3,
                        },
                        truth={},
                    )
                    item.validate()
                    records.append(item.to_dict())
                write_jsonl(str(path), records)
                catalog_paths.append(str(path))

            union = []
            for index in range(3):
                union.append(
                    {
                        "union_id": "U%d" % index,
                        "detection_state": (
                            "two_station" if index < 2 else "one_station_irl"
                        ),
                        "route": "high_resolution_stage4",
                        "operational_eligibility": "eligible",
                        "association": (
                            {
                                "frequency_delta_hz": 1.0 + index,
                                "drift_delta_hz_s": 0.001 + index * 0.0001,
                                "log_width_delta": 0.05 + index * 0.01,
                            }
                            if index < 2
                            else None
                        ),
                    }
                )
            write_jsonl(str(union_path), union)

            primary = []
            controls = []
            for index in range(3):
                row = {
                    "pair_id": "P%d" % index,
                    "source_pair_id": "P%d" % index,
                    "detection_state": (
                        "two_station" if index < 2 else "one_station_irl"
                    ),
                    "operational_eligibility": "eligible",
                    "route": "high_resolution_stage4",
                    "resampling_block_id": "B%d" % (index % 2),
                    "anchor_width_hz": 20.0 + 20 * index,
                    "anchor_snr": 25.0 + 5 * index,
                    "stage3_raw_score": -1.0 + index * 0.1,
                    "stage3_corrected_score": -0.7 + index * 0.1,
                    "matched_filter_score": 0.2 + index * 0.1,
                    "stage4_score": 0.70 + index * 0.08,
                }
                primary.append(row)
                for shift in (-100000, 100000):
                    control = dict(row)
                    control["pair_id"] = "P%d_C%d" % (index, shift)
                    control["source_pair_id"] = row["pair_id"]
                    control["control_shift_hz"] = shift
                    for key in (
                        "stage3_raw_score",
                        "stage3_corrected_score",
                        "matched_filter_score",
                        "stage4_score",
                    ):
                        control[key] = float(row[key]) - 0.05
                    controls.append(control)
            write_jsonl(str(primary_path), primary)
            write_jsonl(str(controls_path), controls)

            out_dir = root / "plots"
            args = SimpleNamespace(
                observations_summary=str(observations),
                bank_audit=str(bank),
                candidate_files=catalog_paths,
                union=str(union_path),
                primary_predictions=str(primary_path),
                control_predictions=str(controls_path),
                out_dir=str(out_dir),
                formats="png",
                dpi=55,
                max_scatter=100,
                top_n=3,
                n_boot=20,
                seed=7,
            )
            plot_main(args)
            inventory = json.loads(
                (out_dir / "roadmap_plot_inventory.json").read_text(encoding="utf-8")
            )
            self.assertEqual(inventory["n_pending"], 0)
            self.assertGreaterEqual(inventory["n_generated"], 14)
            self.assertTrue((out_dir / "00_roadmap_contact_sheet.png").is_file())
            self.assertTrue((out_dir / "09_paired_score_deltas.png").is_file())
            processing = next(
                item
                for item in inventory["plots"]
                if item["plot_id"] == "13_processing_counts"
            )
            self.assertIn("at least two valid controls", processing["interpretation"])
            self.assertIn("unlabeled", inventory["scientific_boundary"].lower())


if __name__ == "__main__":
    unittest.main()
