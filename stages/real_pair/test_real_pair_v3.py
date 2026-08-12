#!/usr/bin/env python3
"""Regression tests for the real Ireland--Sweden Stage-4 pilot extension."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from adapt_naoise_blind_catalog import adapt
from analyze_real_pair import (
    bootstrap_mean,
    clustered_bootstrap_mean,
)
from analyze_real_pair import main as analyze_main
from analyze_real_pair import (
    paired_control_rows,
)
from extract_shifted_controls import CandidateFrequencyIndex, configure_control_policy
from lofts_bliss_schema import (
    CandidateRecord,
    ObservationRecord,
    group_reference_time,
    load_candidates,
    validate_group_observations,
    write_jsonl,
)
from make_observation_manifest import build_record
from real_candidate_union import build_group, component_assignment, sparse_edges
from real_pair_geometry import candidate_track_channels, search_coverage_audit

BANK = (1, 3, 8, 20, 50, 120)


def observation(
    station: str,
    fch1_hz: float = 100_000_000.0,
    signed_foff_hz: float = 1.0,
    n_channels: int = 4000,
) -> ObservationRecord:
    item = ObservationRecord(
        observation_id="OBS_" + station,
        simultaneous_group_id="LOFTS0050_part000",
        station_id=station,
        filterbank_path="/not/required/%s.h5" % station,
        barycentric_status="barycentric",
        time_alignment="absolute_mjd",
        start_mjd=60_592.4,
        n_time=26,
        n_channels=n_channels,
        n_ifs=1,
        fch1_hz=fch1_hz,
        signed_foff_hz=signed_foff_hz,
        tsamp_s=18.11939328,
        nbits=32,
        search_fine_channels_per_coarse=1000,
        search_rolloff_fraction=0.2,
        search_drift_min_hz_s=-0.2,
        search_drift_max_hz_s=0.2,
        search_floor=20.0,
        search_bank_width_channels=BANK,
        search_git_commit="dee329949384f0a0ddb6306d8bbbc2b0db74011a",
    )
    item.validate(require_file=False)
    return item


def candidate(
    candidate_id: str,
    station: str,
    frequency_hz: float,
    drift_hz_s: float = 0.0,
    width_hz: float = 20.0,
    snr: float = 30.0,
) -> CandidateRecord:
    item = CandidateRecord(
        candidate_id=candidate_id,
        observation_id="OBS_" + station,
        simultaneous_group_id="LOFTS0050_part000",
        station_id=station,
        frequency_hz=frequency_hz,
        drift_hz_s=drift_hz_s,
        width_hz=width_hz,
        width_definition="selected nominal Lorentzian FWHM",
        snr=snr,
        snr_definition="unit-L2 bank response / row sigma",
        frequency_ref_mjd=60_592.4,
        frequency_ref_offset_s=None,
        extras={
            "native_width_channels": 20,
            "stage4_restricted_width_hz": width_hz,
            "stage4_restricted_above_floor": True,
            "broadband_rfi_like": False,
        },
        truth={},
    )
    item.validate()
    return item


def policy(width_mode: str = "native"):
    return {
        "status": "frozen_real_pair_pilot",
        "policy_id": "test_policy",
        "association": {
            "frequency_base_hz": 2.0,
            "frequency_width_sum_fraction": 1.0,
            "drift_hz_s": 0.01,
            "log_width": math.log(3.0),
            "component_max_cost": 3.0,
        },
        "deduplication": {"mode": "disabled_input_is_naoise_frequency_primary_nms"},
        "computation": {"maximum_component_nodes": 512},
        "coverage": {"guard_fwhm_fraction": 0.5},
        "resampling": {
            "mode": "common_physical_frequency_blocks",
            "block_width_hz": 1000.0,
        },
        "routing": {
            "width_mode": width_mode,
            "stage4_width_min_hz": 10.0,
            "stage4_width_max_hz": 100.0,
        },
    }


class RealPairV3Tests(unittest.TestCase):
    def test_manifest_requires_header_barycentric_flag_for_real_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "product.h5"
            path.touch()
            row = {
                "observation_id": "OBS_A",
                "simultaneous_group_id": "GROUP",
                "station_id": "A",
                "filterbank_path": str(path),
                "barycentric_status": "barycentric",
                "time_alignment": "absolute_mjd",
                "barycentric_tool": "test_tool",
                "barycentric_version": "test_version",
            }
            args = SimpleNamespace(
                foff_tolerance_hz=1e-6,
                tsamp_tolerance_s=1e-9,
                require_barycentric_provenance=False,
            )
            audit = {
                "path": str(path),
                "fch1_mhz": 100.0,
                "foff_mhz": 1e-6,
                "tsamp_s": 1.0,
                "tstart_mjd": 60000.0,
                "n_time": 2,
                "n_ifs": 1,
                "n_channels": 16,
                "nbits": 32,
                "header_barycentric": 0,
                "source_name": "TEST",
                "source_ra_sigproc": None,
                "source_dec_sigproc": None,
                "telescope_id": None,
                "machine_id": None,
                "data_type": None,
                "header_fingerprint": "test",
            }
            with mock.patch(
                "make_observation_manifest.inspect_filterbank", return_value=audit
            ):
                with self.assertRaisesRegex(
                    ValueError, "does not contain barycentric=1"
                ):
                    build_record(row, args)
            audit["header_barycentric"] = 1
            with mock.patch(
                "make_observation_manifest.inspect_filterbank", return_value=audit
            ):
                record = build_record(row, args)
            self.assertEqual(record.header_barycentric, 1)

    def test_shifted_control_plan_is_frozen_before_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy_path = Path(temporary) / "policy.json"
            frozen = policy()
            frozen["controls"] = {
                "kind": "distant_frequency_shift",
                "shifts_hz": [-300000.0, -100000.0, 100000.0, 300000.0],
                "minimum_controls_per_pair": 2,
                "candidate_exclusion_width_sum_fraction": 4.0,
                "candidate_exclusion_base_hz": 6.0,
                "edge_guard_widths": 4.0,
                "station_rule": (
                    "shift_non_reporting_counterpart_for_one_station_entries_and_"
                    "each_station_separately_for_two_station_entries"
                ),
            }
            policy_path.write_text(json.dumps(frozen), encoding="utf-8")
            args = SimpleNamespace(
                policy=str(policy_path),
                shifts_hz=None,
                minimum_controls_per_pair=None,
                candidate_exclusion_widths=None,
                candidate_exclusion_base_hz=None,
                edge_guard_widths=None,
                shift_station=None,
            )
            audit = configure_control_policy(args)
            self.assertEqual(args.minimum_controls_per_pair, 2)
            self.assertEqual(args.shifts_hz, "-300000,-100000,100000,300000")
            self.assertEqual(audit["policy_id"], "test_policy")
            conflicting = SimpleNamespace(
                policy=str(policy_path),
                shifts_hz="-1,1",
                minimum_controls_per_pair=None,
                candidate_exclusion_widths=None,
                candidate_exclusion_base_hz=None,
                edge_guard_widths=None,
                shift_station=None,
            )
            with self.assertRaisesRegex(ValueError, "conflicts with the frozen"):
                configure_control_policy(conflicting)

    def test_shifted_control_candidate_exclusion_uses_physical_frequency(self):
        known = [
            candidate("A:indexed", "A", 100_001_000.0, width_hz=20.0),
            candidate("A:distant", "A", 100_100_000.0, width_hz=20.0),
        ]
        index = CandidateFrequencyIndex(known)
        conflict = index.first_conflict(
            "LOFTS0050_part000",
            "A",
            "mjd",
            60_592.4,
            100_001_005.0,
            20.0,
            6.0,
            1.0,
        )
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict[1], "A:indexed")
        self.assertIsNone(
            index.first_conflict(
                "LOFTS0050_part000",
                "A",
                "mjd",
                60_592.4,
                100_050_000.0,
                20.0,
                6.0,
                1.0,
            )
        )

    def test_signed_frequency_axis_reverses_channel_track_direction(self):
        positive = observation("A", fch1_hz=100_000_000.0, signed_foff_hz=1.0)
        negative = observation("B", fch1_hz=100_004_000.0, signed_foff_hz=-1.0)
        left = candidate("A:1", "A", 100_000_500.0, drift_hz_s=0.1)
        right = candidate("B:1", "B", 100_003_500.0, drift_hz_s=0.1)
        left_track = candidate_track_channels(left, positive, rows=(0, 1, 2))
        right_track = candidate_track_channels(right, negative, rows=(0, 1, 2))
        self.assertGreater(left_track[-1], left_track[0])
        self.assertLess(right_track[-1], right_track[0])

    def test_rolloff_coverage_is_not_called_a_non_detection(self):
        station_a = observation("A", fch1_hz=100_000_000.0)
        # The same physical frequency maps to channel 100 at B, inside its 20% rolloff.
        station_b = observation("B", fch1_hz=100_000_400.0)
        hit_a = candidate("A:edge_for_B", "A", 100_000_500.0)
        own = search_coverage_audit(hit_a, station_a, guard_fwhm_fraction=0.5)
        other = search_coverage_audit(hit_a, station_b, guard_fwhm_fraction=0.5)
        self.assertTrue(own["searched_clean_band_covered"])
        self.assertFalse(other["searched_clean_band_covered"])
        self.assertEqual(other["reason"], "track_intersects_coarse_channel_rolloff")
        entries, _ = build_group(
            "LOFTS0050_part000", [station_a, station_b], [hit_a], policy()
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(
            entries[0]["operational_eligibility"], "counterpart_not_searched"
        )
        self.assertEqual(
            entries[0]["route"], "high_resolution_stage4_counterpart_not_searched"
        )

    def test_rolloff_clean_high_is_exclusive(self):
        station = observation("A", fch1_hz=100_000_000.0)
        # With 1,000 channels/coarse and 20% rolloff, clean centres are
        # [200, 799]. Channel 800 is the first excluded upper-edge centre.
        last_clean = candidate("A:last_clean", "A", 100_000_799.0, width_hz=1.0)
        first_excluded = candidate("A:first_excluded", "A", 100_000_800.0, width_hz=1.0)
        self.assertTrue(
            search_coverage_audit(last_clean, station, guard_fwhm_fraction=0.0)[
                "searched_clean_band_covered"
            ]
        )
        excluded_audit = search_coverage_audit(
            first_excluded, station, guard_fwhm_fraction=0.0
        )
        self.assertFalse(excluded_audit["searched_clean_band_covered"])
        self.assertEqual(
            excluded_audit["reason"], "track_intersects_coarse_channel_rolloff"
        )

    def test_real_reference_is_midpoint_of_sampled_rows(self):
        left = observation("A")
        right = observation("B")
        kind, value = group_reference_time([left, right])
        expected = left.start_mjd + 12.5 * left.tsamp_s / 86400.0
        self.assertEqual(kind, "mjd")
        self.assertAlmostEqual(value, expected, places=12)

    def test_real_observations_reject_inconsistent_source_metadata(self):
        left = replace(
            observation("A"),
            source_name="TARGET_A",
            source_ra_sigproc=123456.0,
            source_dec_sigproc=12345.0,
        )
        right_name = replace(
            observation("B"),
            source_name="TARGET_B",
            source_ra_sigproc=123456.0,
            source_dec_sigproc=12345.0,
        )
        with self.assertRaisesRegex(ValueError, "different source names"):
            validate_group_observations([left, right_name], False)
        right_position = replace(
            observation("B"),
            source_name="TARGET_A",
            source_ra_sigproc=123457.0,
            source_dec_sigproc=12345.0,
        )
        with self.assertRaisesRegex(ValueError, "different source_ra_sigproc"):
            validate_group_observations([left, right_position], False)

    def test_sparse_union_avoids_quadratic_comparisons_and_preserves_matches(self):
        left = [
            candidate("A:%d" % index, "A", 100_000_000.0 + 10_000.0 * index)
            for index in range(1000)
        ]
        right = [
            candidate("B:%d" % index, "B", 100_000_001.0 + 10_000.0 * index)
            for index in range(1000)
        ]
        edges, comparisons = sparse_edges(left, right, "mjd", 60_592.4, policy())
        self.assertEqual(len(edges), 1000)
        self.assertLess(comparisons, 5000)
        assignments = component_assignment(1000, 1000, edges)
        self.assertEqual(len(assignments), 1000)
        self.assertTrue(
            all(left_index == right_index for left_index, right_index, _ in assignments)
        )

    def test_dense_component_fails_closed(self):
        edges = {
            (0, 0): {"cost": 0.1},
            (0, 1): {"cost": 0.2},
            (1, 0): {"cost": 0.2},
            (1, 1): {"cost": 0.1},
        }
        with self.assertRaisesRegex(RuntimeError, "safety limit"):
            component_assignment(2, 2, edges, maximum_component_nodes=3)

    def test_naoise_adapter_validates_all_per_template_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # The refined candidate is deliberately in the coarse-channel
            # roll-off. It was nevertheless emitted by BLISS and therefore
            # must be retained for the union with an explicit audit flag.
            obs = observation("IRL", fch1_hz=150_000_400.0, signed_foff_hz=1.0)
            observations = root / "observations.jsonl"
            write_jsonl(str(observations), [obs.to_dict()])
            raw = root / "raw.csv"
            with raw.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "OBS",
                        "CANDIDATE_ID",
                        "STATION",
                        "FREQ_MHZ",
                        "DR_HZ_S",
                        "WIDTH",
                        "WIDTH_HZ",
                        "CHAN_BW_HZ",
                        "BANK_SNR",
                        "STANDARD_SNR",
                        "DETECTED",
                        "FLAG",
                        "TEMPLATES_SKIPPED",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "OBS": "LOFTS0050_part000",
                        "CANDIDATE_ID": "0",
                        "STATION": "IRL",
                        "FREQ_MHZ": "150.000500",
                        "DR_HZ_S": "0.0",
                        "WIDTH": "8",
                        "WIDTH_HZ": "8.0",
                        "CHAN_BW_HZ": "1.0",
                        "BANK_SNR": "30.0",
                        "STANDARD_SNR": "10.0",
                        "DETECTED": "1",
                        "FLAG": "",
                        "TEMPLATES_SKIPPED": "",
                    }
                )
            per = root / "per.csv"
            responses = {1: 25.0, 3: 26.0, 8: 30.0, 20: 29.0, 50: 27.0, 120: 23.0}
            with per.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "OBS",
                        "CANDIDATE_ID",
                        "STATION",
                        "FREQ_MHZ",
                        "DR_HZ_S",
                        "WINNING_WIDTH",
                        "TEMPLATE_WIDTH",
                        "TEMPLATE_WIDTH_HZ",
                        "TEMPLATE_BANK_SNR",
                        "STANDARD_SNR",
                        "FLAG",
                    ],
                )
                writer.writeheader()
                for width in BANK:
                    writer.writerow(
                        {
                            "OBS": "LOFTS0050_part000",
                            "CANDIDATE_ID": "0",
                            "STATION": "IRL",
                            "FREQ_MHZ": "150.000500",
                            "DR_HZ_S": "0.0",
                            "WINNING_WIDTH": "8",
                            "TEMPLATE_WIDTH": str(width),
                            "TEMPLATE_WIDTH_HZ": str(float(width)),
                            "TEMPLATE_BANK_SNR": str(responses[width]),
                            "STANDARD_SNR": "10.0",
                            "FLAG": "",
                        }
                    )
            args = SimpleNamespace(
                observations=str(observations),
                raw_csv=str(raw),
                per_template_csv=str(per),
                simultaneous_group_id="LOFTS0050_part000",
                station_id="IRL",
                observation_id="OBS_IRL",
                expected_obs_label="LOFTS0050_part000",
                bank_widths="1,3,8,20,50,120",
                channel_bw_tolerance_hz=1e-6,
                width_hz_tolerance=1e-3,
                snr_rounding_tolerance=1e-3,
                frequency_rounding_tolerance_hz=1.0,
                drift_rounding_tolerance_hz_s=1e-5,
                floor=20.0,
                width_tolerance=0.10,
                stage4_width_min_hz=10.0,
                stage4_width_max_hz=100.0,
                wide_threshold_channels=120,
                rfi_ratio=8.0,
                search_git_commit="dee329949384f0a0ddb6306d8bbbc2b0db74011a",
            )
            records, audit, metrics = adapt(args)
            self.assertEqual(len(records), 1)
            self.assertEqual(audit["per_template_rows_per_candidate"], 6)
            self.assertEqual(records[0].extras["stage4_restricted_width_channels"], 20)
            self.assertTrue(records[0].extras["stage4_restricted_above_floor"])
            self.assertEqual(len(records[0].extras["per_template_snr"]), 6)
            self.assertFalse(
                records[0].extras["own_search_coverage"]["searched_clean_band_covered"]
            )
            self.assertEqual(audit["n_own_tracks_not_fully_in_clean_search_band"], 1)
            self.assertEqual(len(metrics), 1)
            # The adapter output is immediately consumed as canonical JSONL by
            # the union. Exercise that exact serialization boundary so a newly
            # added operational audit field cannot be rejected downstream.
            canonical = root / "canonical.jsonl"
            write_jsonl(
                str(canonical),
                [records[0].to_dict(include_truth=False)],
            )
            reloaded = load_candidates(str(canonical))
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(
                reloaded[0].extras["own_search_coverage"]["reason"],
                "track_intersects_coarse_channel_rolloff",
            )

    def test_real_control_analysis_is_paired_and_label_free(self):
        primary = [
            {
                "pair_id": "P1",
                "detection_state": "two_station",
                "operational_eligibility": "two_station_detected",
                "route": "high_resolution_stage4",
                "stage3_raw_score": 0.3,
                "stage3_corrected_score": 0.4,
                "matched_filter_score": 0.7,
                "stage4_score": 0.9,
                "resampling_block_id": "block-1",
            }
        ]
        controls = [
            {
                "source_pair_id": "P1",
                "stage3_raw_score": 0.2,
                "stage3_corrected_score": 0.3,
                "matched_filter_score": 0.5,
                "stage4_score": 0.6,
            },
            {
                "source_pair_id": "P1",
                "stage3_raw_score": 0.1,
                "stage3_corrected_score": 0.2,
                "matched_filter_score": 0.4,
                "stage4_score": 0.5,
            },
        ]
        rows, audit = paired_control_rows(primary, controls)
        self.assertEqual(audit["n_primary_with_at_least_one_control"], 1)
        self.assertAlmostEqual(rows[0]["stage4_score_delta"], 0.35)
        interval = bootstrap_mean([rows[0]["stage4_score_delta"]], 100, 7)
        self.assertEqual(interval["ci95"], [0.35, 0.35])
        clustered = clustered_bootstrap_mean(
            [0.2, 0.4, -0.1, 0.1],
            ["block-1", "block-1", "block-2", "block-2"],
            200,
            7,
        )
        self.assertEqual(clustered["n_blocks"], 2)
        self.assertIsNotNone(clustered["ci95"][0])
        self.assertNotIn("label", rows[0])

        # Exercise the complete report/plot writer, including the physical-
        # frequency block bootstrap that is used for the real unlabeled pilot.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            write_jsonl(
                str(catalog),
                [
                    candidate("A:report", "A", 100_000_500.0).to_dict(
                        include_truth=False
                    )
                ],
            )
            union = root / "union.jsonl"
            write_jsonl(
                str(union),
                [
                    {
                        "union_id": "P1",
                        "detection_state": "two_station",
                        "operational_eligibility": "two_station_detected",
                        "route": "high_resolution_stage4",
                        "association": None,
                    },
                    {
                        "union_id": "P2",
                        "detection_state": "A_only",
                        "operational_eligibility": "eligible_one_station_non_detection",
                        "route": "high_resolution_stage4",
                        "association": None,
                    },
                ],
            )
            primary_path = root / "primary.jsonl"
            primary_complete = primary + [
                {
                    **primary[0],
                    "pair_id": "P2",
                    "stage4_score": 0.7,
                    "resampling_block_id": "block-2",
                }
            ]
            write_jsonl(str(primary_path), primary_complete)
            controls_path = root / "controls.jsonl"
            controls_complete = controls + [
                {
                    **controls[0],
                    "source_pair_id": "P2",
                    "stage4_score": 0.4,
                },
                {
                    **controls[1],
                    "source_pair_id": "P2",
                    "stage4_score": 0.3,
                },
            ]
            write_jsonl(str(controls_path), controls_complete)
            out = root / "analysis"
            analyze_main(
                SimpleNamespace(
                    candidate_files=[str(catalog)],
                    union=str(union),
                    primary_predictions=str(primary_path),
                    control_predictions=str(controls_path),
                    out_dir=str(out),
                    n_boot=100,
                    seed=7,
                    top_n=10,
                )
            )
            analysis = json.loads((out / "real_pair_analysis.json").read_text())
            self.assertFalse(analysis["labels_used"])
            self.assertEqual(
                analysis["paired_frequency_shift_control"]["method_deltas"][
                    "stage4_score"
                ]["n_blocks"],
                2,
            )
            for name in (
                "real_pair_report.md",
                "real_pair_top_candidates.csv",
                "paired_frequency_shift_deltas.csv",
                "01_catalog_template_occupancy.png",
                "05_paired_control_deltas.pdf",
            ):
                self.assertGreater((out / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
