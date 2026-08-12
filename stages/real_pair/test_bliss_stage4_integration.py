#!/usr/bin/env python3
"""CPU-only regressions for the BLISS -> Stage-4 integration boundary."""

from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from bliss_candidate_adapter import _candidate_record
from build_synthetic_test_b_labels import _event_registry, label_union_entry
from candidate_union import _deduplicate, build_group_union
from estimate_bliss_recovery import _exact_link_matches
from evaluate_bliss_stage4 import (
    _pipeline_denominators,
)
from evaluate_bliss_stage4 import main as evaluate_main
from evaluate_bliss_stage4 import (
    paired_bootstrap,
)
from extract_candidate_pairs import _frequency_window
from link_test_b_recoveries import _validate_exact_links
from lofts_bliss_schema import (
    CandidateRecord,
    InjectionTruthRecord,
    ObservationRecord,
    read_json_records,
    validate_group_observations,
    write_json,
    write_jsonl,
)
from lofts_filterbank import frequency_bounds_mhz


def observation(station: str, signed_foff_hz: float = 3.0) -> ObservationRecord:
    fch1 = 1_000_000_000.0 if signed_foff_hz > 0 else 1_000_030_000.0
    item = ObservationRecord(
        observation_id="obs_" + station,
        simultaneous_group_id="group_1",
        station_id=station,
        filterbank_path="/nonexistent/" + station + ".fil",
        barycentric_status="synthetic_proxy",
        time_alignment="normalized_proxy",
        start_mjd=60000.0,
        n_time=128,
        n_channels=10_000,
        n_ifs=1,
        fch1_hz=fch1,
        signed_foff_hz=signed_foff_hz,
        tsamp_s=1.0,
        nbits=32,
    )
    item.validate()
    return item


def candidate(
    candidate_id: str,
    station: str,
    frequency_hz: float,
    snr: float = 10.0,
    width_hz: float = 30.0,
    extras=None,
) -> CandidateRecord:
    item = CandidateRecord(
        candidate_id=candidate_id,
        observation_id="obs_" + station,
        simultaneous_group_id="group_1",
        station_id=station,
        frequency_hz=frequency_hz,
        drift_hz_s=0.1,
        width_hz=width_hz,
        width_definition="FWHM",
        snr=snr,
        snr_definition="test statistic",
        frequency_ref_mjd=None,
        frequency_ref_offset_s=0.0,
        extras={} if extras is None else extras,
    )
    item.validate()
    return item


def truth(injection_id: str, event_id: str, station: str, label: int = 1):
    item = InjectionTruthRecord(
        injection_id=injection_id,
        event_id=event_id,
        observation_id="obs_" + station,
        simultaneous_group_id="group_1",
        station_id=station,
        frequency_hz=1_000_006_000.0,
        drift_hz_s=0.1,
        width_hz=30.0,
        snr=12.0,
        frequency_ref_mjd=None,
        frequency_ref_offset_s=0.0,
        shape="lorentzian",
        pair_label=label,
        case="genuine_match" if label else "onesided",
        extras={
            "population": "detected_conditioned",
            "evaluation_cell_id": "lorentzian_30_hz",
            "resampling_block_id": "block_01",
        },
    )
    item.validate()
    return item


POLICY = {
    "policy_id": "policy_test",
    "association_tolerances": {
        "frequency_hz": 4.0,
        "drift_hz_s": 0.2,
        "log_width": math.log(1.5),
    },
    "deduplication_tolerances": {
        "frequency_hz": 5.0,
        "drift_hz_s": 0.2,
        "log_width": math.log(1.5),
    },
}


class BlissStage4IntegrationTests(unittest.TestCase):
    def test_proxy_alignment_requires_explicit_opt_in(self):
        observations = [observation("A"), observation("B")]
        with self.assertRaises(ValueError):
            validate_group_observations(observations, allow_normalized_proxy=False)
        validate_group_observations(observations, allow_normalized_proxy=True)

    def test_real_alignment_requires_barycentric_overlapping_products(self):
        a = observation("A")
        b = observation("B")
        real_a = ObservationRecord(
            **{
                **a.to_dict(),
                "barycentric_status": "barycentric",
                "time_alignment": "absolute_mjd",
            }
        )
        real_b = ObservationRecord(
            **{
                **b.to_dict(),
                "barycentric_status": "barycentric",
                "time_alignment": "absolute_mjd",
            }
        )
        validate_group_observations([real_a, real_b], allow_normalized_proxy=False)
        topocentric_b = ObservationRecord(
            **{**real_b.to_dict(), "barycentric_status": "topocentric"}
        )
        with self.assertRaises(ValueError):
            validate_group_observations(
                [real_a, topocentric_b], allow_normalized_proxy=False
            )

    def test_candidate_schema_rejects_truth_adjacent_extras(self):
        with self.assertRaises(ValueError):
            candidate("c", "A", 1_000_006_000.0, extras={"recovery_link_id": "inj"})

    def test_candidate_adapter_allowlists_operational_extras(self):
        obs = observation("A")
        mapping = {
            "columns": {
                "candidate_id": "hit",
                "frequency": "freq",
                "drift": "drift",
                "width": "width",
                "snr": "snr",
            },
            "constants": {
                "observation_id": obs.observation_id,
                "simultaneous_group_id": obs.simultaneous_group_id,
                "station_id": obs.station_id,
                "width_definition": "FWHM",
                "snr_definition": "test",
            },
            "units": {"frequency": "Hz", "drift": "Hz/s", "width": "Hz"},
            "frequency_reference": {"mode": "observation_midpoint"},
            "id_prefix": "A:",
        }
        row = {
            "hit": "h1",
            "freq": "1000006000",
            "drift": "0.1",
            "width": "30",
            "snr": "12",
            "pair_label": "1",
            "injection_id": "secret",
        }
        item = _candidate_record(
            row,
            mapping,
            {(obs.simultaneous_group_id, obs.station_id): obs},
            "table.csv",
            1,
        )
        self.assertEqual(item.candidate_id, "A:h1")
        self.assertEqual(item.extras, {})
        self.assertEqual(item.truth, {})

    def test_signed_frequency_axis_controls_track_direction(self):
        positive = observation("A", +3.0)
        negative = observation("A", -3.0)
        frequency_positive = positive.frequency_hz_for_channel(5000.0)
        frequency_negative = negative.frequency_hz_for_channel(5000.0)
        _, _, track_positive = _frequency_window(
            positive, frequency_positive, +0.3, 12.0, 16, 1024, 7.5, 4.0
        )
        _, _, track_negative = _frequency_window(
            negative, frequency_negative, +0.3, 12.0, 16, 1024, 7.5, 4.0
        )
        self.assertLess(
            track_positive["track_first_channel"], track_positive["track_last_channel"]
        )
        self.assertGreater(
            track_negative["track_first_channel"], track_negative["track_last_channel"]
        )
        # The physical bounds must be ascending for either file orientation.
        for item in (positive, negative):
            low, high = frequency_bounds_mhz(item, 100, 1024)
            self.assertLess(low, high)
            self.assertAlmostEqual((high - low) * 1e6, 1024 * 3.0, places=5)

    def test_deduplication_does_not_chain_incompatible_endpoints(self):
        values = [
            candidate("c1", "A", 1_000_000_000.0, snr=30),
            candidate("c2", "A", 1_000_000_004.0, snr=20),
            candidate("c3", "A", 1_000_000_008.0, snr=10),
        ]
        representatives, members = _deduplicate(
            values, "offset_s", 0.0, POLICY["deduplication_tolerances"]
        )
        self.assertEqual([item.candidate_id for item in representatives], ["c1", "c3"])
        self.assertEqual(members["c1"], ["c1", "c2"])

    def test_union_retains_one_station_detections_and_strips_truth(self):
        observations = [observation("A"), observation("B")]
        values = [
            candidate("a_shared", "A", 1_000_006_000.0, snr=20),
            candidate("b_shared", "B", 1_000_006_002.0, snr=18),
            candidate("a_only", "A", 1_000_009_000.0, snr=15),
        ]
        entries, audit = build_group_union(
            "group_1", observations, values, POLICY, 10.0, 100.0
        )
        self.assertEqual(audit["n_two_station"], 1)
        self.assertEqual(audit["n_one_station"], 1)
        self.assertEqual(len(entries), 2)
        for entry in entries:
            self.assertIsNone(entry["truth"])
            for station in entry["stations"].values():
                if station["candidate"]:
                    self.assertNotIn("truth", station["candidate"])

    def test_post_inference_label_join_handles_one_station_positive(self):
        truths = [
            truth("inj_a", "event_shared", "A"),
            truth("inj_b", "event_shared", "B"),
        ]
        events = _event_registry(truths)
        entry = {
            "union_id": "u1",
            "simultaneous_group_id": "group_1",
            "station_ids": ["A", "B"],
            "detection_state": "one_station",
            "stations": {
                "A": {"detected": True, "candidate": {"candidate_id": "cand_a"}},
                "B": {"detected": False, "candidate": None},
            },
        }
        result = label_union_entry(
            entry,
            {item.injection_id: item for item in truths},
            events,
            {"cand_a": "inj_a"},
        )
        self.assertEqual(result["label"], 1)
        self.assertEqual(result["event_id"], "event_shared")

    def test_post_inference_label_join_rejects_associated_distinct_events(self):
        truth_a = truth("inj_a", "event_a", "A", label=0)
        truth_b = truth("inj_b", "event_b", "B", label=0)
        truths = [truth_a, truth_b]
        entry = {
            "union_id": "u2",
            "simultaneous_group_id": "group_1",
            "station_ids": ["A", "B"],
            "detection_state": "two_station",
            "stations": {
                "A": {"detected": True, "candidate": {"candidate_id": "cand_a"}},
                "B": {"detected": True, "candidate": {"candidate_id": "cand_b"}},
            },
        }
        result = label_union_entry(
            entry,
            {item.injection_id: item for item in truths},
            _event_registry(truths),
            {"cand_a": "inj_a", "cand_b": "inj_b"},
        )
        self.assertEqual(result["label"], 0)
        self.assertEqual(result["case"], "associated_distinct_injections")

    def test_exact_recovery_links_are_segregated_sidecar_input(self):
        recovered = candidate("cand_a", "A", 1_000_006_000.0)
        injected = truth("inj_a", "event_a", "A", label=1)
        matches, false_hits, missed = _exact_link_matches(
            [recovered], [injected], {"cand_a": "inj_a"}
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(false_hits, [])
        self.assertEqual(missed, [])

    def test_post_inference_exact_link_validation_preserves_duplicates(self):
        recovered_a = candidate("cand_a", "A", 1_000_006_000.0)
        recovered_duplicate = candidate("cand_a_dup", "A", 1_000_006_001.0)
        injected = truth("inj_a", "event_a", "A", label=1)
        result = _validate_exact_links(
            [
                {"candidate_id": "cand_a", "recovery_link_id": "inj_a"},
                {"candidate_id": "cand_a_dup", "recovery_link_id": "inj_a"},
            ],
            [recovered_a, recovered_duplicate],
            [injected],
        )
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["n_missed_injections"], 0)

    def test_deduplicated_member_link_is_available_for_test_b_label(self):
        truths = [
            truth("inj_a", "event_shared", "A"),
            truth("inj_b", "event_shared", "B"),
        ]
        entry = {
            "union_id": "u_members",
            "simultaneous_group_id": "group_1",
            "station_ids": ["A", "B"],
            "detection_state": "two_station",
            "stations": {
                "A": {
                    "detected": True,
                    "candidate": {"candidate_id": "representative_a"},
                    "deduplicated_member_ids": [
                        "representative_a",
                        "linked_duplicate_a",
                    ],
                },
                "B": {
                    "detected": True,
                    "candidate": {"candidate_id": "representative_b"},
                    "deduplicated_member_ids": ["representative_b"],
                },
            },
        }
        result = label_union_entry(
            entry,
            {item.injection_id: item for item in truths},
            _event_registry(truths),
            {"linked_duplicate_a": "inj_a", "representative_b": "inj_b"},
        )
        self.assertEqual(result["label"], 1)

    def test_paired_bootstrap_identical_methods_have_zero_delta(self):
        labels = np.asarray([0, 1] * 20, dtype=np.int8)
        scores = np.linspace(0.0, 1.0, len(labels))
        score_map = {
            "Raw Stage 3": scores,
            "Corrected Stage 3": scores,
            "Model-free filter": scores,
            "Stage 4": scores,
        }
        result = paired_bootstrap(
            labels,
            score_map,
            ["case"] * len(labels),
            ["two_station"] * len(labels),
            ["group"] * len(labels),
            n_boot=100,
            ci_level=0.95,
            seed=7,
            bootstrap_unit="pair_stratified",
        )
        delta = result["paired_deltas"]["stage4_minus_filter"]
        self.assertEqual(delta["delta_auc"], 0.0)
        self.assertEqual(delta["ci_lo"], 0.0)
        self.assertEqual(delta["ci_hi"], 0.0)

    def test_pipeline_denominators_are_not_conflated(self):
        events = [
            {
                "event_id": "e1",
                "pair_label": 1,
                "width_hz": 30,
                "entered_candidate_union": True,
            },
            {
                "event_id": "e2",
                "pair_label": 1,
                "width_hz": 30,
                "entered_candidate_union": False,
            },
            {
                "event_id": "e3",
                "pair_label": 1,
                "width_hz": 30,
                "entered_candidate_union": True,
            },
        ]
        joined = [
            {"truth_event_id": "e1", "truth_label": 1, "stage4_predicted_match": True},
        ]
        result = _pipeline_denominators(joined, events, 10, 100)
        self.assertAlmostEqual(result["bliss_union_entry_fraction"], 2 / 3)
        self.assertAlmostEqual(
            result["post_union_extraction_and_scoring_fraction"], 1 / 2
        )
        self.assertEqual(result["conditional_stage4_event_recall_among_scored"], 1.0)
        self.assertAlmostEqual(result["post_union_pipeline_recovery_fraction"], 1 / 2)
        self.assertAlmostEqual(result["end_to_end_final_recovery_fraction"], 1 / 3)

    def test_locked_evaluator_writes_direct_paired_comparison_and_denominators(self):
        """Exercise the complete post-inference evaluator on six independent blocks."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions, labels, events = [], [], []
            for index in range(60):
                label = 1 if index % 2 == 0 else 0
                event_id = "event_%03d" % index if label else None
                block_id = "block_%02d" % (index // 10)
                if label:
                    events.append(
                        {
                            "event_id": event_id,
                            "pair_label": 1,
                            "width_hz": [10, 20, 30, 50, 75, 100][index % 6],
                            # Index 58 represents a BLISS miss and therefore
                            # cannot appear in the candidate-union labels.
                            "entered_candidate_union": index != 58,
                            "population": "detected_conditioned",
                        }
                    )
                if index == 58:
                    continue
                # Raw scores deliberately overlap.  Filter and Stage 4 are
                # strongly ranked, with Stage 4 retaining a small incremental
                # advantage on the difficult examples.
                raw_score = -0.10 - 0.03 * ((index * 7) % 9)
                corrected_score = raw_score + (0.03 if label else -0.01)
                filter_score = (0.73 if label else 0.27) + 0.08 * math.sin(index)
                stage4_score = (0.90 if label else 0.10) + 0.04 * math.sin(index)
                prediction = {
                    "pair_id": "pair_%03d" % index,
                    "simultaneous_group_id": "group_proxy",
                    "resampling_block_id": block_id,
                    "anchor_width_hz": [10, 20, 30, 50, 75, 100][index % 6],
                    "anchor_snr": 8.0 + (index % 16),
                    "stage3_raw_score": raw_score,
                    "stage3_corrected_score": corrected_score,
                    "matched_filter_score": filter_score,
                    "stage4_score": stage4_score,
                    "stage4_predicted_match": stage4_score >= 0.5,
                    "stage4_validation_threshold": 0.5,
                }
                labels.append(
                    {
                        "pair_id": "pair_%03d" % index,
                        "label": label,
                        "case": "genuine_match" if label else "independent",
                        "event_id": event_id,
                        "detection_state": (
                            "two_station" if index % 3 else "one_station"
                        ),
                        "population": "detected_conditioned",
                        "evaluation_cell_id": "cell_%d" % (index % 6),
                        "resampling_block_id": block_id,
                        "injected_shape": "lorentzian",
                        "injected_width_hz": [10, 20, 30, 50, 75, 100][index % 6],
                    }
                )
                # Index 56 entered the BLISS union but was excluded during
                # extraction/scoring; its label remains in the locked union.
                if index != 56:
                    predictions.append(prediction)
            predictions_path = root / "predictions.jsonl"
            labels_path = root / "labels.jsonl"
            events_path = root / "events.jsonl"
            preregistration_path = root / "preregistration.json"
            output_directory = root / "evaluation"
            write_jsonl(str(predictions_path), predictions)
            write_jsonl(str(labels_path), labels)
            write_jsonl(str(events_path), events)
            write_json(
                str(preregistration_path),
                {
                    "status": "frozen_before_test_b",
                    "locked_inputs": {"population": "detected_conditioned"},
                },
            )
            os.environ.setdefault("MPLCONFIGDIR", str(root / "matplotlib"))
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_main(
                    Namespace(
                        predictions=str(predictions_path),
                        labels=str(labels_path),
                        events=str(events_path),
                        preregistration=str(preregistration_path),
                        out_dir=str(output_directory),
                        n_boot=80,
                        ci_level=0.95,
                        seed=17,
                        bootstrap_unit="auto",
                        stage3_margin=0.1833,
                        target_auc=0.80,
                        width_min=10.0,
                        width_max=100.0,
                        calibration_bins=6,
                        primary_population="detected_conditioned",
                    )
                )
            result = read_json_records(
                str(output_directory / "synthetic_test_b_evaluation.json")
            )[0]
            self.assertIn(
                "stage4_minus_filter", result["paired_bootstrap"]["paired_deltas"]
            )
            self.assertEqual(
                result["paired_bootstrap"]["bootstrap_unit"], "simultaneous_group"
            )
            self.assertEqual(
                result["pipeline_denominators"]["n_positive_injected_events"], 30
            )
            self.assertEqual(
                result["pipeline_denominators"]["n_entering_bliss_union"], 29
            )
            self.assertEqual(result["pipeline_denominators"]["n_scored_by_stage4"], 28)
            self.assertLessEqual(
                result["pipeline_denominators"][
                    "post_union_pipeline_recovery_fraction"
                ],
                1.0,
            )
            self.assertTrue(
                (output_directory / "synthetic_test_b_report.txt").is_file()
            )
            self.assertTrue((output_directory / "03_test_b_low_fpr_roc.png").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
