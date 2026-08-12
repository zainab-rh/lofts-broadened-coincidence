#!/usr/bin/env python3
"""Create Synthetic-Test-B union labels after label-blind frozen inference.

The independent injection harness may emit a segregated candidate-to-injection
link sidecar.  This script joins that sidecar to a separate truth registry.
Candidate records, union construction, extraction and inference remain free of
truth identifiers and do not import this module.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

from lofts_bliss_schema import (
    load_truth,
    read_json_records,
    sha256_file,
    write_json,
    write_jsonl,
)


def _event_registry(truth_records):
    events: Dict[str, Dict[str, Any]] = {}
    for truth in truth_records:
        if truth.pair_label is None:
            raise ValueError("truth %s is missing pair_label" % truth.injection_id)
        required_extras = (
            "population",
            "evaluation_cell_id",
            "resampling_block_id",
        )
        missing = [
            key for key in required_extras if not str(truth.extras.get(key, "")).strip()
        ]
        if missing:
            raise ValueError(
                "truth %s lacks locked Test-B fields %s" % (truth.injection_id, missing)
            )
        if truth.event_id not in events:
            events[truth.event_id] = {
                "event_id": truth.event_id,
                "pair_label": int(truth.pair_label),
                "case": truth.case,
                "shape": truth.shape,
                "width_hz": float(truth.width_hz),
                "snr_values": [float(truth.snr)],
                "injection_ids": [truth.injection_id],
                "simultaneous_group_ids": {truth.simultaneous_group_id},
                "station_ids": {truth.station_id},
                "population": str(truth.extras["population"]),
                "evaluation_cell_id": str(truth.extras["evaluation_cell_id"]),
                "resampling_block_id": str(truth.extras["resampling_block_id"]),
            }
        else:
            event = events[truth.event_id]
            if event["pair_label"] != int(truth.pair_label):
                raise ValueError(
                    "event %s has inconsistent pair labels" % truth.event_id
                )
            if event["case"] != truth.case:
                raise ValueError(
                    "event %s has inconsistent case labels" % truth.event_id
                )
            if event["shape"] != truth.shape:
                raise ValueError(
                    "event %s has inconsistent profile shapes" % truth.event_id
                )
            if not abs(event["width_hz"] - float(truth.width_hz)) <= max(
                1e-6, 1e-6 * event["width_hz"]
            ):
                raise ValueError(
                    "event %s has inconsistent injected widths" % truth.event_id
                )
            for key in ("population", "evaluation_cell_id", "resampling_block_id"):
                if event[key] != str(truth.extras[key]):
                    raise ValueError(
                        "event %s has inconsistent %s" % (truth.event_id, key)
                    )
            event["snr_values"].append(float(truth.snr))
            event["injection_ids"].append(truth.injection_id)
            event["simultaneous_group_ids"].add(truth.simultaneous_group_id)
            event["station_ids"].add(truth.station_id)
    for event in events.values():
        if len(event["simultaneous_group_ids"]) != 1:
            raise ValueError("event %s crosses simultaneous groups" % event["event_id"])
    return events


def _load_recovery_links(path: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for row in read_json_records(path):
        candidate_id = str(row.get("candidate_id", "")).strip()
        recovery_link_id = str(row.get("recovery_link_id", "")).strip()
        if not candidate_id or not recovery_link_id:
            raise ValueError(
                "recovery-link rows require candidate_id and recovery_link_id"
            )
        if candidate_id in result:
            raise ValueError("duplicate recovery link for candidate %s" % candidate_id)
        result[candidate_id] = recovery_link_id
    return result


def _candidate_links(
    station_entry: Mapping[str, Any], recovery_links: Mapping[str, str]
) -> List[str]:
    candidate = station_entry.get("candidate")
    if not station_entry.get("detected") or not candidate:
        return []
    candidate_ids = {
        str(candidate["candidate_id"]),
        *(str(value) for value in station_entry.get("deduplicated_member_ids", [])),
    }
    links = sorted(
        {
            str(recovery_links[candidate_id]).strip()
            for candidate_id in candidate_ids
            if candidate_id in recovery_links
            and str(recovery_links[candidate_id]).strip()
        }
    )
    if len(links) > 1:
        raise ValueError(
            "a deduplicated station hit group maps to multiple injected events: %s"
            % links
        )
    return links


def label_union_entry(
    entry: Mapping[str, Any],
    truth_by_id,
    events,
    recovery_links,
    expected_population: str = "detected_conditioned",
) -> Dict[str, Any]:
    detected_links = []
    for station_id in entry["station_ids"]:
        links = _candidate_links(entry["stations"][station_id], recovery_links)
        for link in links:
            if link not in truth_by_id:
                raise ValueError(
                    "union %s references unknown recovery_link_id %s"
                    % (entry["union_id"], link)
                )
            detected_links.append(link)
    linked_events = [truth_by_id[link].event_id for link in detected_links]
    unique_events = sorted(set(linked_events))
    detection_state = str(entry["detection_state"])
    label = 0
    case = "unmatched_bliss_false_positive"
    event_id = None
    if detection_state == "one_station":
        if len(unique_events) == 1:
            event_id = unique_events[0]
            label = int(events[event_id]["pair_label"])
            case = str(events[event_id]["case"])
        elif len(unique_events) > 1:
            raise ValueError("one-station union entry links multiple truth events")
    elif detection_state == "two_station":
        if len(detected_links) == 2 and len(unique_events) == 1:
            event_id = unique_events[0]
            label = int(events[event_id]["pair_label"])
            case = str(events[event_id]["case"])
        elif len(unique_events) > 1:
            label = 0
            case = "associated_distinct_injections"
        elif len(unique_events) == 1:
            # One recovered injection plus a local false hit is not evidence
            # that the reported station candidates identify the same event.
            event_id = unique_events[0]
            label = 0
            case = "true_hit_associated_with_unlinked_false_hit"
    else:
        raise ValueError("unknown detection_state %r" % detection_state)
    linked_event_records = [events[value] for value in unique_events]
    populations = sorted({value["population"] for value in linked_event_records})
    if populations and populations != [expected_population]:
        raise ValueError(
            "union %s contains population %s, expected only %s"
            % (entry["union_id"], populations, expected_population)
        )
    cells = sorted({value["evaluation_cell_id"] for value in linked_event_records})
    if len(cells) > 1:
        raise ValueError(
            "associated injected events cross locked evaluation cells: %s" % cells
        )
    blocks = sorted({value["resampling_block_id"] for value in linked_event_records})
    entry_block = str(entry.get("resampling_block_id") or "").strip()
    if blocks and entry_block and blocks != [entry_block]:
        raise ValueError(
            "union resampling block %r disagrees with linked truth block(s) %s"
            % (entry_block, blocks)
        )
    shapes = sorted({value["shape"] for value in linked_event_records})
    widths = sorted({float(value["width_hz"]) for value in linked_event_records})
    injected_snrs = [
        value for event in linked_event_records for value in event["snr_values"]
    ]
    return {
        "format_version": 1,
        "pair_id": str(entry["union_id"]),
        "union_id": str(entry["union_id"]),
        "simultaneous_group_id": str(entry["simultaneous_group_id"]),
        "label": int(label),
        "case": case,
        "event_id": event_id,
        "linked_injection_ids": detected_links,
        "linked_event_ids": unique_events,
        "detection_state": detection_state,
        "population": expected_population,
        "evaluation_cell_id": cells[0] if len(cells) == 1 else None,
        "resampling_block_id": entry_block or (blocks[0] if len(blocks) == 1 else None),
        "injected_shape": (
            shapes[0] if len(shapes) == 1 else ("mixed" if shapes else None)
        ),
        "injected_width_hz": widths[0] if len(widths) == 1 else None,
        "mean_injected_snr": (
            sum(injected_snrs) / float(len(injected_snrs)) if injected_snrs else None
        ),
    }


def main(args: argparse.Namespace) -> None:
    preregistration = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    if preregistration.get("status") != "frozen_before_test_b":
        raise ValueError("Test-B preregistration is not frozen")
    preregistered_population = preregistration.get("locked_inputs", {}).get(
        "population"
    )
    if preregistered_population != args.expected_population:
        raise ValueError(
            "expected population %r disagrees with preregistration %r"
            % (args.expected_population, preregistered_population)
        )
    union_entries = read_json_records(args.union)
    truths = load_truth(args.truth)
    recovery_links = _load_recovery_links(args.recovery_links)
    truth_by_id = {item.injection_id: item for item in truths}
    events = _event_registry(truths)
    populations = sorted({event["population"] for event in events.values()})
    if populations != [args.expected_population]:
        raise ValueError(
            "locked truth populations %s do not equal expected population %r; "
            "run different physical populations separately"
            % (populations, args.expected_population)
        )
    labels = [
        label_union_entry(
            entry, truth_by_id, events, recovery_links, args.expected_population
        )
        for entry in union_entries
    ]
    ids = [item["union_id"] for item in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("union IDs must be unique")
    write_jsonl(args.output, labels)

    entered_events = defaultdict(list)
    for label in labels:
        for event_id in label["linked_event_ids"]:
            entered_events[event_id].append(label["union_id"])
    event_rows = []
    for event_id, event in sorted(events.items()):
        event_rows.append(
            {
                "event_id": event_id,
                "pair_label": event["pair_label"],
                "case": event["case"],
                "shape": event["shape"],
                "width_hz": event["width_hz"],
                "mean_injected_snr": sum(event["snr_values"])
                / len(event["snr_values"]),
                "injection_ids": sorted(event["injection_ids"]),
                "station_ids": sorted(event["station_ids"]),
                "simultaneous_group_id": next(iter(event["simultaneous_group_ids"])),
                "population": event["population"],
                "evaluation_cell_id": event["evaluation_cell_id"],
                "resampling_block_id": event["resampling_block_id"],
                "entered_candidate_union": bool(entered_events.get(event_id)),
                "union_ids": sorted(entered_events.get(event_id, [])),
            }
        )
    event_path = str(Path(args.output).with_name("synthetic_test_b_events.jsonl"))
    write_jsonl(event_path, event_rows)
    summary = {
        "format_version": 1,
        "n_union_labels": len(labels),
        "n_match_labels": sum(item["label"] == 1 for item in labels),
        "n_mismatch_labels": sum(item["label"] == 0 for item in labels),
        "n_injected_events": len(events),
        "n_positive_injected_events": sum(
            item["pair_label"] == 1 for item in event_rows
        ),
        "n_positive_events_entering_union": sum(
            item["pair_label"] == 1 and item["entered_candidate_union"]
            for item in event_rows
        ),
        "union_path": str(Path(args.union).resolve()),
        "union_sha256": sha256_file(args.union),
        "truth_path": str(Path(args.truth).resolve()),
        "truth_sha256": sha256_file(args.truth),
        "recovery_links_path": str(Path(args.recovery_links).resolve()),
        "recovery_links_sha256": sha256_file(args.recovery_links),
        "event_registry": event_path,
        "label_creation_order": "run only after frozen label-blind inference",
        "expected_population": args.expected_population,
        "preregistration": str(Path(args.preregistration).resolve()),
        "preregistration_sha256": sha256_file(args.preregistration),
    }
    write_json(str(Path(args.output).with_suffix(".summary.json")), summary)
    print("Wrote %d Synthetic-Test-B union labels to %s" % (len(labels), args.output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join the frozen BLISS union to segregated synthetic truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--union", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument(
        "--recovery-links",
        required=True,
        help="segregated candidate-to-injection link sidecar; never used by inference",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument(
        "--expected-population",
        choices=("detected_conditioned", "fixed_power"),
        default="detected_conditioned",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
