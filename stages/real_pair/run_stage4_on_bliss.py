#!/usr/bin/env python3
"""Run the frozen Stage-4 checkpoint on an operational BLISS candidate union.

The output is a ranked candidate table, not an end-to-end detection claim and
not a calibrated astrophysical posterior.  All four downstream comparators use
the exact same extracted station pair: raw Stage 3, corrected Stage 3,
transparent central filter statistic, and frozen Stage 4.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from lofts_bliss_schema import (
    atomic_write_text,
    read_json_records,
    sha256_file,
    write_json,
    write_jsonl,
)


def _stage3_preprocess(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    if float(out.std()) == 0.0:
        return out
    out = out - np.median(out, axis=1, keepdims=True)
    return out / (out.std() + 1e-6)


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _import_stage4(code_dir: str):
    resolved = str(Path(code_dir).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    try:
        import torch
        import torch.nn.functional as functional
        from candidate_preprocessing import (
            CandidateParams,
            centre_column_statistic,
            make_candidate_view,
        )
        from stage4_model import load_stage3_backbone, load_stage4_checkpoint
        from torch.cuda.amp import autocast
    except ImportError as exc:
        raise ImportError(
            "Stage-4 imports failed. --stage4-code-dir must contain "
            "candidate_preprocessing.py and stage4_model.py, and the Stage-3 "
            "directory containing train.py must be on PYTHONPATH."
        ) from exc
    return (
        torch,
        functional,
        autocast,
        CandidateParams,
        centre_column_statistic,
        make_candidate_view,
        load_stage3_backbone,
        load_stage4_checkpoint,
    )


def _encode_stage3(backbone, x, functional):
    e1 = backbone.enc1(x)
    e2 = backbone.enc2(backbone.pool(e1))
    e3 = backbone.enc3(backbone.pool(e2))
    bottleneck = backbone.bottleneck(backbone.pool(e3))
    return functional.normalize(backbone.projection_head(bottleneck), p=2, dim=1)


def _params(station: Mapping[str, Any], CandidateParams):
    return CandidateParams(
        center_channel=float(station["candidate_center_channel"]),
        drift_hz_per_s=float(station["reported_drift_hz_s"]),
        width_hz=float(station["reported_width_fwhm_hz"]),
        channel_step_hz=float(station["signed_foff_hz"]),
        dt_s=float(station["tsamp_s"]),
        reference_row=float(station["candidate_reference_row"]),
    )


def _load_pair(
    record: Mapping[str, Any], verify_hash: bool
) -> Tuple[np.ndarray, np.ndarray]:
    path = str(record["array_file"])
    if verify_hash and sha256_file(path) != str(record["array_sha256"]):
        raise ValueError("pair archive checksum mismatch: %s" % path)
    with np.load(path, allow_pickle=False) as archive:
        raw_a = np.asarray(archive["raw_a"], dtype=np.float32)
        raw_b = np.asarray(archive["raw_b"], dtype=np.float32)
        stored_union = str(np.asarray(archive["union_id"]).item())
    if stored_union != str(record["union_id"]):
        raise ValueError("pair archive union ID mismatch")
    if raw_a.shape != raw_b.shape or raw_a.ndim != 2:
        raise ValueError("station arrays must have the same two-dimensional shape")
    if not np.isfinite(raw_a).all() or not np.isfinite(raw_b).all():
        raise ValueError("pair archive contains non-finite values")
    return raw_a, raw_b


def _plot_qa(
    path: Path,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    view_a: np.ndarray,
    view_b: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    raw_values = np.concatenate((raw_a.ravel(), raw_b.ravel()))
    raw_lo, raw_hi = np.quantile(raw_values, [0.01, 0.99])
    view_values = np.concatenate((view_a.ravel(), view_b.ravel()))
    limit = max(
        abs(float(np.quantile(view_values, 0.01))),
        abs(float(np.quantile(view_values, 0.99))),
    )
    for axis, data, subtitle in zip(
        axes[0], (raw_a, raw_b), ("Station A: raw", "Station B: raw")
    ):
        axis.imshow(data, origin="lower", aspect="auto", vmin=raw_lo, vmax=raw_hi)
        axis.set_title(subtitle)
    for axis, data, subtitle in zip(
        axes[1],
        (view_a, view_b),
        ("Station A: de-chirped + integrated", "Station B: de-chirped + integrated"),
    ):
        axis.imshow(
            data,
            origin="lower",
            aspect="auto",
            vmin=-limit,
            vmax=limit,
            cmap="coolwarm",
        )
        axis.set_title(subtitle)
        axis.set_xlabel("Local frequency channel")
    axes[0, 0].set_ylabel("Time sample")
    axes[1, 0].set_ylabel("Time sample")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if (
        args.dataset_role == "unlabeled_real_pair"
        and not args.acknowledge_synthetic_threshold_is_exploratory
    ):
        raise ValueError(
            "real-pair inference requires an explicit acknowledgement that the "
            "frozen synthetic threshold is exploratory and not deployment calibrated"
        )
    (
        torch,
        functional,
        autocast,
        CandidateParams,
        centre_column_statistic,
        make_candidate_view,
        load_stage3_backbone,
        load_stage4_checkpoint,
    ) = _import_stage4(args.stage4_code_dir)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cpu":
        if args.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
        except RuntimeError:
            # PyTorch permits this setting only before inter-op work starts.
            # The intra-op limit above is the important reproducibility/resource
            # control and remains in force if an embedding already initialised it.
            pass
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        print("WARNING: --amp requested but CUDA is unavailable; using CPU float32.")

    stage3 = load_stage3_backbone(args.stage3_checkpoint, device="cpu").to(device)
    stage3.eval()
    stage4, checkpoint = load_stage4_checkpoint(args.stage4_checkpoint, device=device)
    preprocessing = checkpoint.get("preprocessing", {})
    if not bool(preprocessing.get("signed_foff_required", False)):
        raise ValueError("checkpoint does not assert signed-foff preprocessing")
    integration = str(preprocessing.get("integration", "boxcar"))
    if args.integration and args.integration != integration:
        raise ValueError(
            "frozen checkpoint uses integration=%r; refusing inference override %r"
            % (integration, args.integration)
        )
    validation = checkpoint.get("validation_operating_point", {})
    if "threshold" not in validation:
        raise ValueError("checkpoint lacks its validation-selected operating threshold")
    stage4_threshold = float(validation["threshold"])

    pair_records = read_json_records(args.pair_manifest)
    if not pair_records:
        raise ValueError(
            "pair manifest is empty; refusing to emit an empty inference result"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = out_dir / "qa_preprocessing"
    if args.qa_count:
        qa_dir.mkdir(parents=True, exist_ok=True)
    saved_views_dir = out_dir / "preprocessed_pairs"
    if args.save_preprocessed:
        saved_views_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    symmetry_max = 0.0
    started = time.time()
    for batch_start in range(0, len(pair_records), args.batch_size):
        batch_records = pair_records[batch_start : batch_start + args.batch_size]
        batch_data = []
        for record in batch_records:
            raw_a, raw_b = _load_pair(record, not args.skip_hash_verification)
            station_a, station_b = record["station_order"]
            params_a = _params(record["stations"][station_a], CandidateParams)
            params_b = _params(record["stations"][station_b], CandidateParams)
            view_a = make_candidate_view(
                raw_a,
                params_a,
                integration=integration,
                remove_static_bandpass=bool(
                    preprocessing.get("remove_static_bandpass", True)
                ),
            )
            view_b = make_candidate_view(
                raw_b,
                params_b,
                integration=integration,
                remove_static_bandpass=bool(
                    preprocessing.get("remove_static_bandpass", True)
                ),
            )
            if args.save_preprocessed:
                np.savez_compressed(
                    str(saved_views_dir / (str(record["pair_id"]) + ".npz")),
                    view_a=view_a,
                    view_b=view_b,
                )
            if len(rows) + len(batch_data) < args.qa_count:
                _plot_qa(
                    qa_dir / (str(record["pair_id"]) + ".png"),
                    raw_a,
                    raw_b,
                    view_a,
                    view_b,
                    "%s | %s" % (record["pair_id"], record["detection_state"]),
                )
            batch_data.append((record, raw_a, raw_b, view_a, view_b))

        raw_inputs = np.concatenate(
            (
                np.stack([_stage3_preprocess(item[1]) for item in batch_data]),
                np.stack([_stage3_preprocess(item[2]) for item in batch_data]),
                np.stack([_stage3_preprocess(item[3]) for item in batch_data]),
                np.stack([_stage3_preprocess(item[4]) for item in batch_data]),
            ),
            axis=0,
        )[:, None, :, :]
        tensor = torch.from_numpy(raw_inputs).to(device)
        n = len(batch_data)
        with torch.no_grad(), autocast(enabled=use_amp):
            latent = _encode_stage3(stage3, tensor, functional)
        raw_a_z, raw_b_z, corrected_a_z, corrected_b_z = torch.split(latent, n, dim=0)
        raw_scores = -torch.linalg.vector_norm(raw_a_z - raw_b_z, dim=1)
        corrected_scores = -torch.linalg.vector_norm(
            corrected_a_z - corrected_b_z, dim=1
        )
        view_a_tensor = torch.from_numpy(
            np.stack([item[3] for item in batch_data])[:, None]
        ).to(device)
        view_b_tensor = torch.from_numpy(
            np.stack([item[4] for item in batch_data])[:, None]
        ).to(device)
        with torch.no_grad(), autocast(enabled=use_amp):
            logits_ab, _, _, _ = stage4(view_a_tensor, view_b_tensor)
            logits_ba, _, _, _ = stage4(view_b_tensor, view_a_tensor)
        score_ab = torch.sigmoid(logits_ab).float().cpu().numpy()
        score_ba = torch.sigmoid(logits_ba).float().cpu().numpy()
        symmetry_max = max(symmetry_max, float(np.max(np.abs(score_ab - score_ba))))
        if symmetry_max > args.symmetry_tolerance:
            raise RuntimeError(
                "station-order symmetry audit failed: max |score(A,B)-score(B,A)|=%.3g"
                % symmetry_max
            )
        raw_scores_np = raw_scores.float().cpu().numpy()
        corrected_scores_np = corrected_scores.float().cpu().numpy()
        for index, (record, raw_a, raw_b, view_a, view_b) in enumerate(batch_data):
            station_a, station_b = record["station_order"]
            station_a_extras = record["stations"][station_a].get(
                "candidate_operational_extras", {}
            )
            station_b_extras = record["stations"][station_b].get(
                "candidate_operational_extras", {}
            )
            above_synthetic_threshold = bool(score_ab[index] >= stage4_threshold)
            association = record.get("association") or {}
            rows.append(
                {
                    "pair_id": record["pair_id"],
                    "union_id": record["union_id"],
                    "simultaneous_group_id": record["simultaneous_group_id"],
                    "detection_state": record["detection_state"],
                    "operational_eligibility": record.get("operational_eligibility"),
                    "route": record.get("route"),
                    "contains_broadband_rfi_like": bool(
                        record.get("contains_broadband_rfi_like", False)
                    ),
                    "control_kind": record.get("control_kind", "primary_observed"),
                    "control_shift_hz": record.get("control_shift_hz"),
                    "source_pair_id": record.get("source_pair_id", record["pair_id"]),
                    "resampling_block_id": record.get("resampling_block_id"),
                    "station_a_id": station_a,
                    "station_b_id": station_b,
                    "station_a_detected": bool(
                        record["stations"][station_a]["detected_by_bliss"]
                    ),
                    "station_b_detected": bool(
                        record["stations"][station_b]["detected_by_bliss"]
                    ),
                    "anchor_station_id": record["anchor_station_id"],
                    "anchor_width_hz": float(
                        record["stations"][record["anchor_station_id"]][
                            "reported_width_fwhm_hz"
                        ]
                    ),
                    "anchor_snr": float(
                        record["stations"][record["anchor_station_id"]]["reported_snr"]
                    ),
                    "station_a_native_width_channels": station_a_extras.get(
                        "native_width_channels"
                    ),
                    "station_b_native_width_channels": station_b_extras.get(
                        "native_width_channels"
                    ),
                    "station_a_source_flag": station_a_extras.get("source_flag", ""),
                    "station_b_source_flag": station_b_extras.get("source_flag", ""),
                    "station_a_broadband_rfi_like": station_a_extras.get(
                        "broadband_rfi_like"
                    ),
                    "station_b_broadband_rfi_like": station_b_extras.get(
                        "broadband_rfi_like"
                    ),
                    "association_frequency_delta_hz": association.get(
                        "frequency_delta_hz"
                    ),
                    "association_drift_delta_hz_s": association.get("drift_delta_hz_s"),
                    "association_log_width_delta": association.get("log_width_delta"),
                    "stage3_raw_score": float(raw_scores_np[index]),
                    "stage3_corrected_score": float(corrected_scores_np[index]),
                    "matched_filter_score": float(
                        min(
                            centre_column_statistic(view_a),
                            centre_column_statistic(view_b),
                        )
                    ),
                    "stage4_score": float(score_ab[index]),
                    "stage4_predicted_match": (
                        above_synthetic_threshold
                        if args.dataset_role == "synthetic_test_b"
                        else None
                    ),
                    "stage4_above_frozen_synthetic_threshold": above_synthetic_threshold,
                    "stage4_validation_threshold": stage4_threshold,
                    "score_semantics": (
                        "ranking score; not an astrophysical posterior; the frozen "
                        "threshold is not real-data calibrated"
                        if args.dataset_role == "unlabeled_real_pair"
                        else "ranking score; not an astrophysical posterior"
                    ),
                }
            )
    ranked = sorted(rows, key=lambda item: (-item["stage4_score"], item["pair_id"]))
    for rank, row in enumerate(ranked, 1):
        row["stage4_priority_rank"] = rank
    _write_csv(str(out_dir / "stage4_bliss_predictions.csv"), ranked)
    write_jsonl(str(out_dir / "stage4_bliss_predictions.jsonl"), ranked)
    summary = {
        "format_version": 1,
        "n_scored_union_pairs": len(rows),
        "device": str(device),
        "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
        "amp_enabled": use_amp,
        "integration": integration,
        "stage4_validation_threshold": stage4_threshold,
        "dataset_role": args.dataset_role,
        "n_predicted_matches": (
            sum(bool(item["stage4_predicted_match"]) for item in rows)
            if args.dataset_role == "synthetic_test_b"
            else None
        ),
        "n_above_frozen_synthetic_threshold": sum(
            bool(item["stage4_above_frozen_synthetic_threshold"]) for item in rows
        ),
        "station_order_symmetry_max_abs_score_difference": symmetry_max,
        "pair_manifest": str(Path(args.pair_manifest).resolve()),
        "pair_manifest_sha256": sha256_file(args.pair_manifest),
        "stage3_checkpoint": str(Path(args.stage3_checkpoint).resolve()),
        "stage3_checkpoint_sha256": sha256_file(args.stage3_checkpoint),
        "stage4_checkpoint": str(Path(args.stage4_checkpoint).resolve()),
        "stage4_checkpoint_sha256": sha256_file(args.stage4_checkpoint),
        "elapsed_seconds": time.time() - started,
        "labels_used": False,
        "scientific_boundary": (
            "Real-pair scores are unlabeled rankings. They do not measure AUC, "
            "recall, false-positive rate, deployment precision, or end-to-end "
            "BLISS completeness; the Test-A threshold is shown only as a frozen "
            "synthetic reference."
            if args.dataset_role == "unlabeled_real_pair"
            else "Scores prioritize entries already admitted to the BLISS candidate "
            "union; they do not measure end-to-end BLISS completeness."
        ),
    }
    write_json(str(out_dir / "stage4_bliss_inference_summary.json"), summary)
    print(
        "Scored %d BLISS-union pairs on %s; ranked output: %s"
        % (len(rows), device, out_dir / "stage4_bliss_predictions.csv")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score paired BLISS-union cutouts with the frozen Stage-4 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--stage3-checkpoint", required=True)
    parser.add_argument("--stage4-checkpoint", required=True)
    parser.add_argument("--stage4-code-dir", default=".")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--integration", default=None)
    parser.add_argument("--symmetry-tolerance", type=float, default=1e-6)
    parser.add_argument("--qa-count", type=int, default=20)
    parser.add_argument("--save-preprocessed", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument(
        "--dataset-role",
        choices=("synthetic_test_b", "unlabeled_real_pair"),
        default="synthetic_test_b",
    )
    parser.add_argument(
        "--acknowledge-synthetic-threshold-is-exploratory",
        action="store_true",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
