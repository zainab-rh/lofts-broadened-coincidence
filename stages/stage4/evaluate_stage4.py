#!/usr/bin/env python3
"""Paired, leakage-resistant evaluation of the Stage-4 improvement.

Every method sees the same generated pair:

stage3_raw
    Original Stage-3 encoder on the original raw tile (the deployed baseline).
stage3_corrected
    Original Stage-3 encoder after corrected signed de-chirp/scrunch, without
    retraining (isolates preprocessing alone).
matched_filter
    Model-free weaker-station centre statistic after preprocessing (a useful
    diagnostic that the detector-informed view itself contains signal).
stage4
    Fine-tuned candidate-conditioned pair classifier.

Bootstrap resampling is paired across methods and stratified by class/case.
For the pooled 10--100 Hz endpoint it is additionally stratified by width and
shape, so a change in mixture cannot masquerade as an AUC gain. Thresholds
are never selected on test data: Stage-4 uses the validation threshold stored
inside its checkpoint, and Stage-3 uses the supplied historical margin only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lofts_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from candidate_preprocessing import centre_column_statistic
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from stage4_data import CandidatePairFactory
from stage4_model import load_stage3_backbone, load_stage4_checkpoint
from torch.cuda.amp import autocast

METHODS = ("stage3_raw", "stage3_corrected", "matched_filter", "stage4")
METHOD_LABELS = {
    "stage3_raw": "Stage 3 — raw",
    "stage3_corrected": "Stage 3 — corrected preprocessing",
    "matched_filter": "Model-free filter statistic",
    "stage4": "Stage 4 — fine-tuned",
}
METHOD_COLOURS = {
    "stage3_raw": "#4C78A8",
    "stage3_corrected": "#F58518",
    "matched_filter": "#54A24B",
    "stage4": "#E45756",
}


def parse_csv_floats(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return values


def parse_csv_choices(text: str, allowed: Sequence[str]) -> List[str]:
    values = [item.strip().lower() for item in text.split(",") if item.strip()]
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            "expected comma-separated values from %s; unknown=%s"
            % (list(allowed), unknown)
        )
    return values


def parse_probability_map(text: str, allowed: Sequence[str]) -> Dict[str, float]:
    result = {}
    try:
        for item in text.split(","):
            key, value = item.split(":", 1)
            result[key.strip().lower()] = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected case:p,case:p") from exc
    if set(result) - set(allowed):
        raise argparse.ArgumentTypeError(
            "unknown cases: %s" % sorted(set(result) - set(allowed))
        )
    if any(not np.isfinite(value) or value < 0 for value in result.values()):
        raise argparse.ArgumentTypeError(
            "case probabilities must be finite and non-negative"
        )
    if sum(result.values()) <= 0:
        raise argparse.ArgumentTypeError(
            "at least one case probability must be positive"
        )
    total = float(sum(result.values()))
    return {key: result.get(key, 0.0) / total for key in allowed}


def allocate_counts(total: int, probabilities: Dict[str, float]) -> Dict[str, int]:
    """Largest-remainder allocation with an exact requested total."""

    raw = {key: total * value for key, value in probabilities.items()}
    counts = {key: int(np.floor(value)) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def stage3_preprocess(array: np.ndarray) -> np.ndarray:
    """Mathematical equivalent of the supplied train.preprocess_np."""

    out = np.asarray(array, dtype=np.float32)
    if float(out.std()) == 0.0:
        return out
    out = out - np.median(out, axis=1, keepdims=True)
    return out / (out.std() + 1e-6)


def encode_stage3(backbone, x):
    """Run the original encoder/projection path without the unused decoder."""

    e1 = backbone.enc1(x)
    e2 = backbone.enc2(backbone.pool(e1))
    e3 = backbone.enc3(backbone.pool(e2))
    bottleneck = backbone.bottleneck(backbone.pool(e3))
    return F.normalize(backbone.projection_head(bottleneck), p=2, dim=1)


@torch.no_grad()
def score_batch(
    items: Sequence[Dict[str, object]],
    stage3,
    stage4,
    device,
    use_amp: bool,
) -> Dict[str, np.ndarray]:
    """Score one in-memory batch, then release its large arrays."""

    if not items:
        return {method: np.empty(0, dtype=np.float64) for method in METHODS}
    raw_a = np.stack([stage3_preprocess(item["raw_a"]) for item in items])
    raw_b = np.stack([stage3_preprocess(item["raw_b"]) for item in items])
    corrected_a = np.stack([stage3_preprocess(item["view_a"]) for item in items])
    corrected_b = np.stack([stage3_preprocess(item["view_b"]) for item in items])
    stage3_input = np.concatenate((raw_a, raw_b, corrected_a, corrected_b), axis=0)[
        :, None, :, :
    ]
    tensor = torch.from_numpy(stage3_input).to(device, non_blocking=True)
    n = len(items)
    with autocast(enabled=use_amp):
        z = encode_stage3(stage3, tensor)
    z_raw_a, z_raw_b, z_cor_a, z_cor_b = torch.split(z, n, dim=0)
    raw_score = -torch.linalg.vector_norm(z_raw_a - z_raw_b, dim=1)
    corrected_score = -torch.linalg.vector_norm(z_cor_a - z_cor_b, dim=1)

    view_a_np = np.stack([item["view_a"] for item in items])[:, None, :, :]
    view_b_np = np.stack([item["view_b"] for item in items])[:, None, :, :]
    view_a = torch.from_numpy(view_a_np).to(device, non_blocking=True)
    view_b = torch.from_numpy(view_b_np).to(device, non_blocking=True)
    with autocast(enabled=use_amp):
        logits, _, _, _ = stage4(view_a, view_b)
    stage4_score = torch.sigmoid(logits)

    # A true coincidence is limited by evidence at the weaker station.
    filter_score = np.asarray(
        [
            min(
                centre_column_statistic(item["view_a"]),
                centre_column_statistic(item["view_b"]),
            )
            for item in items
        ],
        dtype=np.float64,
    )
    return {
        "stage3_raw": raw_score.float().cpu().numpy().astype(np.float64),
        "stage3_corrected": (corrected_score.float().cpu().numpy().astype(np.float64)),
        "matched_filter": filter_score,
        "stage4": stage4_score.float().cpu().numpy().astype(np.float64),
    }


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann--Whitney AUC with average ranks for exact ties."""

    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    good = np.isfinite(s)
    y, s = y[good], s[good]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def point_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "auc_roc": auc_rank(labels, scores),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    predicted = np.asarray(scores) >= float(threshold)
    truth = np.asarray(labels).astype(bool)
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def paired_stratified_bootstrap(
    labels: np.ndarray,
    scores: Dict[str, np.ndarray],
    strata: np.ndarray,
    n_boot: int,
    ci_level: float,
    seed: int,
    reference: str = "stage3_raw",
) -> Dict[str, object]:
    """Paired AUC bootstrap preserving every stratum's sample count."""

    labels = np.asarray(labels, dtype=np.int8)
    strata = np.asarray(strata)
    if labels.shape != strata.shape:
        raise ValueError("labels and strata must have the same shape")
    if not (0 < ci_level < 1):
        raise ValueError("ci_level must be in (0, 1)")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    for method in METHODS:
        if np.asarray(scores[method]).shape != labels.shape:
            raise ValueError("score length mismatch for %s" % method)

    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    rng = np.random.default_rng(seed)
    draws = {method: np.empty(n_boot, dtype=np.float64) for method in METHODS}
    for boot in range(n_boot):
        sampled = np.concatenate(
            [rng.choice(group, size=group.size, replace=True) for group in groups]
        )
        y = labels[sampled]
        for method in METHODS:
            draws[method][boot] = auc_rank(y, scores[method][sampled])

    alpha = (1.0 - ci_level) / 2.0
    quantiles = (alpha, 1.0 - alpha)
    result = {"methods": {}, "deltas": {}}
    for method in METHODS:
        lo, hi = np.nanquantile(draws[method], quantiles)
        result["methods"][method] = {
            "auc": auc_rank(labels, scores[method]),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        }
        if method != reference:
            delta_draw = draws[method] - draws[reference]
            delta_lo, delta_hi = np.nanquantile(delta_draw, quantiles)
            result["deltas"][method] = {
                "delta_auc": float(
                    result["methods"][method]["auc"]
                    - result["methods"][reference]["auc"]
                ),
                "ci_lo": float(delta_lo),
                "ci_hi": float(delta_hi),
            }
    return result


def make_negative_cases(n: int, probabilities: Dict[str, float]) -> List[str]:
    counts = allocate_counts(n, probabilities)
    cases = []
    for case in ("onesided", "independent", "noise"):
        cases.extend([case] * counts.get(case, 0))
    rng = np.random.default_rng(918273)
    rng.shuffle(cases)
    return cases


def sample_with_retry(
    factory: CandidatePairFactory,
    seed: int,
    width_hz: float,
    shape: str,
    case: str,
    max_attempts: int = 20,
) -> Tuple[Dict[str, object], int]:
    actual_seed = int(seed)
    for attempt in range(max_attempts):
        try:
            return (
                factory.sample(
                    actual_seed,
                    fixed_width_hz=width_hz,
                    fixed_shape=shape,
                    fixed_case=case,
                    return_raw=True,
                ),
                actual_seed,
            )
        except ValueError as exc:
            message = str(exc).lower()
            if "truncat" not in message and "no mass" not in message:
                raise
            actual_seed = int((seed + (attempt + 1) * 104_729) % (2**63 - 1))
    raise RuntimeError(
        "failed to obtain an untruncated sample after %d attempts" % max_attempts
    )


def evaluate_cell(
    factory: CandidatePairFactory,
    width_hz: float,
    shape: str,
    n_per_class: int,
    negative_probabilities: Dict[str, float],
    stage3,
    stage4,
    device,
    use_amp: bool,
    batch_size: int,
    seed: int,
    manifest_handle,
) -> Dict[str, object]:
    cases = ["match"] * n_per_class + make_negative_cases(
        n_per_class, negative_probabilities
    )
    labels = np.asarray(
        [1 if case == "match" else 0 for case in cases],
        dtype=np.int8,
    )
    accumulated = {method: [] for method in METHODS}
    pending = []

    for order, case in enumerate(cases):
        proposed_seed = int((seed + order * 97_409) % (2**63 - 1))
        item, actual_seed = sample_with_retry(
            factory, proposed_seed, width_hz, shape, case
        )
        record = item["record"].to_dict()
        record.update(
            {
                "evaluation_order": order,
                "proposed_seed": proposed_seed,
                "actual_seed": actual_seed,
            }
        )
        manifest_handle.write(json.dumps(record, sort_keys=True) + "\n")
        pending.append(item)
        if len(pending) >= batch_size or order == len(cases) - 1:
            batch_scores = score_batch(
                pending, stage3, stage4, device=device, use_amp=use_amp
            )
            for method in METHODS:
                accumulated[method].append(batch_scores[method])
            pending.clear()

    scores = {
        method: np.concatenate(accumulated[method]).astype(np.float64)
        for method in METHODS
    }
    return {
        "labels": labels,
        "cases": np.asarray(cases),
        "scores": scores,
    }


def flatten_cell_summary(
    regime: str,
    shape: str,
    width_hz: float,
    data: Dict[str, object],
    bootstrap: Dict[str, object],
    stage3_margin: float,
    stage4_threshold: Optional[float],
) -> Dict[str, object]:
    labels = data["labels"]
    scores = data["scores"]
    row = {
        "snr_mode": regime,
        "shape": shape,
        "width_hz": float(width_hz),
        "n_match": int(np.sum(labels == 1)),
        "n_mismatch": int(np.sum(labels == 0)),
    }
    for method in METHODS:
        point = point_metrics(labels, scores[method])
        interval = bootstrap["methods"][method]
        row["%s_auc" % method] = point["auc_roc"]
        row["%s_auc_ci_lo" % method] = interval["ci_lo"]
        row["%s_auc_ci_hi" % method] = interval["ci_hi"]
        row["%s_average_precision" % method] = point["average_precision"]
    for method, delta in bootstrap["deltas"].items():
        row["delta_%s_minus_raw" % method] = delta["delta_auc"]
        row["delta_%s_minus_raw_ci_lo" % method] = delta["ci_lo"]
        row["delta_%s_minus_raw_ci_hi" % method] = delta["ci_hi"]

    raw_operating = threshold_metrics(labels, scores["stage3_raw"], -stage3_margin)
    for key, value in raw_operating.items():
        row["stage3_raw_%s" % key] = value
    if stage4_threshold is not None:
        stage4_operating = threshold_metrics(labels, scores["stage4"], stage4_threshold)
        for key, value in stage4_operating.items():
            row["stage4_%s" % key] = value
    return row


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_cells(rows: List[Dict[str, object]], out_dir: Path) -> None:
    for regime in sorted({row["snr_mode"] for row in rows}):
        for shape in sorted({row["shape"] for row in rows}):
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["snr_mode"] == regime and row["shape"] == shape
                ],
                key=lambda row: row["width_hz"],
            )
            widths = np.asarray([row["width_hz"] for row in selected])
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
            for method in METHODS:
                auc = np.asarray([row["%s_auc" % method] for row in selected])
                lo = np.asarray([row["%s_auc_ci_lo" % method] for row in selected])
                hi = np.asarray([row["%s_auc_ci_hi" % method] for row in selected])
                axes[0].plot(
                    widths,
                    auc,
                    marker="o",
                    lw=2,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLOURS[method],
                )
                axes[0].fill_between(
                    widths,
                    lo,
                    hi,
                    color=METHOD_COLOURS[method],
                    alpha=0.12,
                )
            axes[0].axhline(
                0.5,
                color="black",
                ls="--",
                lw=1,
                label="Chance",
            )
            axes[0].set_xscale("log")
            axes[0].set_ylim(0.35, 1.02)
            axes[0].set_xlabel("Broadening FWHM (Hz)")
            axes[0].set_ylabel("AUC-ROC")
            axes[0].grid(alpha=0.25, which="both")
            axes[0].legend(fontsize=8)

            delta = np.asarray([row["delta_stage4_minus_raw"] for row in selected])
            delta_lo = np.asarray(
                [row["delta_stage4_minus_raw_ci_lo"] for row in selected]
            )
            delta_hi = np.asarray(
                [row["delta_stage4_minus_raw_ci_hi"] for row in selected]
            )
            axes[1].errorbar(
                widths,
                delta,
                yerr=(delta - delta_lo, delta_hi - delta),
                marker="o",
                capsize=4,
                lw=2,
                color=METHOD_COLOURS["stage4"],
            )
            axes[1].axhline(0.0, color="black", ls="--", lw=1)
            axes[1].set_xscale("log")
            axes[1].set_xlabel("Broadening FWHM (Hz)")
            axes[1].set_ylabel("Paired ΔAUC (Stage 4 − raw Stage 3)")
            axes[1].grid(alpha=0.25, which="both")
            fig.suptitle("%s profile; %s-S/N population" % (shape.capitalize(), regime))
            fig.tight_layout()
            fig.savefig(
                out_dir / ("auc_%s_%s.png" % (regime, shape)),
                dpi=180,
            )
            plt.close(fig)


def pooled_summary(
    cell_data: Dict[Tuple[str, str, float], Dict[str, object]],
    regime: str,
    width_low: float,
    width_high: float,
    n_boot: int,
    ci_level: float,
    seed: int,
    target_auc: float,
) -> Dict[str, object]:
    selected = [
        (key, value)
        for key, value in cell_data.items()
        if key[0] == regime and width_low <= key[2] <= width_high
    ]
    if not selected:
        raise ValueError("no cells fall inside the primary width interval")
    labels = np.concatenate([value["labels"] for _, value in selected])
    scores = {
        method: np.concatenate([value["scores"][method] for _, value in selected])
        for method in METHODS
    }
    strata_parts = []
    for key, value in selected:
        _, shape, width = key
        strata_parts.append(
            np.asarray(
                [
                    "%s|%.9g|%d|%s" % (shape, width, int(label), case)
                    for label, case in zip(value["labels"], value["cases"])
                ]
            )
        )
    strata = np.concatenate(strata_parts)
    boot = paired_stratified_bootstrap(
        labels,
        scores,
        strata,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
    )
    delta = boot["deltas"]["stage4"]
    stage4_auc = boot["methods"]["stage4"]["auc"]
    return {
        "snr_mode": regime,
        "width_interval_hz": [
            float(width_low),
            float(width_high),
        ],
        "shapes": sorted({key[1] for key, _ in selected}),
        "n_pairs": int(labels.size),
        "methods": boot["methods"],
        "deltas_vs_stage3_raw": boot["deltas"],
        "success_criteria": {
            "target_auc": float(target_auc),
            "stage4_auc_at_least_target": bool(stage4_auc >= target_auc),
            "paired_delta_ci_excludes_zero": bool(delta["ci_lo"] > 0.0),
            "primary_success": bool(stage4_auc >= target_auc and delta["ci_lo"] > 0.0),
        },
    }


def main(args) -> None:
    if args.n_per_class <= 0 or args.batch_size <= 0:
        raise ValueError("n_per_class and batch_size must be positive")
    if not (0.5 < args.target_auc <= 1.0):
        raise ValueError("target_auc must be in (0.5, 1]")
    if args.snr_min is not None and args.snr_min <= 0:
        raise ValueError("snr_min must be positive")
    if args.snr_max is not None and args.snr_max <= 0:
        raise ValueError("snr_max must be positive")
    if "detected" not in args.snr_modes:
        raise ValueError(
            "snr_modes must include 'detected' because it is the registered "
            "primary post-BLISS endpoint"
        )
    if len(set(args.widths)) != len(args.widths):
        raise ValueError("widths must be unique")
    if len(set(args.shapes)) != len(args.shapes):
        raise ValueError("shapes must be unique")
    if args.station_b_is_real and args.station_b is None:
        raise ValueError("--station_b_is_real requires --station_b")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type != "cuda":
        warnings.warn(
            "CUDA is unavailable; a small smoke evaluation is fine here, "
            "but run the full bootstrap on the Sweden compute node.",
            RuntimeWarning,
        )

    stage3 = load_stage3_backbone(args.stage3_checkpoint, device="cpu").to(device)
    stage3.eval()
    stage4, checkpoint = load_stage4_checkpoint(args.stage4_checkpoint, device=device)
    preprocessing = checkpoint.get("preprocessing", {})
    integration = str(preprocessing.get("integration", "boxcar"))
    if args.integration is not None and args.integration != integration:
        if not args.allow_preprocessing_override:
            raise ValueError(
                "checkpoint was trained with integration=%r; refusing "
                "test-time override to %r without "
                "--allow_preprocessing_override" % (integration, args.integration)
            )
        integration = args.integration
        warnings.warn(
            "test-time preprocessing differs from training",
            RuntimeWarning,
        )
    if not bool(preprocessing.get("signed_foff_required", False)):
        raise ValueError("Stage-4 checkpoint does not assert signed-foff preprocessing")

    validation_operating = checkpoint.get("validation_operating_point", {})
    stage4_threshold = validation_operating.get("threshold")
    if stage4_threshold is not None:
        stage4_threshold = float(stage4_threshold)
    population = checkpoint.get("population", {})
    run_config = checkpoint.get("run_config", {})
    checkpoint_snr = population.get("target_snr", [8.0, 30.0])
    snr_min = args.snr_min if args.snr_min is not None else float(checkpoint_snr[0])
    snr_max = args.snr_max if args.snr_max is not None else float(checkpoint_snr[1])
    if snr_max < snr_min:
        raise ValueError("snr_max must be >= snr_min")

    negative_probabilities = parse_probability_map(
        args.negative_mix,
        ("onesided", "independent", "noise"),
    )
    cell_rows = []
    cell_data = {}
    manifest_path = out_dir / "test_manifest.jsonl"
    manifest_handle = manifest_path.open("w", encoding="utf-8")
    try:
        for regime_index, regime in enumerate(args.snr_modes):
            factory = CandidatePairFactory(
                mode=args.mode,
                station_a_filterbank=args.station_a,
                station_b_filterbank=args.station_b,
                station_b_is_proxy=not args.station_b_is_real,
                split="test",
                widths_hz=(
                    min(args.widths),
                    max(args.widths),
                ),
                target_snr_range=(snr_min, snr_max),
                snr_mode=regime,
                integration=integration,
                max_abs_drift_hz_s=float(run_config.get("max_abs_drift", 4.0)),
                width_log_error_sigma=float(
                    preprocessing.get("width_log_error_sigma", 0.15)
                ),
                drift_error_channels_per_tile=float(
                    preprocessing.get("drift_error_channels_per_tile", 0.35)
                ),
                center_error_channels=float(
                    preprocessing.get("center_error_channels", 0.35)
                ),
                station_modulation_log_sigma=float(
                    preprocessing.get("station_modulation_log_sigma", 0.06)
                ),
                remove_static_bandpass=bool(
                    preprocessing.get("remove_static_bandpass", True)
                ),
            )
            for shape_index, shape in enumerate(args.shapes):
                for width_index, width in enumerate(args.widths):
                    print(
                        "[%s] shape=%s width=%.3f Hz ..." % (regime, shape, width),
                        flush=True,
                    )
                    cell_seed = int(
                        args.seed
                        + regime_index * 1_000_000_007
                        + shape_index * 10_000_019
                        + width_index * 100_003
                    )
                    data = evaluate_cell(
                        factory=factory,
                        width_hz=width,
                        shape=shape,
                        n_per_class=args.n_per_class,
                        negative_probabilities=(negative_probabilities),
                        stage3=stage3,
                        stage4=stage4,
                        device=device,
                        use_amp=use_amp,
                        batch_size=args.batch_size,
                        seed=cell_seed,
                        manifest_handle=manifest_handle,
                    )
                    strata = np.asarray(
                        [
                            "%d|%s" % (int(label), case)
                            for label, case in zip(data["labels"], data["cases"])
                        ]
                    )
                    boot = paired_stratified_bootstrap(
                        data["labels"],
                        data["scores"],
                        strata,
                        n_boot=args.n_boot,
                        ci_level=args.ci_level,
                        seed=cell_seed + 53,
                    )
                    row = flatten_cell_summary(
                        regime,
                        shape,
                        width,
                        data,
                        boot,
                        stage3_margin=args.stage3_margin,
                        stage4_threshold=stage4_threshold,
                    )
                    cell_rows.append(row)
                    cell_data[(regime, shape, float(width))] = data
                    print(
                        "  raw AUC=%.4f; corrected-only=%.4f; "
                        "Stage4=%.4f; paired Δ=%.4f "
                        "[%.4f, %.4f]"
                        % (
                            row["stage3_raw_auc"],
                            row["stage3_corrected_auc"],
                            row["stage4_auc"],
                            row["delta_stage4_minus_raw"],
                            row["delta_stage4_minus_raw_ci_lo"],
                            row["delta_stage4_minus_raw_ci_hi"],
                        ),
                        flush=True,
                    )
    finally:
        manifest_handle.close()

    write_csv(cell_rows, out_dir / "stage4_width_results.csv")
    plot_cells(cell_rows, out_dir)
    primary = pooled_summary(
        cell_data,
        regime="detected",
        width_low=args.primary_width_min,
        width_high=args.primary_width_max,
        n_boot=args.n_boot,
        ci_level=args.ci_level,
        seed=args.seed + 700_001,
        target_auc=args.target_auc,
    )
    secondary = None
    if "power" in args.snr_modes:
        secondary = pooled_summary(
            cell_data,
            regime="power",
            width_low=args.primary_width_min,
            width_high=args.primary_width_max,
            n_boot=args.n_boot,
            ci_level=args.ci_level,
            seed=args.seed + 800_011,
            target_auc=args.target_auc,
        )
    result = {
        "format_version": 1,
        "stage3_checkpoint": str(Path(args.stage3_checkpoint).resolve()),
        "stage4_checkpoint": str(Path(args.stage4_checkpoint).resolve()),
        "device": str(device),
        "ci_level": args.ci_level,
        "n_boot": args.n_boot,
        "n_per_class_per_cell": args.n_per_class,
        "integration": integration,
        "negative_mix": negative_probabilities,
        "station_b_proxy": bool(not args.station_b_is_real),
        "primary_detected_conditioned": primary,
        "secondary_fixed_power": secondary,
        "cell_results": cell_rows,
    }
    with (out_dir / "stage4_evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    primary_delta = primary["deltas_vs_stage3_raw"]["stage4"]
    primary_stage4 = primary["methods"]["stage4"]
    status = (
        "PASS" if primary["success_criteria"]["primary_success"] else "NOT YET PASSED"
    )
    lines = [
        "Stage-4 paired evaluation",
        (
            "Primary endpoint: detected-conditioned, "
            "%.1f--%.1f Hz"
            % (
                args.primary_width_min,
                args.primary_width_max,
            )
        ),
        "Stage-4 pooled AUC: %.6f [%.6f, %.6f]"
        % (
            primary_stage4["auc"],
            primary_stage4["ci_lo"],
            primary_stage4["ci_hi"],
        ),
        ("Paired delta vs raw Stage 3: " "%.6f [%.6f, %.6f]")
        % (
            primary_delta["delta_auc"],
            primary_delta["ci_lo"],
            primary_delta["ci_hi"],
        ),
        ("Pre-registered criterion (AUC >= %.2f and " "paired lower CI > 0): %s")
        % (args.target_auc, status),
        "Station B background is a synthetic station proxy: %s"
        % bool(not args.station_b_is_real),
        "",
        (
            "Interpretation boundary: this is synthetic "
            "dual-station injection on real Sweden-accessible "
            "backgrounds. It is not real Ireland--Sweden "
            "candidate validation."
        ),
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "stage4_evaluation_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paired evaluation of corrected preprocessing " "and Stage-4 fine-tuning"
        ),
        formatter_class=(argparse.ArgumentDefaultsHelpFormatter),
    )
    parser.add_argument("--mode", default="high_freq", choices=("high_freq",))
    parser.add_argument("--station_a", required=True)
    parser.add_argument("--station_b", default=None)
    parser.add_argument(
        "--station_b_is_real",
        action="store_true",
        help="Set only for a genuinely different telescope station",
    )
    parser.add_argument("--stage3_checkpoint", required=True)
    parser.add_argument("--stage4_checkpoint", required=True)
    parser.add_argument(
        "--out_dir",
        default="eval_stage4_candidate_conditioned",
    )
    parser.add_argument(
        "--widths",
        type=parse_csv_floats,
        default=parse_csv_floats("3,10,20,30,50,75,100"),
    )
    parser.add_argument(
        "--shapes",
        type=lambda text: parse_csv_choices(text, ("lorentzian", "box", "gaussian")),
        default=["lorentzian", "box", "gaussian"],
    )
    parser.add_argument(
        "--snr_modes",
        type=lambda text: parse_csv_choices(text, ("detected", "power")),
        default=["detected", "power"],
    )
    parser.add_argument(
        "--negative_mix",
        default=("onesided:0.38,independent:0.38," "noise:0.24"),
    )
    parser.add_argument("--n_per_class", type=int, default=300)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--ci_level", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--snr_min", type=float, default=None)
    parser.add_argument("--snr_max", type=float, default=None)
    parser.add_argument("--stage3_margin", type=float, default=0.1833)
    parser.add_argument("--primary_width_min", type=float, default=10.0)
    parser.add_argument("--primary_width_max", type=float, default=100.0)
    parser.add_argument("--target_auc", type=float, default=0.80)
    parser.add_argument(
        "--integration",
        choices=("boxcar", "lorentzian", "gaussian"),
        default=None,
        help=(
            "Defaults to checkpoint value; overrides are blocked "
            "unless explicitly allowed"
        ),
    )
    parser.add_argument(
        "--allow_preprocessing_override",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
