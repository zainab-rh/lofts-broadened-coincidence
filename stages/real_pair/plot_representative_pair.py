#!/usr/bin/env python3
"""Create a publication-ready raw/candidate-informed paired-waterfall figure.

The script reads an already extracted pair archive and applies only the frozen
candidate preprocessing.  It does not run BLISS, extract HDF5 data, load a
neural-network checkpoint, or recompute inference scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lofts-matplotlib-cache")
)


STATION_LABELS = {
    "IRL": "Ireland station (IRL)",
    "SWE": "Sweden station (SWE)",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record %d is not a JSON object" % line_number)
            records.append(value)
    return records


def select_record(
    records: Sequence[Mapping[str, Any]], pair_id: Optional[str]
) -> Tuple[Mapping[str, Any], str]:
    if not records:
        raise ValueError("pair manifest is empty")
    if pair_id:
        selected = [item for item in records if str(item.get("pair_id")) == pair_id]
        if len(selected) != 1:
            raise ValueError(
                "--pair-id must identify exactly one manifest record; found %d"
                % len(selected)
            )
        return selected[0], "explicit_pair_id"
    return records[0], "first_manifest_record"


def import_preprocessing(stage4_code_dir: Path):
    directory = str(stage4_code_dir.expanduser().resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        from candidate_preprocessing import CandidateParams, make_candidate_view
    except ImportError as exc:
        raise ImportError(
            "--stage4-code-dir must contain candidate_preprocessing.py"
        ) from exc
    return CandidateParams, make_candidate_view


def candidate_params(station: Mapping[str, Any], CandidateParams):
    return CandidateParams(
        center_channel=float(station["candidate_center_channel"]),
        drift_hz_per_s=float(station["reported_drift_hz_s"]),
        width_hz=float(station["reported_width_fwhm_hz"]),
        channel_step_hz=float(station["signed_foff_hz"]),
        dt_s=float(station["tsamp_s"]),
        reference_row=float(station["candidate_reference_row"]),
    )


def load_pair(record: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, Path]:
    archive_path = Path(str(record["array_file"])).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    expected = str(record.get("array_sha256") or "")
    if expected and sha256_file(archive_path) != expected:
        raise ValueError("pair archive checksum mismatch: %s" % archive_path)
    with np.load(str(archive_path), allow_pickle=False) as archive:
        raw_a = np.asarray(archive["raw_a"], dtype=np.float32)
        raw_b = np.asarray(archive["raw_b"], dtype=np.float32)
        stored_union = str(np.asarray(archive["union_id"]).item())
    if stored_union != str(record["union_id"]):
        raise ValueError("pair archive union ID does not match the manifest")
    if raw_a.ndim != 2 or raw_a.shape != raw_b.shape:
        raise ValueError("paired waterfalls must be equal-sized two-dimensional arrays")
    if not np.isfinite(raw_a).all() or not np.isfinite(raw_b).all():
        raise ValueError("paired waterfall contains non-finite samples")
    return raw_a, raw_b, archive_path


def effective_integration(
    requested: Optional[str], inference_summary: Optional[Path]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    summary = None
    recorded = None
    if inference_summary is not None:
        with inference_summary.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, dict):
            raise ValueError("inference summary must be a JSON object")
        recorded = summary.get("integration")
    if requested and recorded and requested != recorded:
        raise ValueError(
            "requested integration %r conflicts with inference summary %r"
            % (requested, recorded)
        )
    return str(requested or recorded or "boxcar"), summary


def oriented_image(
    data: np.ndarray, station: Mapping[str, Any]
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    n_rows, n_cols = data.shape
    center = float(station["candidate_center_channel"])
    step_hz = float(station["signed_foff_hz"])
    reference_row = float(station["candidate_reference_row"])
    dt_s = float(station["tsamp_s"])
    frequency_khz = (np.arange(n_cols, dtype=float) - center) * step_hz / 1000.0
    time_s = (np.arange(n_rows, dtype=float) - reference_row) * dt_s
    oriented = data
    if frequency_khz[-1] < frequency_khz[0]:
        oriented = data[:, ::-1]
        frequency_khz = frequency_khz[::-1]
    half_step_khz = abs(step_hz) / 2000.0
    half_time_s = abs(dt_s) / 2.0
    extent = (
        float(frequency_khz[0] - half_step_khz),
        float(frequency_khz[-1] + half_step_khz),
        float(time_s[0] - half_time_s),
        float(time_s[-1] + half_time_s),
    )
    return oriented, extent


def robust_limits(values: np.ndarray, symmetric: bool = False) -> Tuple[float, float]:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if not finite_values.size:
        raise ValueError("cannot scale an empty or non-finite image")
    if symmetric:
        limit = float(np.quantile(np.abs(finite_values), 0.99))
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        return -limit, limit
    low, high = [float(value) for value in np.quantile(finite_values, [0.01, 0.99])]
    if not high > low:
        scale = max(abs(low), 1.0) * 1e-6
        low, high = low - scale, high + scale
    return low, high


def parse_formats(value: str) -> Tuple[str, ...]:
    formats = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unsupported = sorted(set(formats) - {"png", "pdf", "svg"})
    if unsupported:
        raise argparse.ArgumentTypeError("unsupported formats: %s" % unsupported)
    if not formats:
        raise argparse.ArgumentTypeError("at least one output format is required")
    return formats


def save_figure(fig, base_path: Path, formats: Sequence[str], dpi: int) -> List[Path]:
    outputs = []
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        destination = base_path.with_suffix("." + suffix)
        fig.savefig(
            str(destination),
            format=suffix,
            dpi=dpi if suffix == "png" else None,
            bbox_inches="tight",
            metadata={"Creator": "LOFTS Stage-4 presentation plotter"},
        )
        outputs.append(destination)
    return outputs


def main(args: argparse.Namespace) -> None:
    if args.dpi < 72:
        raise ValueError("dpi must be at least 72")
    manifest = args.pair_manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    summary_path = (
        None
        if args.inference_summary is None
        else args.inference_summary.expanduser().resolve()
    )
    if summary_path is not None and not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    records = read_json_records(manifest)
    record, selection_rule = select_record(records, args.pair_id)
    raw_a, raw_b, archive_path = load_pair(record)
    integration, inference_summary = effective_integration(
        args.integration, summary_path
    )
    CandidateParams, make_candidate_view = import_preprocessing(args.stage4_code_dir)
    station_a, station_b = [str(value) for value in record["station_order"]]
    stations = record["stations"]
    remove_static_bandpass = not args.keep_static_bandpass
    corrected_a = make_candidate_view(
        raw_a,
        candidate_params(stations[station_a], CandidateParams),
        integration=integration,
        remove_static_bandpass=remove_static_bandpass,
    )
    corrected_b = make_candidate_view(
        raw_b,
        candidate_params(stations[station_b], CandidateParams),
        integration=integration,
        remove_static_bandpass=remove_static_bandpass,
    )
    corrected_a = np.asarray(corrected_a, dtype=np.float32)
    corrected_b = np.asarray(corrected_b, dtype=np.float32)
    if corrected_a.shape != raw_a.shape or corrected_b.shape != raw_b.shape:
        raise ValueError("candidate preprocessing changed the waterfall shape")
    if not np.isfinite(corrected_a).all() or not np.isfinite(corrected_b).all():
        raise ValueError("candidate preprocessing produced non-finite samples")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    raw_limits = robust_limits(np.concatenate((raw_a.ravel(), raw_b.ravel())))
    corrected_limits = robust_limits(
        np.concatenate((corrected_a.ravel(), corrected_b.ravel())), symmetric=True
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4), sharey=True)
    station_ids = (station_a, station_b)
    raw_arrays = (raw_a, raw_b)
    corrected_arrays = (corrected_a, corrected_b)
    raw_image = None
    corrected_image = None
    for column, station_id in enumerate(station_ids):
        station = stations[station_id]
        raw, extent = oriented_image(raw_arrays[column], station)
        corrected, corrected_extent = oriented_image(corrected_arrays[column], station)
        raw_image = axes[0, column].imshow(
            raw,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=raw_limits[0],
            vmax=raw_limits[1],
            cmap="viridis",
            interpolation="nearest",
            rasterized=True,
        )
        corrected_image = axes[1, column].imshow(
            corrected,
            origin="lower",
            aspect="auto",
            extent=corrected_extent,
            vmin=corrected_limits[0],
            vmax=corrected_limits[1],
            cmap="RdBu_r",
            interpolation="nearest",
            rasterized=True,
        )
        axes[0, column].set_title(STATION_LABELS.get(station_id, station_id))
        axes[1, column].set_xlabel("Frequency offset from candidate reference (kHz)")
        for row in range(2):
            axes[row, column].axvline(0.0, color="white", alpha=0.65, linewidth=0.8)
    axes[0, 0].set_ylabel("Raw waterfall\nTime relative to candidate reference (s)")
    axes[1, 0].set_ylabel(
        "Candidate-informed representation\nTime relative to candidate reference (s)"
    )
    raw_bar = fig.colorbar(raw_image, ax=list(axes[0, :]), fraction=0.025, pad=0.025)
    raw_bar.set_label("Relative intensity (native units)")
    corrected_bar = fig.colorbar(
        corrected_image, ax=list(axes[1, :]), fraction=0.025, pad=0.025
    )
    corrected_bar.set_label("Robust normalised intensity")
    fig.suptitle(
        "Representative paired-waterfall preprocessing example",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.925,
        "LOFTS0050 unlabeled real-pair pilot: signed de-chirping and width-dependent frequency integration",
        ha="center",
        fontsize=10.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.025,
        "Illustrative preprocessing example selected independently of model score; it is not classification evidence or an astrophysical identification.",
        ha="center",
        fontsize=9,
        color="#4A4A4A",
    )
    fig.subplots_adjust(
        top=0.86, bottom=0.14, left=0.10, right=0.88, hspace=0.24, wspace=0.16
    )
    outputs = save_figure(
        fig,
        args.out_dir.expanduser().resolve() / "14_representative_pair_preprocessing",
        args.formats,
        args.dpi,
    )
    plt.close(fig)

    metadata_path = (
        args.out_dir.expanduser().resolve()
        / "14_representative_pair_preprocessing.metadata.json"
    )
    metadata = {
        "format_version": 1,
        "figure_role": "representative_raw_and_candidate_informed_pair",
        "dataset_role": "unlabeled_real_barycentric_pair",
        "selection_rule": selection_rule,
        "pair_id": str(record["pair_id"]),
        "union_id": str(record["union_id"]),
        "detection_state": record.get("detection_state"),
        "station_order": list(station_ids),
        "integration": integration,
        "remove_static_bandpass": remove_static_bandpass,
        "pair_manifest": str(manifest),
        "pair_manifest_sha256": sha256_file(manifest),
        "pair_archive": str(archive_path),
        "pair_archive_sha256": sha256_file(archive_path),
        "inference_summary": None if summary_path is None else str(summary_path),
        "inference_summary_sha256": (
            None if summary_path is None else sha256_file(summary_path)
        ),
        "inference_was_rerun": False,
        "scientific_boundary": (
            "This figure illustrates frozen candidate preprocessing on one "
            "unlabeled real pair. It is not evidence of classification "
            "correctness or astrophysical origin."
        ),
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Created representative preprocessing figure:")
    for path in outputs:
        print("  %s" % path)
    print("  %s" % metadata_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one existing raw/candidate-informed LOFTS pair",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--stage4-code-dir", type=Path, required=True)
    parser.add_argument("--inference-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--integration", default=None)
    parser.add_argument(
        "--keep-static-bandpass",
        action="store_true",
        help="Disable the frozen default static-bandpass removal",
    )
    parser.add_argument("--formats", type=parse_formats, default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
