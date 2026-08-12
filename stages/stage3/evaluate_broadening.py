"""Controlled synthetic evaluation of broadened-signal coincidence.

The evaluator measures AUC-ROC, bootstrap confidence intervals, latent
separation, and threshold-dependent metrics across fixed broadening widths,
intrinsic S/N values, and profile shapes.  It supports side-by-side comparison
of narrowband-only and broadened-aware checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lorentzian_signals as ls
import matplotlib.pyplot as plt
import numpy as np
import setigen as stg
import torch
import torch.nn.functional as F
from json_utils import json_safe
from train import (
    CONFIGS,
    FILTERBANK_PATHS,
    UNet,
    get_real_slice,
    load_background,
    preprocess_np,
)

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None
DEFAULT_WIDTH_GRID_HZ = [0.5, 1, 3, 10, 30, 100, 300, 1000, 3000]
DEFAULT_SHAPE_WIDTHS_HZ = [30.0, 100.0]
DEFAULT_SNR_RANGE = (5.0, 30.0)
# =============================================================================
# 1. MODEL AND SCORING UTILITIES
# =============================================================================


def load_checkpoint(path, device):
    """Load a UNet checkpoint. Returns None if the path does not exist,
    with a printed warning (rather than raising), so a two-checkpoint
    comparison degrades gracefully to a single-checkpoint run if one is
    missing."""
    path = Path(path)
    if not path.exists():
        print(f"WARNING: checkpoint not found at {path}; skipping this model.")
        return None
    model = UNet().to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def score_pairs_batch(
    model, device, arrs1: list, arrs2: list, batch_size: int = 32
) -> np.ndarray:
    """Batched forward pass over a list of raw (tchans, fchans) array pairs.
    Returns a 1D array of latent-space Euclidean distances, one per pair."""
    distances = []
    n = len(arrs1)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b1 = arrs1[i : i + batch_size]
            b2 = arrs2[i : i + batch_size]
            t1 = torch.from_numpy(
                np.stack([preprocess_np(a)[None, ...] for a in b1])
            ).to(device)
            t2 = torch.from_numpy(
                np.stack([preprocess_np(a)[None, ...] for a in b2])
            ).to(device)
            _, z1 = model(t1)
            _, z2 = model(t2)
            d = F.pairwise_distance(z1, z2).cpu().numpy()
            distances.extend(d.tolist())
    return np.array(distances)


def bootstrap_auc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    ci: float = 0.90,
    seed: int = 0,
) -> tuple:
    """Percentile bootstrap confidence interval for AUC-ROC, stratified by
    class (resampling positives and negatives independently) so the class
    balance of each bootstrap replicate matches the observed data."""
    if roc_auc_score is None:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return (float("nan"), float("nan"))
    boot_aucs = []
    for _ in range(n_boot):
        bp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        bn = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        idx = np.concatenate([bp, bn])
        try:
            boot_aucs.append(roc_auc_score(labels[idx], scores[idx]))
        except ValueError:
            continue
    if not boot_aucs:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(boot_aucs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_aucs, (1 + ci) / 2 * 100))
    return (lo, hi)


def compute_threshold_metrics(
    distances: np.ndarray, labels: np.ndarray, margin: float
) -> dict:
    """Compute precision, recall, and F1 at a fixed distance threshold."""
    pred_match = (distances < margin).astype(int)
    labels = labels.astype(int)
    tp = int(np.sum((labels == 1) & (pred_match == 1)))
    fn = int(np.sum((labels == 1) & (pred_match == 0)))
    fp = int(np.sum((labels == 0) & (pred_match == 1)))
    tn = int(np.sum((labels == 0) & (pred_match == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# =============================================================================
# 2. CONTROLLED FIXED-WIDTH PAIR GENERATION
# =============================================================================


def make_frame_pair_background(
    mode: str, real_data: np.ndarray, header: dict, real_data_b: np.ndarray = None
):
    """Draws two real-noise slices and wraps them as setigen Frames using
    station A's channel grid (both real LOFAR stations share the same
    nominal HBA channelisation, Johnson et al. 2023 Table 1)."""
    tile_shape = CONFIGS[mode]["frame_shape"]
    df, dt = abs(header["foff"]) * 1e6, header["tsamp"]
    fch1, ascending = header["fch1"], header["foff"] > 0
    data_b_source = real_data_b if real_data_b is not None else real_data
    slice1 = get_real_slice(real_data, tile_shape)
    slice2 = get_real_slice(data_b_source, tile_shape)
    if slice1 is None or slice2 is None:
        return None, None
    frame1 = stg.Frame.from_data(
        df=df, dt=dt, fch1=fch1, ascending=ascending, data=slice1
    )
    frame2 = stg.Frame.from_data(
        df=df, dt=dt, fch1=fch1, ascending=ascending, data=slice2
    )
    return frame1, frame2


_INJECT_FN = {
    "lorentzian": ls.inject_lorentzian_signal,
    "box": ls.inject_box_decoy_signal,
    "gaussian": ls.inject_gaussian_decoy_signal,
}


def generate_eval_pair(
    mode: str,
    real_data: np.ndarray,
    header: dict,
    real_data_b,
    width_hz: float,
    shape: str,
    case: str,
    snr_range: tuple,
    rng: np.random.Generator,
):
    """Generate one labelled pair at a prescribed width and profile shape.

    The controlled sweep intentionally bypasses the population-conditioned
    training sampler. ``case`` may be ``match``, ``mismatch_onesided``,
    ``mismatch_independent``, or ``mismatch_nosignal``.
    """
    frame1, frame2 = make_frame_pair_background(mode, real_data, header, real_data_b)
    if frame1 is None:
        return None, None, None
    inject_fn = _INJECT_FN[shape]

    def _level(frame):
        nm, ns = frame.get_noise_stats()
        if ns == 0:
            ns = 1.0
        snr = float(rng.uniform(*snr_range))
        return nm + ns * snr

    if case == "mismatch_nosignal":
        return frame1.get_data(), frame2.get_data(), 0
    draw = ls.sample_one_broadening(rng=rng)
    drift_hz_s = draw.drift_hz_per_s
    if case == "match":
        r1 = inject_fn(frame1, None, drift_hz_s, _level(frame1), width_hz)
        inject_fn(frame2, r1["f_start_hz"], drift_hz_s, _level(frame2), width_hz)
        label = 1
    elif case == "mismatch_onesided":
        inject_fn(frame1, None, drift_hz_s, _level(frame1), width_hz)
        label = 0
    elif case == "mismatch_independent":
        draw2 = ls.sample_one_broadening(rng=rng)
        inject_fn(frame1, None, drift_hz_s, _level(frame1), width_hz)
        inject_fn(frame2, None, draw2.drift_hz_per_s, _level(frame2), width_hz)
        label = 0
    else:
        raise ValueError(f"Unknown case: {case!r}")
    return frame1.get_data(), frame2.get_data(), label


# =============================================================================
# 3. THE WIDTH SWEEP
# =============================================================================


def run_width_sweep(
    model,
    device,
    mode: str,
    real_data: np.ndarray,
    header: dict,
    real_data_b,
    width_grid_hz: list,
    n_match: int,
    n_mismatch: int,
    snr_range: tuple,
    shape: str = "lorentzian",
    seed: int = 0,
    batch_size: int = 32,
    margin: float = None,
) -> list:
    """Evaluate match discrimination across a fixed broadening-width grid.

    Each row includes AUC-ROC with a stratified bootstrap interval, latent
    separation, and optional threshold-dependent classification metrics.
    """
    rng = np.random.default_rng(seed)
    results = []
    for width_hz in width_grid_hz:
        arrs1, arrs2, labels = [], [], []
        for _ in range(n_match):
            a1, a2, lbl = generate_eval_pair(
                mode,
                real_data,
                header,
                real_data_b,
                width_hz,
                shape,
                "match",
                snr_range,
                rng,
            )
            if a1 is None:
                continue
            arrs1.append(a1)
            arrs2.append(a2)
            labels.append(lbl)
        n_onesided = n_mismatch // 2
        n_independent = n_mismatch - n_onesided
        for _ in range(n_onesided):
            a1, a2, lbl = generate_eval_pair(
                mode,
                real_data,
                header,
                real_data_b,
                width_hz,
                shape,
                "mismatch_onesided",
                snr_range,
                rng,
            )
            if a1 is None:
                continue
            arrs1.append(a1)
            arrs2.append(a2)
            labels.append(lbl)
        for _ in range(n_independent):
            a1, a2, lbl = generate_eval_pair(
                mode,
                real_data,
                header,
                real_data_b,
                width_hz,
                shape,
                "mismatch_independent",
                snr_range,
                rng,
            )
            if a1 is None:
                continue
            arrs1.append(a1)
            arrs2.append(a2)
            labels.append(lbl)
        if not arrs1:
            print(
                f"  width={width_hz:>8} Hz | WARNING: no valid pairs generated, skipping."
            )
            continue
        distances = score_pairs_batch(model, device, arrs1, arrs2, batch_size)
        labels_arr = np.array(labels)
        match_d = distances[labels_arr == 1]
        mismatch_d = distances[labels_arr == 0]
        row = {
            "width_hz": float(width_hz),
            "shape": shape,
            "n_match": int(len(match_d)),
            "n_mismatch": int(len(mismatch_d)),
            "match_mean": float(match_d.mean()) if len(match_d) else float("nan"),
            "match_std": float(match_d.std()) if len(match_d) else float("nan"),
            "mismatch_mean": (
                float(mismatch_d.mean()) if len(mismatch_d) else float("nan")
            ),
            "mismatch_std": (
                float(mismatch_d.std()) if len(mismatch_d) else float("nan")
            ),
            "peak_retention_pct": float(ls.peak_retention_factor(width_hz) * 100.0),
        }
        if len(match_d) and len(mismatch_d):
            row["separation_gap"] = row["mismatch_mean"] - row["match_mean"]
            if roc_auc_score is not None:
                scores = -distances
                try:
                    row["auc_roc"] = float(roc_auc_score(labels_arr, scores))
                except ValueError:
                    row["auc_roc"] = float("nan")
                ci_lo, ci_hi = bootstrap_auc_ci(labels_arr, scores, seed=seed)
                row["auc_roc_ci_lo"] = ci_lo
                row["auc_roc_ci_hi"] = ci_hi
            else:
                row["auc_roc"] = None
                row["auc_roc_ci_lo"] = row["auc_roc_ci_hi"] = None
        if margin is not None:
            thresh_metrics = compute_threshold_metrics(distances, labels_arr, margin)
            row.update({f"thresh_{k}": v for k, v in thresh_metrics.items()})
        print(
            f"  width={width_hz:>8.1f} Hz | n={len(labels_arr):>4} | "
            f"retention={row['peak_retention_pct']:6.2f}% | "
            f"gap={row.get('separation_gap', float('nan')):.4f} | "
            f"AUC={row.get('auc_roc', float('nan')):.4f}"
            + (
                f" [{row['auc_roc_ci_lo']:.3f}, {row['auc_roc_ci_hi']:.3f}]"
                if row.get("auc_roc_ci_lo") is not None
                else ""
            )
            + (
                f" | recall={row.get('thresh_recall', float('nan')):.3f} "
                f"precision={row.get('thresh_precision', float('nan')):.3f} "
                f"F1={row.get('thresh_f1', float('nan')):.3f}"
                if margin is not None
                else ""
            )
        )
        results.append(row)
    return results


def run_width_snr_grid(
    model,
    device,
    mode: str,
    real_data: np.ndarray,
    header: dict,
    real_data_b,
    width_grid_hz: list,
    snr_grid: list,
    n_match: int,
    n_mismatch: int,
    margin: float,
    shape: str = "lorentzian",
    seed: int = 0,
    batch_size: int = 32,
) -> list:
    """Evaluate threshold and rank metrics on a fixed width-by-S/N grid."""
    rng = np.random.default_rng(seed)
    results = []
    for width_hz in width_grid_hz:
        for snr in snr_grid:
            fixed_snr_range = (snr, snr)
            arrs1, arrs2, labels = [], [], []
            for _ in range(n_match):
                a1, a2, lbl = generate_eval_pair(
                    mode,
                    real_data,
                    header,
                    real_data_b,
                    width_hz,
                    shape,
                    "match",
                    fixed_snr_range,
                    rng,
                )
                if a1 is None:
                    continue
                arrs1.append(a1)
                arrs2.append(a2)
                labels.append(lbl)
            n_onesided = n_mismatch // 2
            for _ in range(n_onesided):
                a1, a2, lbl = generate_eval_pair(
                    mode,
                    real_data,
                    header,
                    real_data_b,
                    width_hz,
                    shape,
                    "mismatch_onesided",
                    fixed_snr_range,
                    rng,
                )
                if a1 is None:
                    continue
                arrs1.append(a1)
                arrs2.append(a2)
                labels.append(lbl)
            for _ in range(n_mismatch - n_onesided):
                a1, a2, lbl = generate_eval_pair(
                    mode,
                    real_data,
                    header,
                    real_data_b,
                    width_hz,
                    shape,
                    "mismatch_independent",
                    fixed_snr_range,
                    rng,
                )
                if a1 is None:
                    continue
                arrs1.append(a1)
                arrs2.append(a2)
                labels.append(lbl)
            if not arrs1:
                continue
            distances = score_pairs_batch(model, device, arrs1, arrs2, batch_size)
            labels_arr = np.array(labels)
            thresh = compute_threshold_metrics(distances, labels_arr, margin)
            row = {"width_hz": float(width_hz), "snr": float(snr), **thresh}
            match_d = distances[labels_arr == 1]
            mismatch_d = distances[labels_arr == 0]
            if len(match_d) and len(mismatch_d) and roc_auc_score is not None:
                try:
                    row["auc_roc"] = float(roc_auc_score(labels_arr, -distances))
                except ValueError:
                    row["auc_roc"] = float("nan")
            else:
                row["auc_roc"] = float("nan")
            print(
                f"  width={width_hz:>8.1f} Hz | snr={snr:>5.1f} | "
                f"recall={row['recall']:.3f} precision={row['precision']:.3f} "
                f"F1={row['f1']:.3f} AUC={row['auc_roc']:.3f}"
            )
            results.append(row)
    return results


def plot_width_snr_heatmap(
    results: list, metric: str, width_grid_hz: list, snr_grid: list, save_path: Path
):
    """Renders `metric` as a (width x SNR) heatmap."""
    grid = np.full((len(snr_grid), len(width_grid_hz)), np.nan)
    for r in results:
        i = snr_grid.index(r["snr"])
        j = width_grid_hz.index(r["width_hz"])
        grid[i, j] = r.get(metric, np.nan)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(width_grid_hz)))
    ax.set_xticklabels([f"{w:g}" for w in width_grid_hz], rotation=45)
    ax.set_yticks(range(len(snr_grid)))
    ax.set_yticklabels([f"{s:g}" for s in snr_grid])
    ax.set_xlabel("Spectral broadening Δνsb (Hz)")
    ax.set_ylabel("Intrinsic (pre-broadening) SNR")
    ax.set_title(f"Width-S/N evaluation: {metric} as a function of width and SNR")
    fig.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# =============================================================================
# 4. SHAPE-AGNOSTIC GENERALIZATION CHECK
# =============================================================================


def run_shape_comparison(
    model,
    device,
    mode: str,
    real_data: np.ndarray,
    header: dict,
    real_data_b,
    shape_widths_hz: list,
    n_match: int,
    n_mismatch: int,
    snr_range: tuple,
    seed: int = 100,
    batch_size: int = 32,
) -> dict:
    """Compare Lorentzian, box, and Gaussian profiles at matched widths.

    All shapes use the same pair-generation and bootstrap procedure, allowing
    profile generalisation to be assessed without changing the evaluator.
    """
    out = {}
    for width_hz in shape_widths_hz:
        out[width_hz] = {}
        for shape in ("lorentzian", "box", "gaussian"):
            print(f"\n  -- shape comparison: width={width_hz} Hz, shape={shape} --")
            rows = run_width_sweep(
                model,
                device,
                mode,
                real_data,
                header,
                real_data_b,
                [width_hz],
                n_match,
                n_mismatch,
                snr_range,
                shape=shape,
                seed=seed,
                batch_size=batch_size,
            )
            out[width_hz][shape] = rows[0] if rows else None
    return out


# =============================================================================
# 5. PLOTTING
# =============================================================================


def plot_physics_crosscheck(results: list, label: str, save_path: Path):
    """Overlay theoretical peak retention and empirical separation."""
    widths = [r["width_hz"] for r in results]
    gaps = [r.get("separation_gap", np.nan) for r in results]
    retention_pct = [r.get("peak_retention_pct", np.nan) for r in results]
    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        widths,
        retention_pct,
        marker="s",
        color="gold",
        lw=2,
        label="Theoretical peak retention (%) — lorentzian_signals.py",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Spectral broadening Δνsb (Hz)")
    ax1.set_ylabel("Theoretical peak retention (%)", color="gold")
    ax1.tick_params(axis="y", labelcolor="gold")
    ax2 = ax1.twinx()
    ax2.plot(
        widths,
        gaps,
        marker="o",
        color="cyan",
        lw=2,
        label=f"Empirical separation gap — {label}",
    )
    ax2.set_ylabel("Empirical separation gap (latent distance units)", color="cyan")
    ax2.tick_params(axis="y", labelcolor="cyan")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title(
        "Physics Cross-Check: Theoretical Retention vs.\n"
        "Empirical Model Degradation (should track together)"
    )
    ax1.grid(alpha=0.25, which="both")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_width_sweep(
    results_a: list,
    label_a: str,
    results_b: list = None,
    label_b: str = None,
    save_path: Path = None,
):
    """Two-panel figure: AUC-ROC vs. width (with bootstrap CI shading) and
    separation gap vs. width."""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    def _plot_one(ax_auc, ax_gap, results, label, color):
        widths = [r["width_hz"] for r in results]
        aucs = [r.get("auc_roc", np.nan) for r in results]
        gaps = [r.get("separation_gap", np.nan) for r in results]
        ci_lo = [r.get("auc_roc_ci_lo", np.nan) for r in results]
        ci_hi = [r.get("auc_roc_ci_hi", np.nan) for r in results]
        ax_auc.plot(widths, aucs, marker="o", color=color, label=label, lw=2)
        if all(v is not None and not np.isnan(v) for v in ci_lo + ci_hi):
            ax_auc.fill_between(widths, ci_lo, ci_hi, color=color, alpha=0.15)
        ax_gap.plot(widths, gaps, marker="o", color=color, label=label, lw=2)

    _plot_one(ax1, ax2, results_a, label_a, "cyan")
    if results_b:
        _plot_one(ax1, ax2, results_b, label_b, "orange")
    ax1.axhline(0.5, color="gray", ls="--", lw=1, label="Chance (AUC=0.5)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Spectral broadening Δνsb (Hz)")
    ax1.set_ylabel("AUC-ROC (dual-station match detection)")
    ax1.set_title(
        "Detection performance vs. broadening width\n(shaded = 90% bootstrap CI)"
    )
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, which="both")
    ax1.set_ylim(0.4, 1.02)
    ax2.axhline(0.0, color="gray", ls="--", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("Spectral broadening Δνsb (Hz)")
    ax2.set_ylabel("Separation gap (mismatch_mean - match_mean)")
    ax2.set_title("Latent-space separation vs. broadening width")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_shape_comparison(shape_results: dict, model_label: str, save_path: Path):
    """Plot profile-specific AUC values with 90% bootstrap intervals."""
    plt.style.use("dark_background")
    widths = sorted(shape_results.keys())
    shapes = ["lorentzian", "box", "gaussian"]
    colors = {"lorentzian": "cyan", "box": "salmon", "gaussian": "lightgreen"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_widths = len(widths)
    bar_width = 0.8 / len(shapes)
    x = np.arange(n_widths)
    for i, shape in enumerate(shapes):
        vals, err_lo, err_hi = [], [], []
        for w in widths:
            row = shape_results[w][shape]
            auc = row["auc_roc"] if row else np.nan
            vals.append(auc)
            if (
                row
                and row.get("auc_roc_ci_lo") is not None
                and not np.isnan(row.get("auc_roc_ci_lo", np.nan))
            ):
                err_lo.append(max(auc - row["auc_roc_ci_lo"], 0))
                err_hi.append(max(row["auc_roc_ci_hi"] - auc, 0))
            else:
                err_lo.append(0)
                err_hi.append(0)
        offset = (i - (len(shapes) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            vals,
            width=bar_width,
            label=shape,
            color=colors[shape],
            alpha=0.85,
            yerr=[err_lo, err_hi],
            capsize=3,
        )
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Chance (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w:.0f} Hz" for w in widths])
    ax.set_ylabel("AUC-ROC (± 90% bootstrap CI)")
    ax.set_title(
        f"Shape-agnostic generalization check — model: {model_label}\n"
        "(Lorentzian vs. box- and Gaussian-shaped decoys, matched width/drift/energy)"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0.3, 1.05)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =============================================================================
# 6. MAIN
# =============================================================================


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        "\n[1] Physics context — peak retention at each width in the sweep grid "
        "(Gajjar & Brown 2026, verified formula):"
    )
    width_grid = [float(w) for w in args.width_grid.split(",")]
    for w in width_grid:
        print(
            f"    Delta_nu_sb={w:>8.1f} Hz -> peak retention = "
            f"{ls.peak_retention_factor(w) * 100:6.2f}%"
        )
    print("\n[2] Loading model checkpoint(s)...")
    model_a = load_checkpoint(args.checkpoint_a, device)
    if model_a is None:
        print("FATAL ERROR: --checkpoint_a could not be loaded. Aborting.")
        return
    model_b = load_checkpoint(args.checkpoint_b, device) if args.checkpoint_b else None
    print("\n[3] Loading background data...")
    fb_path = args.filterbank or FILTERBANK_PATHS[args.mode]
    real_data, header = load_background(fb_path)
    real_data_b = None
    if args.station_b_filterbank:
        real_data_b, _ = load_background(args.station_b_filterbank)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snr_range = (args.broadened_snr_min, args.broadened_snr_max)
    print(
        f"\n[4] Running width sweep for '{args.checkpoint_a_label}' "
        f"({len(width_grid)} widths, {args.n_per_width} pairs/label each)..."
    )
    results_a = run_width_sweep(
        model_a,
        device,
        args.mode,
        real_data,
        header,
        real_data_b,
        width_grid,
        args.n_per_width,
        args.n_per_width,
        snr_range,
        shape="lorentzian",
        seed=args.seed,
        margin=args.margin,
    )
    results_b = None
    if model_b is not None:
        print(f"\n[5] Running width sweep for '{args.checkpoint_b_label}'...")
        results_b = run_width_sweep(
            model_b,
            device,
            args.mode,
            real_data,
            header,
            real_data_b,
            width_grid,
            args.n_per_width,
            args.n_per_width,
            snr_range,
            shape="lorentzian",
            seed=args.seed + 1,
            margin=args.margin,
        )
    plot_width_sweep(
        results_a,
        args.checkpoint_a_label,
        results_b,
        args.checkpoint_b_label,
        save_path=out_dir / "auc_vs_width.png",
    )
    plot_physics_crosscheck(
        results_a, args.checkpoint_a_label, out_dir / "physics_crosscheck.png"
    )
    with open(out_dir / "auc_vs_width_a.json", "w") as f:
        json.dump(json_safe(results_a), f, indent=2, allow_nan=False)
    if results_b:
        with open(out_dir / "auc_vs_width_b.json", "w") as f:
            json.dump(json_safe(results_b), f, indent=2, allow_nan=False)
    grid_results = None
    if args.run_snr_grid:
        print(f"\n[5b] Running width x SNR grid for '{args.checkpoint_a_label}'...")
        snr_grid = [float(s) for s in args.snr_grid.split(",")]
        grid_width_grid = [float(w) for w in args.snr_grid_widths.split(",")]
        grid_results = run_width_snr_grid(
            model_a,
            device,
            args.mode,
            real_data,
            header,
            real_data_b,
            grid_width_grid,
            snr_grid,
            args.n_per_grid_point,
            args.n_per_grid_point,
            margin=args.margin,
            seed=args.seed,
        )
        for metric in ("recall", "precision", "f1", "auc_roc"):
            plot_width_snr_heatmap(
                grid_results,
                metric,
                grid_width_grid,
                snr_grid,
                out_dir / f"width_snr_grid_{metric}.png",
            )
        with open(out_dir / "width_snr_grid.json", "w") as f:
            json.dump(json_safe(grid_results), f, indent=2, allow_nan=False)
    # Prefer the broadened-aware checkpoint for the profile-generalisation
    # test when a second model is available.
    choice = args.shape_check_checkpoint
    if choice is None:
        choice = "b" if model_b is not None else "a"
    if choice == "b" and model_b is None:
        print(
            "WARNING: --shape_check_checkpoint=b requested but no --checkpoint_b given; "
            "falling back to 'a'."
        )
        choice = "a"
    shape_targets = []
    if choice in ("a", "both"):
        shape_targets.append((model_a, args.checkpoint_a_label, "a"))
    if choice in ("b", "both") and model_b is not None:
        shape_targets.append((model_b, args.checkpoint_b_label, "b"))
    shape_widths = [float(w) for w in args.shape_widths.split(",")]
    all_shape_results = {}
    for shape_model, shape_label, tag in shape_targets:
        print(
            f"\n[6] Running shape-agnostic generalization check "
            f"(widths={args.shape_widths}, model='{shape_label}')..."
        )
        shape_results = run_shape_comparison(
            shape_model,
            device,
            args.mode,
            real_data,
            header,
            real_data_b,
            shape_widths,
            args.n_shape_trials,
            args.n_shape_trials,
            snr_range,
            seed=args.seed + 100,
        )
        all_shape_results[tag] = (shape_label, shape_results)
        plot_shape_comparison(
            shape_results,
            shape_label,
            save_path=out_dir / f"shape_comparison_{tag}.png",
        )
        with open(out_dir / f"shape_comparison_{tag}.json", "w") as f:
            json.dump(
                json_safe({str(k): v for k, v in shape_results.items()}),
                f,
                indent=2,
                allow_nan=False,
            )
    # --- Written report -------------------------------------------------
    report_path = out_dir / "evaluate_broadening_report.txt"
    with open(report_path, "w") as f:
        f.write("=== Broadened-Signal Dual-Station Coincidence Evaluation ===\n\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Background: {fb_path}\n")
        f.write(
            f"Station-B background: {args.station_b_filterbank or '(same file, single-source fallback)'}\n"
        )
        f.write(
            f"Margin used for confidence conversion (display only, AUC is margin-free): {args.margin}\n\n"
        )
        f.write("--- Width sweep: " + args.checkpoint_a_label + " ---\n")
        for r in results_a:
            f.write(
                f"  Delta_nu_sb={r['width_hz']:>8.1f} Hz | retention={r['peak_retention_pct']:6.2f}% | "
                f"gap={r.get('separation_gap', float('nan')):.4f} | "
                f"AUC={r.get('auc_roc', float('nan')):.4f} "
                f"[{r.get('auc_roc_ci_lo', float('nan')):.3f}, {r.get('auc_roc_ci_hi', float('nan')):.3f}]\n"
            )
        if results_b:
            f.write(f"\n--- Width sweep: {args.checkpoint_b_label} ---\n")
            for r in results_b:
                f.write(
                    f"  Delta_nu_sb={r['width_hz']:>8.1f} Hz | retention={r['peak_retention_pct']:6.2f}% | "
                    f"gap={r.get('separation_gap', float('nan')):.4f} | "
                    f"AUC={r.get('auc_roc', float('nan')):.4f} "
                    f"[{r.get('auc_roc_ci_lo', float('nan')):.3f}, {r.get('auc_roc_ci_hi', float('nan')):.3f}]\n"
                )
        for tag, (shape_label, shape_results) in all_shape_results.items():
            f.write(
                f"\n--- Shape-agnostic generalization check (model: {shape_label}) ---\n"
            )
            for w in shape_widths:
                f.write(f"  Width = {w:.0f} Hz:\n")
                for shape in ("lorentzian", "box", "gaussian"):
                    row = shape_results[w][shape]
                    if row is None:
                        f.write(f"    {shape:<12}: (no valid pairs generated)\n")
                        continue
                    f.write(
                        f"    {shape:<12}: AUC={row.get('auc_roc', float('nan')):.4f} "
                        f"[{row.get('auc_roc_ci_lo', float('nan')):.3f}, "
                        f"{row.get('auc_roc_ci_hi', float('nan')):.3f}] | "
                        f"gap={row.get('separation_gap', float('nan')):.4f}\n"
                    )
        if grid_results:
            f.write("\n--- Width x S/N grid ---\n")
            for r in grid_results:
                f.write(
                    f"  width={r['width_hz']:>8.1f} Hz | snr={r['snr']:>5.1f} | "
                    f"recall={r['recall']:.3f} precision={r['precision']:.3f} "
                    f"F1={r['f1']:.3f} AUC={r.get('auc_roc', float('nan')):.3f}\n"
                )
        f.write("\n--- Interpretation notes ---\n")
        f.write(
            "1. AUC near the original pipeline's narrowband-only performance across the\n"
            "   full width grid indicates the existing comparison mechanism already\n"
            "   generalises to broadened signals without any architecture change.\n"
        )
        f.write(
            "2. A widening bootstrap CI at large widths reflects genuinely lower\n"
            "   achievable statistical confidence at low observed SNR, not necessarily a\n"
            "   methodological flaw.\n"
        )
        f.write(
            "3. The shape-agnostic check uses the broadened-aware checkpoint when\n"
            "   available. A shape difference is meaningful only when it is larger\n"
            "   than the associated bootstrap uncertainty.\n"
        )
        f.write(
            "4. These results are for SYNTHETIC injections into real background NOISE\n"
            "   only (not real BLISS candidates), and are not barycentric-corrected (not\n"
            "   required here since the same drift is injected into both stations by\n"
            "   construction).\n"
        )
    print(f"\nAll outputs written to {out_dir}/")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Width-stratified evaluation of dual-station coincidence "
        "detection for spectrally broadened signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, required=True, choices=CONFIGS.keys())
    parser.add_argument("--checkpoint_a", type=str, required=True)
    parser.add_argument("--checkpoint_a_label", type=str, default="Model A")
    parser.add_argument(
        "--checkpoint_b",
        type=str,
        default=None,
        help="Optional SECOND checkpoint (e.g. Stage-3 broadened-aware) "
        "for a side-by-side comparison.",
    )
    parser.add_argument("--checkpoint_b_label", type=str, default="Model B")
    parser.add_argument("--filterbank", type=str, default=None)
    parser.add_argument("--station_b_filterbank", type=str, default=None)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.8,
        help="Threshold for display/precision-recall context; AUC itself "
        "is threshold-free. If you ran train.py's Stage 3, "
        "pass the value from that run's recommended_margin_stage3.json "
        "here rather than the Stage-1/2 default.",
    )
    parser.add_argument(
        "--width_grid",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WIDTH_GRID_HZ),
    )
    parser.add_argument(
        "--shape_widths",
        type=str,
        default=",".join(str(w) for w in DEFAULT_SHAPE_WIDTHS_HZ),
    )
    parser.add_argument("--n_per_width", type=int, default=200)
    parser.add_argument(
        "--n_shape_trials",
        type=int,
        default=300,
        help="Pairs per class and shape used for the bootstrap comparison.",
    )
    parser.add_argument("--broadened_snr_min", type=float, default=DEFAULT_SNR_RANGE[0])
    parser.add_argument("--broadened_snr_max", type=float, default=DEFAULT_SNR_RANGE[1])
    parser.add_argument("--run_snr_grid", action="store_true")
    parser.add_argument("--snr_grid", type=str, default="3,5,10,20,30")
    parser.add_argument("--snr_grid_widths", type=str, default="1,10,30,100,300,1000")
    parser.add_argument("--n_per_grid_point", type=int, default=80)
    parser.add_argument(
        "--shape_check_checkpoint",
        type=str,
        default=None,
        choices=["a", "b", "both"],
        help="Which checkpoint the shape-agnostic "
        "generalization check runs against. Default: 'b' when "
        "--checkpoint_b is supplied, otherwise 'a'.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="eval_broadening_results")
    args = parser.parse_args()
    main(args)
