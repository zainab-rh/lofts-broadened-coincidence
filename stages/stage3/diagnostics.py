"""Visual and statistical diagnostics for the Stage 3 pipeline.

The module compares station backgrounds, renders representative pair classes
and injection profiles, and evaluates whether controlled noise-scale mismatch
changes latent-distance separation at several broadening widths.
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
from astropy import units as u
from check_background_quality import flag_pulsar_like_background
from json_utils import json_safe
from scipy import stats as sp_stats
from train import (
    BROADENED_CASE_PROBS,
    CONFIGS,
    DEFAULT_BROADENED_SNR_RANGE,
    DEFAULT_MIN_EFFECTIVE_SNR,
    FILTERBANK_PATHS,
    UNet,
    generate_pair,
    get_real_slice,
    load_background,
    preprocess_np,
)

# =============================================================================
# 1. Noise statistics for the two station backgrounds
# =============================================================================


def compare_noise_statistics(
    real_data_a: np.ndarray,
    real_data_b: np.ndarray,
    label_a: str = "Sweden",
    label_b: str = "Ireland",
    out_dir: Path = None,
) -> dict:
    """
    Report distribution statistics and normalised bandpass profiles for both
    station backgrounds.  The background-quality screen converts these
    descriptive statistics into a documented heuristic verdict.
    """

    def _stats(arr, label):
        n_total = arr.size
        if n_total > 5000000:
            rng = np.random.default_rng(0)
            indices = rng.choice(n_total, size=5000000, replace=False)
            flat = np.take(arr, indices).astype(np.float64)
        else:
            flat = np.asarray(arr, dtype=np.float64).ravel()
        return {
            "label": label,
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "median": float(np.median(flat)),
            "skewness": float(sp_stats.skew(flat)),
            "excess_kurtosis": float(sp_stats.kurtosis(flat)),
            "min": float(flat.min()),
            "max": float(flat.max()),
        }

    stats_a = _stats(real_data_a, label_a)
    stats_b = _stats(real_data_b, label_b)
    print(f"\n--- Noise statistics: {label_a} vs {label_b} ---")
    for key in ("mean", "std", "median", "skewness", "excess_kurtosis", "min", "max"):
        print(
            f"  {key:<18} {label_a}={stats_a[key]:>12.4f}   {label_b}={stats_b[key]:>12.4f}   "
            f"ratio={stats_a[key] / stats_b[key] if stats_b[key] != 0 else float('nan'):.3f}"
        )
    result = {
        "station_a": stats_a,
        "station_b": stats_b,
        "std_ratio": (
            stats_a["std"] / stats_b["std"] if stats_b["std"] != 0 else float("nan")
        ),
    }
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        rng = np.random.default_rng(0)
        n_a, n_b = real_data_a.size, real_data_b.size

        if n_a > 200_000:
            idx_a = rng.choice(n_a, 200_000, replace=False)
            sample_a = np.take(real_data_a, idx_a).astype(np.float64)
        else:
            sample_a = np.asarray(real_data_a, dtype=np.float64).ravel()

        if n_b > 200_000:
            idx_b = rng.choice(n_b, 200_000, replace=False)
            sample_b = np.take(real_data_b, idx_b).astype(np.float64)
        else:
            sample_b = np.asarray(real_data_b, dtype=np.float64).ravel()
        ax1.hist(sample_a, bins=100, alpha=0.6, label=label_a, density=True)
        ax1.hist(sample_b, bins=100, alpha=0.6, label=label_b, density=True)
        ax1.set_title("Raw amplitude distribution")
        ax1.set_xlabel("Raw value")
        ax1.legend()
        ax1.grid(alpha=0.3)
        bandpass_a = real_data_a.mean(axis=0)
        bandpass_b = real_data_b.mean(axis=0)
        ax2.plot(
            bandpass_a / bandpass_a.mean(), label=f"{label_a} (normalised)", alpha=0.8
        )
        ax2.plot(
            bandpass_b / bandpass_b.mean(), label=f"{label_b} (normalised)", alpha=0.8
        )
        ax2.set_title("Mean bandpass shape (normalised to own mean)")
        ax2.set_xlabel("Channel")
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "noise_statistics_comparison.png", dpi=130)
        plt.close(fig)
        with open(out_dir / "noise_statistics.json", "w") as f:
            json.dump(json_safe(result), f, indent=2, allow_nan=False)
    return result


# =============================================================================
# 2. Representative image pair for each case category
# =============================================================================


def visualize_all_cases(
    mode: str,
    real_data: np.ndarray,
    header: dict,
    real_data_b,
    freq_ghz: float,
    p_mdwarf: float,
    snr_range: tuple,
    out_dir: Path,
    seed: int = 0,
    broadened_max_width_hz: float = 20000.0,
    broadened_min_effective_snr: float = DEFAULT_MIN_EFFECTIVE_SNR,
    broadened_use_conditioned_sampling: bool = True,
):
    """
    Save one two-station spectrogram pair for every case in
    ``BROADENED_CASE_PROBS``.  The same conditioned-sampling parameters used
    during training are applied here so the examples represent the model's
    actual training distribution.
    """
    np.random.seed(
        seed
    )  # generate_pair uses the global np.random, not an injectable Generator
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    descriptions = {
        "astro_match": "Narrowband signal shared by both stations -> match (label=1)",
        "astro_mismatch": "Narrowband signal present at only one station, or two DIFFERENT "
        "narrowband signals -> genuine mismatch (label=0)",
        "rfi_only_mismatch": "NO injected signal at all -- pure real background/local-RFI "
        "noise from each station, no shared content -> mismatch (label=0)",
        "broadened_match": "Lorentzian-broadened signal with the same width and drift at both "
        "stations -> match (label=1)",
        "shape_decoy_match": "Box- or Gaussian-shaped decoy with matched width, drift, and energy "
        "at both stations -> match (label=1). Tests shape-agnostic "
        "generalisation, not Lorentzian memorisation.",
        "broadened_mismatch_onesided": "Broadened signal at one station only, empty background "
        "at the other -> mismatch (label=0)",
        "broadened_mismatch_different": "Broadened signal at BOTH stations simultaneously, but "
        "with different width/drift (two unrelated local events) "
        "-> mismatch (label=0)",
    }
    for case_name in BROADENED_CASE_PROBS:
        frame1, frame2, label = generate_pair(
            mode,
            case_name,
            real_data,
            header,
            "broadened",
            real_data_b=real_data_b,
            freq_ghz=freq_ghz,
            p_mdwarf=p_mdwarf,
            broadened_snr_range=snr_range,
            broadened_max_width_hz=broadened_max_width_hz,
            broadened_min_effective_snr=broadened_min_effective_snr,
            broadened_use_conditioned_sampling=broadened_use_conditioned_sampling,
        )
        if frame1 is None:
            print(f"WARNING: could not generate an example for case '{case_name}'.")
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        ax1.imshow(frame1.get_data(), aspect="auto", origin="lower", cmap="viridis")
        ax1.set_title("Station A")
        ax2.imshow(frame2.get_data(), aspect="auto", origin="lower", cmap="viridis")
        ax2.set_title("Station B")
        for ax in (ax1, ax2):
            ax.set_xlabel("Frequency channel")
            ax.set_ylabel("Time sample")
        fig.suptitle(f"Case: {case_name}   (label={label})", fontsize=13, y=1.02)
        fig.text(
            0.5,
            -0.04,
            descriptions[case_name],
            ha="center",
            va="top",
            fontsize=9,
            wrap=True,
            transform=fig.transFigure,
        )
        plt.tight_layout()
        save_path = out_dir / f"example_{case_name}.png"
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


def render_signal_only_preview(
    shape_widths_hz: tuple,
    out_dir: Path,
    fchans: int = 256,
    tchans: int = 64,
    df_hz: float = ls.LOFAR_DF_HZ,
    dt_s: float = ls.LOFAR_DT_S,
    fch1_mhz: float = ls.LOFAR_FCH1_MHZ,
    seed: int = 0,
):
    """
    Render each injection shape on low-variance Gaussian noise.  This view
    isolates injection morphology from real bandpass and RFI structure and is
    intended as a visual implementation check.
    Parameters
    ----------
    shape_widths_hz : tuple of (shape_name, width_hz) pairs to render.
    out_dir : Path
    fchans, tchans, df_hz, dt_s, fch1_mhz : Frame geometry (defaults use a
        higher tchans than the `high_freq` mode's native 16 rows,
        specifically so the continuity of the injected streak is easy to
        see by eye).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    np.random.seed(
        seed
    )  # setigen's add_noise draws from the global RNG, not an injectable one
    fig, axes = plt.subplots(
        1, len(shape_widths_hz), figsize=(5 * len(shape_widths_hz), 4.5)
    )
    if len(shape_widths_hz) == 1:
        axes = [axes]
    for ax, (shape, width_hz) in zip(axes, shape_widths_hz):
        frame = stg.Frame(
            fchans=fchans,
            tchans=tchans,
            df=df_hz * u.Hz,
            dt=dt_s * u.s,
            fch1=fch1_mhz * u.MHz,
            ascending=True,
        )
        frame.add_noise(x_mean=0.0, x_std=0.05, noise_type="gaussian")
        f_start = frame.get_frequency(fchans // 2)
        drift_hz_s = 0.0  # zero drift: isolates shape/continuity, not drift rendering
        base_level = 10.0  # high SNR, purely for visual clarity
        if shape == "lorentzian":
            ls.inject_lorentzian_signal(
                frame, f_start, drift_hz_s, base_level, width_hz
            )
        elif shape == "box":
            ls.inject_box_decoy_signal(frame, f_start, drift_hz_s, base_level, width_hz)
        elif shape == "gaussian":
            ls.inject_gaussian_decoy_signal(
                frame, f_start, drift_hz_s, base_level, width_hz
            )
        ax.imshow(frame.get_data(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(
            f"{shape}, Δνsb={width_hz} Hz\n(retention={ls.peak_retention_factor(width_hz)*100:.1f}%)"
        )
        ax.set_xlabel("Frequency channel")
        ax.set_ylabel("Time sample")
    fig.suptitle(
        "Signal-only injection preview (Δt=0 real background, isolates injection "
        "morphology)\nExpect a SMOOTH, CONTINUOUS streak -- t_profile is constant "
        "in time by construction",
        fontsize=11,
    )
    plt.tight_layout()
    save_path = out_dir / "signal_only_preview.png"
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# 3. Controlled inter-station noise-scale mismatch
# =============================================================================


def _rescale_slice_noise(slice_data: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scale a real-noise slice about its mean without changing its structure."""
    mean_val = slice_data.mean()
    return (
        mean_val + (slice_data.astype(np.float64) - mean_val) * scale_factor
    ).astype(np.float32)


def test_signal_with_mismatched_noise(
    model,
    mode: str,
    real_data: np.ndarray,
    header: dict,
    real_data_b,
    device,
    width_hz: float = 30.0,
    noise_scale_factors: tuple = (1.0, 1.5, 2.0, 3.0, 5.0),
    snr_range: tuple = DEFAULT_BROADENED_SNR_RANGE,
    margin: float = 0.8,
    n_trials: int = 60,
    seed: int = 0,
) -> dict:
    """
    Compare pairs with and without a shared signal while scaling the variance
    of station B.  Running the experiment at several fixed widths separates a
    noise-mismatch effect from broadening-limited detectability.
    """
    rng = np.random.default_rng(seed)
    tile_shape = CONFIGS[mode]["frame_shape"]
    df, dt = abs(header["foff"]) * 1e6, header["tsamp"]
    fch1, ascending = header["fch1"], header["foff"] > 0
    data_b_source = real_data_b if real_data_b is not None else real_data
    results = {}
    for scale in noise_scale_factors:
        for condition in ("with_signal", "without_signal"):
            distances = []
            for _ in range(n_trials):
                slice1 = get_real_slice(real_data, tile_shape)
                slice2 = get_real_slice(data_b_source, tile_shape)
                if slice1 is None or slice2 is None:
                    continue
                slice2 = _rescale_slice_noise(slice2, scale)
                frame1 = stg.Frame.from_data(
                    df=df, dt=dt, fch1=fch1, ascending=ascending, data=slice1
                )
                frame2 = stg.Frame.from_data(
                    df=df, dt=dt, fch1=fch1, ascending=ascending, data=slice2
                )
                if condition == "with_signal":
                    draw = ls.sample_one_broadening(rng=rng)
                    noise_mean1, noise_std1 = frame1.get_noise_stats()
                    noise_mean2, noise_std2 = frame2.get_noise_stats()
                    snr1 = float(rng.uniform(*snr_range))
                    snr2 = float(rng.uniform(*snr_range))
                    level1 = noise_mean1 + (noise_std1 or 1.0) * snr1
                    level2 = noise_mean2 + (noise_std2 or 1.0) * snr2
                    r1 = ls.inject_lorentzian_signal(
                        frame1, None, draw.drift_hz_per_s, level1, width_hz
                    )
                    ls.inject_lorentzian_signal(
                        frame2, r1["f_start_hz"], draw.drift_hz_per_s, level2, width_hz
                    )
                img1 = torch.from_numpy(
                    preprocess_np(frame1.get_data())[None, None, ...]
                ).to(device)
                img2 = torch.from_numpy(
                    preprocess_np(frame2.get_data())[None, None, ...]
                ).to(device)
                with torch.no_grad():
                    _, z1 = model(img1)
                    _, z2 = model(img2)
                d = F.pairwise_distance(z1, z2).item()
                distances.append(d)
            distances = np.array(distances)
            key = f"scale_{scale}"
            results.setdefault(key, {})
            confidence = (
                float(np.mean(np.maximum(0, 1 - distances / margin) * 100))
                if len(distances)
                else float("nan")
            )
            results[key][condition] = {
                "mean_distance": (
                    float(distances.mean()) if len(distances) else float("nan")
                ),
                "std_distance": (
                    float(distances.std()) if len(distances) else float("nan")
                ),
                "mean_confidence_pct": confidence,
                "n": int(len(distances)),
            }
            print(
                f"  width={width_hz:>5.1f}Hz scale={scale:>4.1f} | {condition:<15} | "
                f"mean_dist={distances.mean():.4f}  mean_conf={confidence:.1f}%"
            )
    return results


def plot_mismatched_noise_test_multiwidth(
    results_by_width: dict, noise_scale_factors: list, margin: float, save_path: Path
):
    """
    Plot with-signal and without-signal distances at each tested width.
    """
    widths = sorted(results_by_width.keys())
    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        1, len(widths), figsize=(6 * len(widths), 5.5), sharey=True
    )
    if len(widths) == 1:
        axes = [axes]
    for ax, width_hz in zip(axes, widths):
        results = results_by_width[width_hz]
        with_sig = [
            results[f"scale_{s}"]["with_signal"]["mean_distance"]
            for s in noise_scale_factors
        ]
        without_sig = [
            results[f"scale_{s}"]["without_signal"]["mean_distance"]
            for s in noise_scale_factors
        ]
        ax.plot(
            noise_scale_factors,
            with_sig,
            marker="o",
            color="lightgreen",
            label="Same signal, both stations",
            lw=2,
        )
        ax.plot(
            noise_scale_factors,
            without_sig,
            marker="o",
            color="salmon",
            label="No signal (RFI-mismatch analogue)",
            lw=2,
        )
        ax.axhline(margin, color="yellow", ls="--", lw=1, label=f"Margin ({margin})")
        retention_pct = ls.peak_retention_factor(width_hz) * 100
        ax.set_xlabel("Station-B noise variance rescale factor")
        ax.set_title(f"Δνsb = {width_hz:.1f} Hz (retention={retention_pct:.1f}%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean latent distance")
    fig.suptitle(
        "Sensitivity to an inter-station noise-scale mismatch\n"
        "(fixed widths from readily detectable to near the detection floor)",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fb_path = args.filterbank or FILTERBANK_PATHS[args.mode]
    real_data, header = load_background(fb_path)
    real_data_b = None
    if args.station_b_filterbank:
        real_data_b, _ = load_background(args.station_b_filterbank)
    else:
        print(
            "WARNING: no --station_b_filterbank given; station B will reuse station A's "
            "background, which means Parts 1 and 3 below are not meaningful cross-station "
            "tests. Pass --station_b_filterbank for a genuine dual-site diagnostic."
        )
    print("\n=== PART 0: background quality check ===")
    qa_a = flag_pulsar_like_background(
        real_data, label=f"station_A:{Path(fb_path).name}"
    )
    print(f"  {qa_a['message']}")
    qa_results = [qa_a]
    if real_data_b is not None:
        qa_b = flag_pulsar_like_background(
            real_data_b, label=f"station_B:{Path(args.station_b_filterbank).name}"
        )
        print(f"  {qa_b['message']}")
        qa_results.append(qa_b)
    with open(out_dir / "background_quality.json", "w") as f:
        json.dump(json_safe(qa_results), f, indent=2, allow_nan=False)
    print("\n=== PART 1: noise statistics comparison ===")
    if real_data_b is not None:
        compare_noise_statistics(
            real_data, real_data_b, "Station A", "Station B", out_dir
        )
    else:
        print("Skipped (no station-B background provided).")
    print("\n=== PART 2: one example image per case ===")
    visualize_all_cases(
        args.mode,
        real_data,
        header,
        real_data_b,
        args.freq_ghz,
        args.p_mdwarf,
        (args.broadened_snr_min, args.broadened_snr_max),
        out_dir,
        seed=args.seed,
        broadened_max_width_hz=20000.0,
        broadened_min_effective_snr=args.broadened_min_effective_snr,
        broadened_use_conditioned_sampling=not args.broadened_disable_conditioning,
    )
    print("\n=== PART 2b: clean signal-only injection preview ===")
    render_signal_only_preview(
        shape_widths_hz=(("lorentzian", 10.0), ("box", 10.0), ("gaussian", 10.0)),
        out_dir=out_dir,
        seed=args.seed,
    )
    if args.checkpoint:
        print("\n=== PART 3: same signal under mismatched station noise scales ===")
        model = UNet().to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        scales = [float(s) for s in args.noise_scales.split(",")]
        widths = [float(w) for w in args.mismatched_noise_widths_hz.split(",")]
        results_by_width = {}
        for width_hz in widths:
            print(
                f"\n  -- width={width_hz} Hz "
                f"(retention={ls.peak_retention_factor(width_hz)*100:.1f}%) --"
            )
            results_by_width[width_hz] = test_signal_with_mismatched_noise(
                model,
                args.mode,
                real_data,
                header,
                real_data_b,
                device,
                width_hz=width_hz,
                noise_scale_factors=scales,
                snr_range=(args.broadened_snr_min, args.broadened_snr_max),
                margin=args.margin,
                n_trials=args.n_trials,
                seed=args.seed,
            )
        plot_mismatched_noise_test_multiwidth(
            results_by_width, scales, args.margin, out_dir / "mismatched_noise_test.png"
        )
        with open(out_dir / "mismatched_noise_test.json", "w") as f:
            json.dump(
                json_safe({str(k): v for k, v in results_by_width.items()}),
                f,
                indent=2,
                allow_nan=False,
            )
        print("\n  Interpretation guide:")
        print(
            "  - At the EASIEST width (lowest Hz, highest retention%), 'with_signal' should"
        )
        print(
            "    sit CLEARLY below 'without_signal' at every noise scale -- if it does not,"
        )
        print("    that is evidence of a noise-statistics confound.")
        print(
            "  - At the HARDEST width (highest Hz, lowest retention%), the two curves"
        )
        print(
            "    converging is PHYSICALLY EXPECTED (the signal itself is near-invisible)"
        )
        print("    and should not be read as a confound at that width specifically.")
    else:
        print("\nPART 3 skipped: pass --checkpoint to run the mismatched-noise sweep.")
    print(f"\nAll diagnostics written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual and statistical diagnostics for the dual-station comparison pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, required=True, choices=CONFIGS.keys())
    parser.add_argument("--filterbank", type=str, default=None)
    parser.add_argument(
        "--station_b_filterbank",
        type=str,
        default=None,
        help="Station B's real filterbank -- required for a genuine dual-site "
        "diagnostic (Parts 1 and 3 are not meaningful without it).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Trained UNet checkpoint. If omitted, Part 3 is skipped.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.8,
        help="For a Stage-3 checkpoint, pass the "
        "value from that run's recommended_margin_stage3.json here.",
    )
    parser.add_argument("--freq_ghz", type=float, default=0.150)
    parser.add_argument("--p_mdwarf", type=float, default=0.75)
    parser.add_argument(
        "--broadened_snr_min", type=float, default=DEFAULT_BROADENED_SNR_RANGE[0]
    )
    parser.add_argument(
        "--broadened_snr_max", type=float, default=DEFAULT_BROADENED_SNR_RANGE[1]
    )
    parser.add_argument(
        "--broadened_min_effective_snr", type=float, default=DEFAULT_MIN_EFFECTIVE_SNR
    )
    parser.add_argument(
        "--broadened_disable_conditioning",
        action="store_true",
        help="Use the raw, unconditioned population sampler for the "
        "example images in Part 2 (legacy ablation).",
    )
    parser.add_argument(
        "--mismatched_noise_widths_hz",
        type=str,
        default="1.0,10.0,30.0",
        help="Comma-separated widths (Hz) for the noise-mismatch sweep. "
        "The default spans a readily detectable signal to one near the "
        "broadening-limited detection floor.",
    )
    parser.add_argument("--noise_scales", type=str, default="1.0,1.5,2.0,3.0,5.0")
    parser.add_argument("--n_trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="diagnostics_results")
    args = parser.parse_args()
    main(args)
