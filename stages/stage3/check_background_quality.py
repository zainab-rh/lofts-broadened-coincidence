"""Screen filterbank backgrounds for non-Gaussian contamination.

The scanner reads a bounded time window by default, computes ordinary and
trimmed distribution statistics, and records failed or degenerate reads
explicitly.  Verdicts are screening heuristics; candidate backgrounds should
also be inspected as waterfalls before they are used for training.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
from pathlib import Path

import numpy as np
from json_utils import json_safe
from scipy import stats as sp_stats

# ------------------------------------------------------------------------
# Heuristic thresholds were calibrated against two B1508+55 observations
# (excess kurtosis approximately 3.2e7 and 1.6e8) and the zero excess
# kurtosis expected for ideal Gaussian noise. They are screening criteria,
# not statistically fitted decision boundaries.
# ------------------------------------------------------------------------
KURTOSIS_SAFE_MAX = 50.0
KURTOSIS_CAUTION_MAX = 1000.0
OUTLIER_SIGMA = 6.0
OUTLIER_FRAC_CAUTION = 1e-5
OUTLIER_FRAC_CONTAMINATED = 1e-3
# A field is considered potentially clippable when trimming 0.5% from each
# tail brings its excess kurtosis below the same threshold used for SAFE.
TRIMMED_KURTOSIS_SAFE_MAX = KURTOSIS_SAFE_MAX
TRIM_PCT_PER_TAIL = 0.5  # percent; 0.5% each tail = 1% total, comfortably
# above the largest outlier_fraction actually
# observed in the background scan (~0.13%)
# Very small arrays usually indicate an incomplete or placeholder read.
# They are classified as LOAD_FAILED before contamination is assessed.
MIN_SAMPLES_FOR_VERDICT = 100_000


def compute_background_stats(
    arr: np.ndarray,
    subsample_max: int = 5_000_000,
    seed: int = 0,
    trim_pct_per_tail: float = TRIM_PCT_PER_TAIL,
) -> dict:
    """
    Computes the same distributional statistics as
    `diagnostics.compare_noise_statistics`, plus an outlier-fraction metric
    and a trimmed/robust kurtosis, on a (possibly subsampled, for speed on
    large files) flattened array.
    Parameters
    ----------
    arr : np.ndarray
        Raw filterbank data array (any shape; flattened internally).
    subsample_max : int
        If the flattened array exceeds this many samples, a random
        subsample of this size is used (statistics on radio background
        data are stable well below this size; this keeps the scan fast
        for full-file arrays with tens of millions of channels).
    seed : int
        RNG seed for the subsample, for reproducibility.
    trim_pct_per_tail : float
        Percent (0-50) of samples clipped from EACH tail before computing
        `trimmed_excess_kurtosis`. See module documentation for rationale.
    Returns
    -------
    dict with mean, std, median, skewness, excess_kurtosis,
    trimmed_excess_kurtosis, min, max, n_samples_used, n_samples_total,
    and outlier_fraction (fraction of samples more than OUTLIER_SIGMA
    standard deviations from the mean).
    """
    arr = np.asarray(arr)
    n_total = int(arr.size)
    if n_total <= 1:
        # Treat empty and one-sample arrays explicitly; higher moments are
        # undefined and must not be interpreted as evidence of quiet data.
        single = float(arr.reshape(-1)[0]) if n_total == 1 else float("nan")
        return {
            "n_samples_used": n_total,
            "n_samples_total": n_total,
            "mean": single,
            "std": 0.0,
            "median": single,
            "skewness": float("nan"),
            "excess_kurtosis": float("nan"),
            "trimmed_excess_kurtosis": float("nan"),
            "min": single,
            "max": single,
            "outlier_fraction": float("nan"),
        }
    # Subsample before converting to float64 to avoid a full-array copy.
    if n_total > subsample_max:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_total, size=subsample_max, replace=False)
        flat = np.take(arr, indices).astype(np.float64)
    else:
        flat = arr.astype(np.float64).ravel()
    mean = float(flat.mean())
    std = float(flat.std())
    outlier_frac = (
        float(np.mean(np.abs(flat - mean) > OUTLIER_SIGMA * std)) if std else 0.0
    )
    # trimmed/robust excess kurtosis. See module documentation.
    if trim_pct_per_tail > 0 and flat.size > 10:
        lo, hi = np.percentile(flat, [trim_pct_per_tail, 100.0 - trim_pct_per_tail])
        trimmed = flat[(flat >= lo) & (flat <= hi)]
        trimmed_kurt = (
            float(sp_stats.kurtosis(trimmed)) if trimmed.size > 1 else float("nan")
        )
    else:
        trimmed_kurt = float("nan")
    return {
        "n_samples_used": int(flat.size),
        "n_samples_total": n_total,
        "mean": mean,
        "std": std,
        "median": float(np.median(flat)),
        "skewness": float(sp_stats.skew(flat)),
        "excess_kurtosis": float(sp_stats.kurtosis(flat)),
        "trimmed_excess_kurtosis": trimmed_kurt,
        "min": float(flat.min()),
        "max": float(flat.max()),
        "outlier_fraction": outlier_frac,
    }


def flag_pulsar_like_background(
    arr: np.ndarray,
    label: str = "background",
    kurtosis_safe_max: float = KURTOSIS_SAFE_MAX,
    kurtosis_caution_max: float = KURTOSIS_CAUTION_MAX,
    outlier_frac_caution: float = OUTLIER_FRAC_CAUTION,
    outlier_frac_contaminated: float = OUTLIER_FRAC_CONTAMINATED,
    trimmed_kurtosis_safe_max: float = TRIMMED_KURTOSIS_SAFE_MAX,
    min_samples_for_verdict: int = MIN_SAMPLES_FOR_VERDICT,
) -> dict:
    """
    Classify a background as quiet, cautionary, contaminated, potentially
    recoverable by clipping, or unreadable.
    This is a HEURISTIC, not a certainty. Always pair a SAFE or
    CONTAMINATED_BUT_CLIPPABLE verdict with a visual check (a waterfall
    plot) before committing to a field as the primary training background.
    Returns
    -------
    dict with keys: 'label', 'verdict' (one of 'SAFE',
    'CONTAMINATED_BUT_CLIPPABLE', 'CAUTION', 'LIKELY_CONTAMINATED',
    'LOAD_FAILED'), 'stats' (the compute_background_stats dict), and
    'message' (a one-paragraph, printable explanation).
    """
    stats = compute_background_stats(arr)
    k = stats["excess_kurtosis"]
    tk = stats["trimmed_excess_kurtosis"]
    of = stats["outlier_fraction"]
    n_used = stats["n_samples_used"]
    # Check validity first because comparisons with NaN are always false.
    if n_used < min_samples_for_verdict or not np.isfinite(k):
        verdict = "LOAD_FAILED"
    elif k >= kurtosis_caution_max or of >= outlier_frac_contaminated:
        if np.isfinite(tk) and tk < trimmed_kurtosis_safe_max:
            verdict = "CONTAMINATED_BUT_CLIPPABLE"
        else:
            verdict = "LIKELY_CONTAMINATED"
    elif k >= kurtosis_safe_max or of >= outlier_frac_caution:
        verdict = "CAUTION"
    else:
        verdict = "SAFE"
    messages = {
        "SAFE": (
            f"'{label}': excess kurtosis={k:.2f}, outlier_fraction={of:.2e} "
            f"-> SAFE. Consistent with a thermal-noise/ordinary-RFI-dominated "
            f"background; no strong periodic-transient contamination detected."
        ),
        "CONTAMINATED_BUT_CLIPPABLE": (
            f"'{label}': excess kurtosis={k:.2f} but TRIMMED excess kurtosis (after "
            f"clipping the extreme {TRIM_PCT_PER_TAIL:.1f}% per tail)={tk:.2f}, "
            f"outlier_fraction={of:.2e} -> CONTAMINATED_BUT_CLIPPABLE. The raw "
            f"kurtosis is dominated by a small number of extreme samples, not by the "
            f"bulk of the distribution -- consistent with occasional broadband RFI "
            f"spikes sitting on an otherwise quiet floor, rather than a genuinely "
            f"non-Gaussian bulk signal (e.g. a folded pulsar). Standard sigma-clipping "
            f"of the top outliers before use as a training background may make this "
            f"field usable; verify with a waterfall plot before committing to it."
        ),
        "CAUTION": (
            f"'{label}': excess kurtosis={k:.2f}, outlier_fraction={of:.2e} "
            f"-> CAUTION. Elevated beyond a quiescent field, but well below "
            f"pulsar-grade contamination (reference: B1508+55 measured "
            f"~3.2e7-1.6e8). Could be heavy but ordinary RFI. Recommend a "
            f"visual waterfall check before using as a primary training "
            f"background."
        ),
        "LIKELY_CONTAMINATED": (
            f"'{label}': excess kurtosis={k:.2f}, trimmed kurtosis={tk:.2f}, "
            f"outlier_fraction={of:.2e} -> LIKELY_CONTAMINATED. Unlike "
            f"CONTAMINATED_BUT_CLIPPABLE, the contamination here does not go away "
            f"after trimming extreme samples, meaning it's a property of the bulk "
            f"distribution, not just a few spikes -- characteristic of strong, "
            f"narrow, periodic transients (e.g. a pulsar). Using this as the ML "
            f"training background risks (a) diverting the weighted-MSE "
            f"reconstruction loss toward reconstructing pulses instead of injected "
            f"technosignatures, and (b) letting the network key off 'does this tile "
            f"contain a pulse' as a spurious per-station discriminative feature. "
            f"Strongly recommend switching to a quiescent SETI target field."
        ),
        "LOAD_FAILED": (
            f"'{label}': only {n_used} sample(s) were available to compute "
            f"statistics (excess kurtosis={k}). This is not evidence the field is "
            f"quiet -- the read did not actually succeed. Most likely cause in this "
            f"pipeline: blimpy silently substituting a placeholder array (instead of "
            f"raising) when a multi-GB file exceeded its internal max_load or "
            f"available memory at read time, especially deep into a long batch scan "
            f"where memory from prior large files had not been released. This file "
            f"was not evaluated and must be excluded from both SAFE and "
            f"CONTAMINATED decisions until it is re-scanned successfully -- see "
            f"'--files <this path> --full_load' or a smaller --chunk_t_samples."
        ),
    }
    return {
        "label": label,
        "verdict": verdict,
        "stats": stats,
        "message": messages[verdict],
    }


def summarize_background_quality(results: list) -> str:
    """Formats a list of `flag_pulsar_like_background` results as a
    plain-text ranked table (best/quietest first), for CLI printing or
    inclusion in a training report. LOAD_FAILED entries are grouped at the
    bottom because their kurtosis is not a meaningful ranking signal."""
    verdict_rank = {
        "SAFE": 0,
        "CONTAMINATED_BUT_CLIPPABLE": 1,
        "CAUTION": 2,
        "LIKELY_CONTAMINATED": 3,
        "LOAD_FAILED": 4,
    }

    def _sort_key(r):
        rank = verdict_rank.get(r["verdict"], 5)
        k = r["stats"].get("excess_kurtosis", float("inf"))
        k = k if (k is not None and np.isfinite(k)) else float("inf")
        return (rank, k)

    ranked = sorted(results, key=_sort_key)
    lines = [
        f"{'Verdict':<26}{'Excess Kurt':>14}{'Trimmed Kurt':>14}{'Outlier Frac':>14}  {'Label'}"
    ]
    lines.append("-" * 100)
    for r in ranked:
        s = r["stats"]
        k = s.get("excess_kurtosis", float("nan"))
        tk = s.get("trimmed_excess_kurtosis", float("nan"))
        of = s.get("outlier_fraction", float("nan"))
        lines.append(
            f"{r['verdict']:<26}{k:>14.2f}{tk:>14.2f}{of:>14.2e}  {r['label']}"
        )
    return "\n".join(lines)


def _construct_waterfall(path: str, kwargs: dict):
    """Constructs a blimpy Waterfall, dropping `max_load` from kwargs and
    retrying if the installed blimpy version doesn't accept it as a
    keyword (blimpy's accepted kwargs vary by version; this
    keeps the loader working either way instead of hard-failing on a
    version mismatch)."""
    import blimpy as bl

    try:
        return bl.Waterfall(str(path), **kwargs)
    except TypeError as e:
        if "max_load" in kwargs and "max_load" in str(e):
            kwargs = {k: v for k, v in kwargs.items() if k != "max_load"}
            return bl.Waterfall(str(path), **kwargs)
        raise


def _load_filterbank_array(
    path: str, max_load_gb: float = 16.0, chunk_t_samples: "int | None" = 16
) -> np.ndarray:
    """
    Load a bounded filterbank array through blimpy.

    The import is local so the statistics functions remain testable without
    blimpy. Bounded time reads prevent multi-gigabyte observations from being
    loaded in full during a directory scan.
    Parameters
    ----------
    path : str
    max_load_gb : float
        Passed to blimpy as its own internal size guard, so behaviour is
        an explicit, tunable quantity rather than a silent version-
        dependent default. Dropped automatically if the installed blimpy
        version doesn't accept it (see `_construct_waterfall`).
    chunk_t_samples : int or None
        Number of initial time integrations to read across the full band.
        Pass ``None`` to read the complete file. A bounded window is suitable
        for screening but does not replace a full-file check of shortlisted
        backgrounds.
    """
    path = Path(path)
    kwargs = {"max_load": max_load_gb}
    if chunk_t_samples is not None:
        kwargs["t_start"] = 0
        kwargs["t_stop"] = int(chunk_t_samples)
    wf = _construct_waterfall(str(path), kwargs)
    data = wf.data
    arr = data[:, 0, :] if data.ndim == 3 else data
    # Reject placeholder arrays returned by some blimpy failure modes.
    if arr.size < 1000:
        raise RuntimeError(
            f"blimpy returned only {arr.size} data sample(s) (shape {arr.shape}) "
            f"for {path.name}. This is almost certainly a silent load failure "
            f"(max_load exceeded, insufficient memory at read time, or a "
            f"truncated/empty file) rather than a genuinely tiny observation. "
            f"Re-scan this file on its own with --full_load and more available "
            f"memory, or try a smaller --chunk_t_samples."
        )
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Scan filterbank files for pulsar/transient contamination "
        "before use as machine-learning backgrounds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit list of .fil file paths to scan.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Glob pattern of .fil files to scan (alternative to --files). "
        "Use '**' with a recursive glob (Python's glob.glob(..., "
        "recursive=True)) to scan an entire directory tree.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional path to write the full results as JSON.",
    )
    parser.add_argument(
        "--max_load_gb",
        type=float,
        default=16.0,
        help="Passed to blimpy's Waterfall as its own size "
        "guard (GB). Dropped automatically if unsupported by your "
        "blimpy version.",
    )
    parser.add_argument(
        "--chunk_t_samples",
        type=int,
        default=16,
        help="Read only this many time integrations per file "
        "instead of the whole file. See _load_filterbank_array's "
        "docstring. Ignored if --full_load is set.",
    )
    parser.add_argument(
        "--full_load",
        action="store_true",
        help="Disable chunked reading and load each file in "
        "full. Only advisable for a small number of files at a "
        "time (e.g. final spot-check of your top 1-2 candidates), "
        "not a full-tree scan.",
    )
    args = parser.parse_args()
    paths = list(args.files) if args.files else []
    if args.glob:
        paths.extend(sorted(glob.glob(args.glob, recursive=True)))
    if not paths:
        parser.error("Provide --files and/or --glob.")
    chunk = None if args.full_load else args.chunk_t_samples
    results = []
    for p in paths:
        p = Path(p)
        print(f"Scanning: {p}")
        try:
            arr = _load_filterbank_array(
                str(p), max_load_gb=args.max_load_gb, chunk_t_samples=chunk
            )
        except Exception as e:
            # Preserve failed reads in the report rather than dropping them.
            print(f"  ERROR loading {p}: {e}")
            results.append(
                {
                    "label": p.name,
                    "path": str(p),
                    "verdict": "LOAD_FAILED",
                    "stats": {
                        "n_samples_used": 0,
                        "n_samples_total": 0,
                        "excess_kurtosis": float("nan"),
                        "trimmed_excess_kurtosis": float("nan"),
                        "outlier_fraction": float("nan"),
                    },
                    "message": f"Load error: {e}",
                }
            )
            continue
        result = flag_pulsar_like_background(arr, label=str(p.name))
        result["path"] = str(p)
        print(f"  {result['message']}")
        results.append(result)
        # explicit memory hygiene between files, so one file's
        # footprint can't accumulate into the next file's read.
        del arr, result
        gc.collect()
    print("\n" + "=" * 100)
    print(
        "SUMMARY (quietest first; LOAD_FAILED grouped at the bottom -- never measured)"
    )
    print("=" * 100)
    print(summarize_background_quality(results))
    safe = [r for r in results if r["verdict"] == "SAFE"]
    clippable = [r for r in results if r["verdict"] == "CONTAMINATED_BUT_CLIPPABLE"]
    failed = [r for r in results if r["verdict"] == "LOAD_FAILED"]
    if safe:
        print(f"\nRecommended for training background use ({len(safe)} candidate(s)):")
        for r in safe:
            print(f"  - {r['path']}")
    else:
        print("\nWARNING: no file scanned came back SAFE.")
        if clippable:
            print(
                f"{len(clippable)} file(s) came back CONTAMINATED_BUT_CLIPPABLE -- their "
                f"raw kurtosis is driven by a small number of extreme samples, not the "
                f"bulk distribution. These are worth investigating as fallback candidates "
                f"IF paired with sigma-clipping before use, after a visual waterfall check:"
            )
            for r in clippable:
                print(f"  - {r['path']}")
        else:
            print(
                "Consider widening the search, or proceed with a CAUTION-tier file after "
                "a manual waterfall inspection."
            )
    if failed:
        print(
            f"\n{len(failed)} file(s) returned LOAD_FAILED and were not evaluated either "
            f"way. Re-scan these individually, e.g.:"
        )
        print("  python check_background_quality.py --files <path> --full_load")
        for r in failed[:10]:
            print(f"  - {r['path']}")
        if len(failed) > 10:
            print(
                f"  ... and {len(failed) - 10} more (see --out_json for the full list)"
            )
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(json_safe(results), f, indent=2, allow_nan=False)
        print(f"\nFull results written to {args.out_json}")


if __name__ == "__main__":
    main()
