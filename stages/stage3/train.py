"""Train the three-stage dual-station coincidence model.

Stages 1 and 2 learn narrowband coincidence on simplified and realistic
backgrounds.  The optional third stage fine-tunes the same U-Net encoder on
physically broadened signals, uses a separately configurable contrastive
margin, and derives a post-hoc operating threshold from held-out case-wise
distances.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import blimpy as bl
import matplotlib.pyplot as plt
import numpy as np
import setigen as stg
import torch
import torch._dynamo
import torch.nn.functional as F
from astropy import units as u
from torch import nn, optim

try:
    # torch>=2.4 moved GradScaler/autocast to a device-agnostic torch.amp API
    # and deprecated the old torch.cuda.amp aliases (still functional but
    # noisy). This keeps CUDA-targeted mixed precision
    # on both old and new torch installs without depending on which one is
    # on the actual training machine.
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast

    def GradScaler(*args, **kwargs):
        return _GradScaler("cuda", *args, **kwargs)

    def autocast(*args, **kwargs):
        return _autocast("cuda", *args, **kwargs)

except (ImportError, TypeError):  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast

# Physical broadening and injection model.
import lorentzian_signals as ls
import threshold_utils as tu

# Background-quality screening and threshold calibration.
from check_background_quality import (
    flag_pulsar_like_background,
    summarize_background_quality,
)
from json_utils import json_safe
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Optional dependency for the Stage-3 stratified AUC-ROC report. Baseline
# training can still run when scikit-learn is unavailable.
try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None
# ==============================================================================
#  1. CONFIGURATION
# ==============================================================================
# Signal generation parameters
SIGNAL_SNR_RANGE = (1.5, 10)
SIGNAL_DEFINITIONS = {
    "drifting_narrowband": {
        "generator": "add_constant_signal",
        "params": {
            "width_df_multiplier": lambda: np.random.uniform(1, 10),
            "f_profile_type": "gaussian",
            "drift_rate": lambda: np.random.uniform(-0.5, 0.5) * u.Hz / u.s,
        },
    },
    "frb_pulse": {
        "generator": "add_signal",
        "params": {
            "f_width_percent": 0.4,
            "duration": lambda frame: frame.dt * np.random.randint(2, 5),
        },
    },
    "blob": {
        "generator": "add_signal",
        "params": {
            "f_width_percent": lambda: np.random.uniform(0.1, 0.25),
            "duration": lambda frame: frame.dt
            * np.random.randint(int(frame.tchans * 0.1), int(frame.tchans * 0.3)),
        },
    },
}
ALLOWED_SIGNALS = list(SIGNAL_DEFINITIONS.keys())
# Model and data resolution configurations
CONFIGS = {
    "high_freq": {"frame_shape": (16, 1024), "allowed_signals": ALLOWED_SIGNALS},
    "high_time": {"frame_shape": (512, 128), "allowed_signals": ALLOWED_SIGNALS},
    "mid_res": {"frame_shape": (256, 256), "allowed_signals": ALLOWED_SIGNALS},
}
# Paths to background noise data from filterbank files.
#
# The broadening analysis requires the fine-frequency ``0000`` product.
# 'high_freq' -> *.rawspec.0000.fil is the CORRECT product for narrowband/
# broadening work (fine spectral resolution, Johnson et al. 2023 Table 1's
# 2.98 Hz channel width), per the selected fine-frequency analysis. Do not
# use *.rawspec.0001.fil (fine-TIME, kHz-level spectral resolution --
# spectral broadening is not meaningful at that resolution) for this work.
#
# The path below (B1508+55) is a PULSAR and was confirmed, via
# `check_background_quality.py`, to have excess kurtosis of ~3.2e7-1.6e8 --
# 5-6 orders of magnitude above a quiescent field. Before a full training
# run, scan the available SETI
# target fields (`python check_background_quality.py --glob
# "/datax2/projects/LOFTS/*/LOFTS*/LOFTS*.rawspec.0000.fil"`) and replace
# these paths with screened background fields.
FILTERBANK_PATHS = {
    "high_freq": "/datax2/projects/LOFTS/2025-05-14/LOFTS0192/LOFTS0192.rawspec.0000.fil",
    "high_time": "/datax2/projects/LOFTS/2024-10-09/B1508+55/B1508+55.rawspec.0001.fil",
    "mid_res": "/datax2/projects/LOFTS/2024-10-09/B1508+55/B1508+55.rawspec.0002.fil",
}
# --- Broadened-signal training-case mix ------------------------------------
# The RFI-only mismatch receives additional weight because it was the hardest
# negative case in the initial Stage-3 evaluation. Probabilities sum to one.
BROADENED_CASE_PROBS = {
    "astro_match": 0.07,
    "astro_mismatch": 0.07,
    "rfi_only_mismatch": 0.12,
    "broadened_match": 0.28,
    "shape_decoy_match": 0.14,
    "broadened_mismatch_onesided": 0.16,
    "broadened_mismatch_different": 0.16,
}
assert abs(sum(BROADENED_CASE_PROBS.values()) - 1.0) < 1e-9
# Default "intrinsic" (pre-broadening) SNR range for broadened injections.
# This range is wider and higher than ``SIGNAL_SNR_RANGE`` (1.5--10): a
# BLISS-flagged
# broadened candidate is conditioned on having survived detection despite
# the peak_retention_factor() loss, so its underlying transmitter-power
# population is not the same as the unconditioned narrowband population.
DEFAULT_BROADENED_SNR_RANGE = (5.0, 30.0)
# default floor for the conditioned width/SNR sampler (see
# lorentzian_signals.sample_one_broadening_conditioned's docstring for the
# full justification). 3.0 is deliberately moderate -- strict enough to
# exclude the "physically invisible" tail, looser than the real search's
# own SNmin=10 (Johnson et al. 2023) so training still sees genuinely hard
# cases, not just trivially easy ones.
DEFAULT_MIN_EFFECTIVE_SNR = 3.0
# default curriculum width-cap schedule for
# --broadened_curriculum, in Hz. Kept simple (2 finite caps + uncapped)
# since conditioning already does the heavy lifting; this is purely an
# optimization-stability aid.
DEFAULT_CURRICULUM_WIDTHS_HZ = [10.0, 100.0, 20000.0]
# ==============================================================================
#  2. HELPER FUNCTIONS
# ==============================================================================


def format_time(seconds: float) -> str:
    """Formats a duration in seconds into a human-readable string (h/m/s)."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def setup_logger(log_file_path: Path) -> logging.Logger:
    """Configures a logger to write to both a file and stdout."""
    logger = logging.getLogger(str(log_file_path))
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def preprocess_np(arr: np.ndarray) -> np.ndarray:
    """Normalises a numpy array slice by subtracting the median and scaling by the standard deviation."""
    arr = arr.astype(np.float32)
    if arr.std() == 0:
        return arr
    arr = arr - np.median(arr, axis=1, keepdims=True)
    return arr / (arr.std() + 1e-6)


def add_astrosignal(frame: stg.Frame, mode: str, **kwargs) -> dict:
    """Inject one of the three baseline signal classes into a setigen frame."""
    config = CONFIGS[mode]
    noise_mean, noise_std = frame.get_noise_stats()
    if noise_std == 0:
        noise_std = 1.0
    signal_level = noise_mean + noise_std * np.random.uniform(*SIGNAL_SNR_RANGE)
    signal_type = kwargs.get("signal_type", np.random.choice(config["allowed_signals"]))
    signal_def = SIGNAL_DEFINITIONS[signal_type]
    f_start = kwargs.get(
        "f_start",
        frame.get_frequency(
            np.random.randint(int(0.2 * frame.fchans), int(0.8 * frame.fchans))
        ),
    )
    if signal_def["generator"] == "add_constant_signal":
        params = signal_def["params"]
        drift_rate = kwargs.get("drift_rate", params["drift_rate"]())
        width_mult = params["width_df_multiplier"]
        width_hz = frame.df * (width_mult() if callable(width_mult) else width_mult)
        frame.add_constant_signal(
            f_start=f_start,
            drift_rate=drift_rate,
            level=signal_level,
            width=width_hz,
            f_profile_type=params["f_profile_type"],
        )
        return {
            "signal_type": signal_type,
            "f_start": f_start,
            "drift_rate": drift_rate,
        }
    elif signal_def["generator"] == "add_signal":
        params = signal_def["params"]
        get_duration = params["duration"]
        duration_s = kwargs.get("duration", get_duration(frame))
        max_t_start = max(0, frame.tchans * frame.dt - duration_s)
        t_start_s = kwargs.get("t_start", np.random.uniform(0, max_t_start))
        f_width_percent_val = params["f_width_percent"]
        f_width_percent = (
            f_width_percent_val()
            if callable(f_width_percent_val)
            else f_width_percent_val
        )
        f_width_hz = frame.df * frame.fchans * f_width_percent

        def t_profile(t):
            return np.where(
                (t >= t_start_s) & (t < t_start_s + duration_s), signal_level, 0
            )

        frame.add_signal(
            stg.constant_path(f_start=f_start, drift_rate=0 * u.Hz / u.s),
            t_profile,
            stg.gaussian_f_profile(width=f_width_hz),
        )
        return {
            "signal_type": signal_type,
            "f_start": f_start,
            "duration": duration_s,
            "t_start": t_start_s,
        }


def _draw_broadening_sample(
    freq_ghz: float,
    p_mdwarf: float,
    broadened_snr_range: tuple,
    broadened_max_width_hz: float,
    broadened_min_effective_snr: float,
    use_conditioned_sampling: bool,
):
    """
    Draw a width and drift rate for a broadened case. The default sampler
    rejects combinations below the configured effective-S/N floor; the raw
    sampler is retained for the explicit unconditioned ablation.
    Returns a `lorentzian_signals.BroadeningDraw` (only `.width_hz` and
    `.drift_hz_per_s` are used by callers below; the full draw is returned
    so future callers can access orbital metadata if needed).
    """
    if use_conditioned_sampling:
        draw, _eff, _n_att, _accepted = ls.sample_one_broadening_conditioned(
            freq_GHz=freq_ghz,
            p_mdwarf=p_mdwarf,
            snr_range=broadened_snr_range,
            min_effective_snr=broadened_min_effective_snr,
            max_width_hz=broadened_max_width_hz,
        )
        return draw
    return ls.sample_one_broadening(
        freq_GHz=freq_ghz, p_mdwarf=p_mdwarf, max_width_hz=broadened_max_width_hz
    )


def add_broadened_signal(
    frame: stg.Frame,
    shape: str,
    delta_nu_sb_hz: float,
    drift_rate_hz_s: float,
    f_start_hz=None,
    snr_range: tuple = DEFAULT_BROADENED_SNR_RANGE,
) -> dict:
    """
    Inject a physically broadened signal or shape-matched decoy. Each station
    samples its own base S/N to represent an independent local noise and
    scintillation realisation.
    """
    noise_mean, noise_std = frame.get_noise_stats()
    if noise_std == 0:
        noise_std = 1.0
    base_snr = float(np.random.uniform(*snr_range))
    base_level = noise_mean + noise_std * base_snr
    if shape == "lorentzian":
        result = ls.inject_lorentzian_signal(
            frame, f_start_hz, drift_rate_hz_s, base_level, delta_nu_sb_hz
        )
    elif shape == "box":
        result = ls.inject_box_decoy_signal(
            frame, f_start_hz, drift_rate_hz_s, base_level, delta_nu_sb_hz
        )
    elif shape == "gaussian":
        result = ls.inject_gaussian_decoy_signal(
            frame, f_start_hz, drift_rate_hz_s, base_level, delta_nu_sb_hz
        )
    else:
        raise ValueError(
            f"Unknown shape: {shape!r}. Must be 'lorentzian', 'box', or 'gaussian'."
        )
    result["base_snr"] = base_snr
    return result


def get_real_slice(full_data: np.ndarray, tile_shape: tuple):
    """Extracts a random 2D slice from a larger data array."""
    if full_data.shape[0] < tile_shape[0] or full_data.shape[1] < tile_shape[1]:
        return None
    max_t_start = full_data.shape[0] - tile_shape[0]
    max_f_start = full_data.shape[1] - tile_shape[1]
    t_start = np.random.randint(0, max_t_start + 1)
    f_start = np.random.randint(0, max_f_start + 1)
    return full_data[
        t_start : t_start + tile_shape[0], f_start : f_start + tile_shape[1]
    ]


def load_background(filterbank_path) -> tuple:
    """Loads a filterbank file's header and memory-maps its data array."""
    print(f"INFO: Memory-mapping real background data from {filterbank_path}...")

    # 1. Get header and shape without loading the data into RAM
    wf = bl.Waterfall(str(filterbank_path), load_data=False)
    header = wf.header
    shape = wf.file_shape  # Usually (n_ints, n_ifs, n_chans)

    # 2. Determine bit depth to pick correct numpy datatype
    nbits = header.get("nbits", 32)
    dtype = np.float32 if nbits == 32 else (np.float16 if nbits == 16 else np.int8)

    # 3. Manually find the exact byte offset where the raw data starts
    with open(filterbank_path, "rb") as f:
        head_bytes = b""
        while b"HEADER_END" not in head_bytes:
            head_bytes += f.read(8192)
        offset = head_bytes.find(b"HEADER_END") + len(b"HEADER_END")

    # 4. Memory-map the file directly from disk! (Uses ~0 MB of RAM)
    real_data = np.memmap(
        str(filterbank_path), dtype=dtype, mode="r", offset=offset, shape=shape
    )

    # 5. Squeeze out the middle polarization dimension if it's 1
    real_data = real_data[:, 0, :] if real_data.ndim == 3 else real_data

    print("INFO: Data memory-mapped successfully (RAM completely bypassed).")
    return real_data, header


def generate_pair(
    mode: str,
    case: str,
    real_data: np.ndarray,
    header: dict,
    difficulty: str,
    real_data_b: np.ndarray = None,
    freq_ghz: float = 0.150,
    p_mdwarf: float = 0.75,
    broadened_snr_range: tuple = DEFAULT_BROADENED_SNR_RANGE,
    broadened_max_width_hz: float = 20000.0,
    broadened_min_effective_snr: float = DEFAULT_MIN_EFFECTIVE_SNR,
    broadened_use_conditioned_sampling: bool = True,
) -> tuple:
    """
    Generate one labelled two-station pair. Baseline cases contain matched or
    mismatched narrowband injections and noise-only negatives. Broadened cases
    contain matched Lorentzian or shape-decoy injections, a one-sided signal,
    or independent signals at both stations. Conditioned width/S/N sampling
    is enabled by default and can be disabled for the registered ablation.
    """
    tile_shape = CONFIGS[mode]["frame_shape"]
    df, dt = abs(header["foff"]) * 1e6, header["tsamp"]
    fch1, ascending = header["fch1"], header["foff"] > 0
    data_b_source = real_data_b if real_data_b is not None else real_data
    slice1 = get_real_slice(real_data, tile_shape)
    if difficulty == "simplified":
        slice2 = copy.deepcopy(slice1)
    else:  # 'realistic' or 'broadened'
        slice2 = get_real_slice(data_b_source, tile_shape)
    if slice1 is None or slice2 is None:
        return None, None, 0
    frame1 = stg.Frame.from_data(
        df=df, dt=dt, fch1=fch1, ascending=ascending, data=slice1
    )
    frame2 = stg.Frame.from_data(
        df=df, dt=dt, fch1=fch1, ascending=ascending, data=slice2
    )
    # Baseline narrowband and noise-only cases
    if case == "astro_match":
        params = add_astrosignal(frame1, mode)
        add_astrosignal(frame2, mode, **params)
        label = 1
    elif case == "astro_mismatch":
        if np.random.rand() > 0.5:
            add_astrosignal(frame1, mode)
        else:
            add_astrosignal(frame1, mode)
            add_astrosignal(frame2, mode)
        label = 0
    elif case == "rfi_only_mismatch":
        label = 0
    elif case == "empty_sky_match":
        label = 1 if difficulty == "simplified" else 0
    # Broadened-signal cases
    elif case == "broadened_match":
        draw = _draw_broadening_sample(
            freq_ghz,
            p_mdwarf,
            broadened_snr_range,
            broadened_max_width_hz,
            broadened_min_effective_snr,
            broadened_use_conditioned_sampling,
        )
        r1 = add_broadened_signal(
            frame1,
            "lorentzian",
            draw.width_hz,
            draw.drift_hz_per_s,
            f_start_hz=None,
            snr_range=broadened_snr_range,
        )
        add_broadened_signal(
            frame2,
            "lorentzian",
            draw.width_hz,
            draw.drift_hz_per_s,
            f_start_hz=r1["f_start_hz"],
            snr_range=broadened_snr_range,
        )
        label = 1
    elif case == "shape_decoy_match":
        shape = np.random.choice(["box", "gaussian"])
        draw = _draw_broadening_sample(
            freq_ghz,
            p_mdwarf,
            broadened_snr_range,
            broadened_max_width_hz,
            broadened_min_effective_snr,
            broadened_use_conditioned_sampling,
        )
        r1 = add_broadened_signal(
            frame1,
            shape,
            draw.width_hz,
            draw.drift_hz_per_s,
            f_start_hz=None,
            snr_range=broadened_snr_range,
        )
        add_broadened_signal(
            frame2,
            shape,
            draw.width_hz,
            draw.drift_hz_per_s,
            f_start_hz=r1["f_start_hz"],
            snr_range=broadened_snr_range,
        )
        label = 1
    elif case == "broadened_mismatch_onesided":
        draw = _draw_broadening_sample(
            freq_ghz,
            p_mdwarf,
            broadened_snr_range,
            broadened_max_width_hz,
            broadened_min_effective_snr,
            broadened_use_conditioned_sampling,
        )
        add_broadened_signal(
            frame1,
            "lorentzian",
            draw.width_hz,
            draw.drift_hz_per_s,
            f_start_hz=None,
            snr_range=broadened_snr_range,
        )
        label = 0
    elif case == "broadened_mismatch_different":
        draw_a = _draw_broadening_sample(
            freq_ghz,
            p_mdwarf,
            broadened_snr_range,
            broadened_max_width_hz,
            broadened_min_effective_snr,
            broadened_use_conditioned_sampling,
        )
        draw_b = _draw_broadening_sample(
            freq_ghz,
            p_mdwarf,
            broadened_snr_range,
            broadened_max_width_hz,
            broadened_min_effective_snr,
            broadened_use_conditioned_sampling,
        )
        add_broadened_signal(
            frame1,
            "lorentzian",
            draw_a.width_hz,
            draw_a.drift_hz_per_s,
            f_start_hz=None,
            snr_range=broadened_snr_range,
        )
        add_broadened_signal(
            frame2,
            "lorentzian",
            draw_b.width_hz,
            draw_b.drift_hz_per_s,
            f_start_hz=None,
            snr_range=broadened_snr_range,
        )
        label = 0
    else:
        raise ValueError(f"Unknown case: {case}")
    return frame1, frame2, label


# ==============================================================================
#  3. MODEL AND DATASET CLASSES
# ==============================================================================


def GN(channels: int, groups: int = 8) -> nn.Module:
    """Group Normalization layer, ensuring groups are not more than channels."""
    return nn.GroupNorm(min(groups, channels), channels)


class UNet(nn.Module):
    """U-Net reconstruction model with a contrastive projection head."""

    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, latent_dim: int = 512
    ):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(128, 128)
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, latent_dim),
        )

    @staticmethod
    def _conv_block(ch_in: int, ch_out: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            GN(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        z = self.projection_head(b)
        z_normalized = F.normalize(z, p=2, dim=1)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        recon = torch.sigmoid(self.out_conv(d1))
        return recon, z_normalized


class CombinedLoss(nn.Module):
    """Weighted reconstruction loss plus margin-based contrastive loss."""

    def __init__(
        self, alpha: float, margin: float, weight_factor: int, percentile: float = 99.5
    ):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.weight_factor = weight_factor
        self.percentile = percentile

    def _weighted_mse(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = target.shape
        threshold = torch.quantile(
            target.view(B, C, -1), self.percentile / 100.0, dim=2, keepdim=True
        ).view(B, C, 1, 1)
        weights = 1.0 + (self.weight_factor - 1.0) * (target > threshold).float()
        return (weights * (output - target).pow(2)).mean()

    def _contrastive_loss(
        self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        distance = F.pairwise_distance(z1, z2)
        loss_positive = y * distance.pow(2)
        loss_negative = (1 - y) * F.relu(self.margin - distance).pow(2)
        return (loss_positive + loss_negative).mean()

    def forward(self, rec1, img1, rec2, img2, z1, z2, y) -> tuple:
        recon_loss = self._weighted_mse(rec1, img1) + self._weighted_mse(rec2, img2)
        contrastive_loss = self._contrastive_loss(z1, z2, y)
        total_loss = recon_loss + self.alpha * contrastive_loss
        return total_loss, recon_loss, contrastive_loss


class SpectrogramPairDataset(Dataset):
    """Generate spectrogram pairs on demand for one curriculum stage."""

    def __init__(
        self,
        num_samples: int,
        mode: str,
        filterbank_path: str,
        difficulty: str,
        station_b_filterbank_path: str = None,
        freq_ghz: float = 0.150,
        p_mdwarf: float = 0.75,
        broadened_snr_range: tuple = DEFAULT_BROADENED_SNR_RANGE,
        broadened_max_width_hz: float = 20000.0,
        broadened_min_effective_snr: float = DEFAULT_MIN_EFFECTIVE_SNR,
        broadened_use_conditioned_sampling: bool = True,
    ):
        self.num_samples = num_samples
        self.mode = mode
        self.difficulty = difficulty
        self.freq_ghz = freq_ghz
        self.p_mdwarf = p_mdwarf
        self.broadened_snr_range = broadened_snr_range
        self.broadened_max_width_hz = broadened_max_width_hz
        self.broadened_min_effective_snr = broadened_min_effective_snr
        self.broadened_use_conditioned_sampling = broadened_use_conditioned_sampling
        if self.difficulty == "simplified":
            self.case_probs = {
                "astro_match": 0.45,
                "astro_mismatch": 0.45,
                "empty_sky_match": 0.1,
            }
        elif self.difficulty == "realistic":
            self.case_probs = {
                "astro_match": 0.4,
                "astro_mismatch": 0.4,
                "rfi_only_mismatch": 0.2,
            }
        elif self.difficulty == "broadened":
            self.case_probs = dict(BROADENED_CASE_PROBS)
        else:
            raise ValueError(
                f"Unknown difficulty: {difficulty!r}. Must be 'simplified', "
                f"'realistic', or 'broadened'."
            )
        self.cases = list(self.case_probs.keys())
        self.probs = list(self.case_probs.values())
        self.real_data, self.header = load_background(filterbank_path)
        self.real_data_b = None
        if station_b_filterbank_path:
            self.real_data_b, _ = load_background(station_b_filterbank_path)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        while True:
            case = np.random.choice(self.cases, p=self.probs)
            frame1, frame2, label_val = generate_pair(
                self.mode,
                case,
                self.real_data,
                self.header,
                self.difficulty,
                real_data_b=self.real_data_b,
                freq_ghz=self.freq_ghz,
                p_mdwarf=self.p_mdwarf,
                broadened_snr_range=self.broadened_snr_range,
                broadened_max_width_hz=self.broadened_max_width_hz,
                broadened_min_effective_snr=self.broadened_min_effective_snr,
                broadened_use_conditioned_sampling=self.broadened_use_conditioned_sampling,
            )
            # If we successfully generated a pair, break out of the retry loop
            if frame1 is not None and frame2 is not None:
                break

        img1 = torch.from_numpy(preprocess_np(frame1.get_data())[None, ...])
        img2 = torch.from_numpy(preprocess_np(frame2.get_data())[None, ...])
        label = torch.tensor(label_val, dtype=torch.float32)
        return img1, img2, label


# ==============================================================================
#  4. TRAINING AND EVALUATION LOOPS
# ==============================================================================


def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    scaler,
    device,
    logger,
    epoch,
    total_epochs,
    freeze_head=False,
):
    """Runs a single training epoch."""
    model.train()
    if freeze_head:
        for param in model.projection_head.parameters():
            param.requires_grad = False
    running_loss = {"total": 0.0, "recon": 0.0, "contrast": 0.0}
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for imgs1, imgs2, labels in pbar:
        imgs = torch.cat([imgs1, imgs2]).to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            recs, z_vecs = model(imgs)
            rec1, rec2 = recs.chunk(2)
            z1, z2 = z_vecs.chunk(2)
            total, recon, contrast = loss_fn(
                rec1, imgs1.to(device), rec2, imgs2.to(device), z1, z2, labels
            )
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss["total"] += total.item()
        running_loss["recon"] += recon.item()
        running_loss["contrast"] += contrast.item()
        pbar.set_postfix(loss=total.item())
    if freeze_head:
        for param in model.projection_head.parameters():
            param.requires_grad = True
    n = len(dataloader)
    avg_total = running_loss["total"] / n
    avg_recon = running_loss["recon"] / n
    avg_contrast = running_loss["contrast"] / n
    logger.info(
        f"Epoch {epoch}: Total={avg_total:.4f} | Recon={avg_recon:.4f} | Contrast={avg_contrast:.4f}"
    )
    return avg_total, avg_recon, avg_contrast


def collect_distances(model, dataloader, device, n_samples):
    """Collect latent distances for matched and mismatched pairs."""
    model.eval()
    match_distances, mismatch_distances = [], []
    count = 0
    with torch.no_grad():
        for img1, img2, label in tqdm(
            dataloader,
            desc="Collecting distances",
            total=min(len(dataloader), n_samples // dataloader.batch_size),
        ):
            _, z1 = model(img1.to(device))
            _, z2 = model(img2.to(device))
            d = F.pairwise_distance(z1, z2)
            for i in range(label.shape[0]):
                if label[i].item() == 1:
                    match_distances.append(d[i].item())
                else:
                    mismatch_distances.append(d[i].item())
                count += 1
                if count >= n_samples:
                    return np.array(match_distances), np.array(mismatch_distances)
    return np.array(match_distances), np.array(mismatch_distances)


def compute_classification_metrics(
    model, dataloader, device, margin: float, max_samples: int = 300
) -> dict:
    """Compute classification metrics at the active stage's distance threshold."""
    model.eval()
    tp = fp = fn = tn = 0
    n_seen = 0
    with torch.no_grad():
        for img1, img2, label in dataloader:
            if n_seen >= max_samples:
                break
            _, z1 = model(img1.to(device))
            _, z2 = model(img2.to(device))
            d = F.pairwise_distance(z1, z2).cpu().numpy()
            y = label.numpy().astype(int)
            pred_match = (d < margin).astype(int)
            for yi, pi in zip(y, pred_match):
                if yi == 1 and pi == 1:
                    tp += 1
                elif yi == 1 and pi == 0:
                    fn += 1
                elif yi == 0 and pi == 1:
                    fp += 1
                else:
                    tn += 1
                n_seen += 1
                if n_seen >= max_samples:
                    break
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / max(n_seen, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": n_seen,
        "margin_used": float(margin),
    }


def plot_f1_curve(
    metrics_history: list,
    stage_boundaries: list,
    save_path: Path,
    broadened_margin: Optional[float] = None,
):
    """Plot precision, recall, and F1 against training epoch. If
    `broadened_margin` differs from the Stage-1/2 margin, an annotation
    marks the threshold change at the LAST stage boundary, since
    `compute_classification_metrics` is now evaluated at a DIFFERENT
    threshold once Stage 3 begins, the plot marks that change explicitly."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = list(range(1, len(metrics_history) + 1))
    ax.plot(epochs, [m["f1"] for m in metrics_history], label="F1", color="white", lw=2)
    ax.plot(
        epochs,
        [m["precision"] for m in metrics_history],
        label="Precision",
        color="cyan",
        ls="--",
        alpha=0.8,
    )
    ax.plot(
        epochs,
        [m["recall"] for m in metrics_history],
        label="Recall",
        color="orange",
        ls="--",
        alpha=0.8,
    )
    for i, boundary in enumerate(stage_boundaries):
        ax.axvline(
            x=boundary - 0.5,
            color="yellow",
            ls=":",
            label="Stage/curriculum boundary" if i == 0 else None,
        )
    if broadened_margin is not None and stage_boundaries:
        last_boundary = (
            stage_boundaries[-1] if len(stage_boundaries) <= 2 else stage_boundaries[1]
        )
        ax.text(
            last_boundary,
            0.03,
            f"Stage-3 threshold={broadened_margin:.2f}",
            color="yellow",
            fontsize=9,
            rotation=90,
            va="bottom",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (at in-effect margin threshold)")
    ax.set_title("Classification Performance vs. Training Epoch")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_loss_curves(
    all_losses: dict, stage_boundaries: list, mode: str, save_path: Path
):
    """Plot all loss components and mark curriculum-stage boundaries."""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(all_losses["total"], label="Total")
    ax1.plot(all_losses["recon"], label="Recon", ls="--")
    ax1.set_title(f"Training Loss ({mode.upper()})")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(all_losses["contrast"], label="Contrastive", color="lime", ls=":")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)
    for i, boundary in enumerate(stage_boundaries):
        ax1.axvline(
            x=boundary - 0.5,
            color="yellow",
            ls=":",
            label="Stage/curriculum boundary" if i == 0 else None,
        )
        ax2.axvline(x=boundary - 0.5, color="yellow", ls=":")
    if stage_boundaries:
        ax1.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def collect_distances_by_case(
    model,
    mode: str,
    real_data: np.ndarray,
    header: dict,
    device,
    real_data_b: np.ndarray = None,
    freq_ghz: float = 0.150,
    p_mdwarf: float = 0.75,
    broadened_snr_range: tuple = DEFAULT_BROADENED_SNR_RANGE,
    broadened_max_width_hz: float = 20000.0,
    broadened_min_effective_snr: float = DEFAULT_MIN_EFFECTIVE_SNR,
    broadened_use_conditioned_sampling: bool = True,
    n_per_case: int = 300,
    batch_size: int = 32,
) -> tuple:
    """
    Generate fresh pairs for every case and report latent-distance separation
    and AUC-ROC. By default, evaluation uses the same conditioned population as
    training; set ``broadened_use_conditioned_sampling=False`` for the
    registered unconditioned-population ablation.
    """
    model.eval()
    case_probs = BROADENED_CASE_PROBS
    match_cases = [c for c in case_probs if c.endswith("_match")]
    mismatch_cases = [c for c in case_probs if c not in match_cases]

    def _collect_for_case(case_name, n):
        distances = []
        batch1, batch2 = [], []
        with torch.no_grad():
            for _ in range(n):
                f1, f2, _ = generate_pair(
                    mode,
                    case_name,
                    real_data,
                    header,
                    "broadened",
                    real_data_b=real_data_b,
                    freq_ghz=freq_ghz,
                    p_mdwarf=p_mdwarf,
                    broadened_snr_range=broadened_snr_range,
                    broadened_max_width_hz=broadened_max_width_hz,
                    broadened_min_effective_snr=broadened_min_effective_snr,
                    broadened_use_conditioned_sampling=broadened_use_conditioned_sampling,
                )
                if f1 is None or f2 is None:
                    continue
                img1 = preprocess_np(f1.get_data())[None, None, ...]
                img2 = preprocess_np(f2.get_data())[None, None, ...]
                batch1.append(img1)
                batch2.append(img2)
                if len(batch1) == batch_size:
                    t1 = torch.from_numpy(np.concatenate(batch1)).to(device)
                    t2 = torch.from_numpy(np.concatenate(batch2)).to(device)
                    _, z1 = model(t1)
                    _, z2 = model(t2)
                    d = F.pairwise_distance(z1, z2).cpu().numpy()
                    distances.extend(d.tolist())
                    batch1, batch2 = [], []
            if batch1:
                t1 = torch.from_numpy(np.concatenate(batch1)).to(device)
                t2 = torch.from_numpy(np.concatenate(batch2)).to(device)
                _, z1 = model(t1)
                _, z2 = model(t2)
                d = F.pairwise_distance(z1, z2).cpu().numpy()
                distances.extend(d.tolist())
        return np.array(distances)

    raw_distances = {}
    for case_name in tqdm(case_probs, desc="Stratified evaluation by case"):
        raw_distances[case_name] = _collect_for_case(case_name, n_per_case)
    match_d = (
        np.concatenate([raw_distances[c] for c in match_cases])
        if match_cases
        else np.array([])
    )
    mismatch_d = (
        np.concatenate([raw_distances[c] for c in mismatch_cases])
        if mismatch_cases
        else np.array([])
    )
    summary = {
        "per_case_mean_distance": {
            c: (float(v.mean()) if len(v) else float("nan"))
            for c, v in raw_distances.items()
        },
        "per_case_std_distance": {
            c: (float(v.std()) if len(v) else float("nan"))
            for c, v in raw_distances.items()
        },
        "per_case_n": {c: int(len(v)) for c, v in raw_distances.items()},
        "overall_match_mean": float(match_d.mean()) if len(match_d) else float("nan"),
        "overall_mismatch_mean": (
            float(mismatch_d.mean()) if len(mismatch_d) else float("nan")
        ),
    }
    if len(match_d) and len(mismatch_d):
        summary["separation_gap"] = (
            summary["overall_mismatch_mean"] - summary["overall_match_mean"]
        )
        if roc_auc_score is not None:
            scores = np.concatenate([-match_d, -mismatch_d])
            labels = np.concatenate([np.ones(len(match_d)), np.zeros(len(mismatch_d))])
            try:
                summary["auc_roc"] = float(roc_auc_score(labels, scores))
            except ValueError:
                summary["auc_roc"] = float("nan")
        else:
            summary["auc_roc"] = None
            print("NOTE: scikit-learn not installed; AUC-ROC skipped.")
    return summary, raw_distances


def plot_stratified_distances(raw_distances: dict, margin: float, save_path: Path):
    """Bar chart (mean +/- std) of latent distance per broadened-signal
    case, with the training margin marked."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6))
    cases = list(raw_distances.keys())
    means = [
        raw_distances[c].mean() if len(raw_distances[c]) else np.nan for c in cases
    ]
    stds = [raw_distances[c].std() if len(raw_distances[c]) else np.nan for c in cases]
    colors = ["lightgreen" if c.endswith("_match") else "lightcoral" for c in cases]
    ax.bar(range(len(cases)), means, yerr=stds, color=colors, alpha=0.85, capsize=5)
    ax.axhline(margin, color="yellow", ls="--", lw=1.5, label=f"Margin ({margin})")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cases], fontsize=9)
    ax.set_ylabel("Latent Distance (mean +/- std)")
    ax.set_title(
        "Per-case latent distance for broadened-signal pairs\n"
        "(green = genuine match cases, red = genuine mismatch cases)"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ==============================================================================
#  5. MAIN EXECUTION
# ==============================================================================


def _run_background_qa(arr: np.ndarray, label: str, logger: logging.Logger) -> dict:
    """Run a non-blocking background-quality screen and log its verdict."""
    try:
        result = flag_pulsar_like_background(arr, label=label)
        logger.info(f"[Background QA] {result['message']}")
        if result["verdict"] != "SAFE":
            logger.warning(
                f"[Background QA] verdict={result['verdict']} for '{label}'. "
                f"Training proceeds regardless (non-blocking); see "
                f"check_background_quality.py for remediation guidance."
            )
        return result
    except Exception as e:  # never let a QA check crash a real training run
        logger.warning(f"[Background QA] check failed for '{label}': {e}")
        return {
            "label": label,
            "verdict": "CHECK_FAILED",
            "stats": {},
            "message": str(e),
        }


def _split_epochs(total_epochs: int, n_stages: int) -> list:
    """Distributes `total_epochs` across `n_stages` sub-stages as evenly as
    possible, with any remainder given to the LAST sub-stage (the
    full/uncapped-width stage, which represents the true deployment target
    and so reasonably deserves relatively more training time)."""
    base = total_epochs // n_stages
    counts = [base] * n_stages
    counts[-1] += total_epochs - base * n_stages
    return counts


def main(args):
    overall_start_time = time.time()
    SAMPLES_PER_EPOCH = 1024
    BATCH_SIZE = 32
    EPOCHS_STAGE1, LR_STAGE1 = 30, 5e-4
    EPOCHS_STAGE2, LR_STAGE2 = 35, 1e-4
    WARMUP_EPOCHS = 5
    EVAL_SAMPLES = 20000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(os.cpu_count(), args.num_workers)
    print("=" * 70)
    print(f"Starting training run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {device} with {num_workers} workers.")
    print(f"Mode to run: {args.mode or 'all'}")
    print(f"Include broadened-signal Stage 3: {args.include_broadened}")
    if args.include_broadened:
        print(
            f"  Conditioned width/SNR sampling: {not args.broadened_disable_conditioning}"
        )
        print(f"  Curriculum sub-staging: {args.broadened_curriculum}")
    print("=" * 70)
    if args.run_id:
        session_id = args.run_id
    else:
        hyperparam_str = (
            f"A{int(args.alpha*100)}_M{int(args.margin*100)}_W{args.weight_factor}"
        )
        training_runs_dir = Path("training_runs")
        training_runs_dir.mkdir(exist_ok=True)
        run_numbers = [
            int(re.match(r"run_(\d+)", d.name).group(1))
            for d in training_runs_dir.iterdir()
            if d.is_dir() and re.match(r"run_(\d+)", d.name)
        ]
        next_run_number = max(run_numbers) + 1 if run_numbers else 1
        session_id = f"run_{next_run_number}_{hyperparam_str}"
    print(f"INFO: Run ID set to: {session_id}")
    modes_to_run = [args.mode] if args.mode else CONFIGS.keys()
    for mode in modes_to_run:
        mode_start_time = time.time()
        print(f"\n{'='*25} STARTING MODE: {mode.upper()} {'='*25}")
        fb_path = Path(args.filterbank or FILTERBANK_PATHS.get(mode, ""))
        if not fb_path.exists():
            print(f"WARNING: Filterbank for '{mode}' not found at {fb_path}. Skipping.")
            continue
        run_dir = Path("training_runs") / f"{session_id}_{mode}"
        plot_dir = run_dir / "plots"
        run_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(exist_ok=True)
        logger = setup_logger(run_dir / "training_log.log")
        logger.info(f"Using real background from: {fb_path}")
        model = UNet().to(device)
        if torch.__version__.startswith("2."):
            try:
                model = torch.compile(model)
                logger.info("PyTorch 2.0+ detected. Model compiled successfully.")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed: {e}. Falling back to eager mode."
                )
        loss_fn = CombinedLoss(
            alpha=args.alpha, margin=args.margin, weight_factor=args.weight_factor
        )
        scaler = GradScaler()
        all_losses = {"total": [], "recon": [], "contrast": []}
        all_metrics = []
        stage_boundaries = []
        bg_qa_results = []
        # --- Stage 1: Simplified Backgrounds ---
        stage1_start_time = time.time()
        logger.info("--- Starting Stage 1: Simplified Backgrounds ---")
        dataset1 = SpectrogramPairDataset(
            SAMPLES_PER_EPOCH, mode, fb_path, "simplified"
        )
        bg_qa_results.append(
            _run_background_qa(
                dataset1.real_data, f"{mode}:station_A:{fb_path.name}", logger
            )
        )
        dl1 = DataLoader(
            dataset1,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        optimizer = optim.Adam(model.parameters(), lr=LR_STAGE1)
        for ep in range(1, EPOCHS_STAGE1 + 1):
            avg_tot, avg_rec, avg_con = train_epoch(
                model,
                dl1,
                optimizer,
                loss_fn,
                scaler,
                device,
                logger,
                ep,
                EPOCHS_STAGE1,
            )
            all_losses["total"].append(avg_tot)
            all_losses["recon"].append(avg_rec)
            all_losses["contrast"].append(avg_con)
            all_metrics.append(
                compute_classification_metrics(model, dl1, device, args.margin)
            )
        logger.info(
            f"--- Stage 1 complete. Time: {format_time(time.time() - stage1_start_time)} ---"
        )
        # --- Stage 2: Realistic Backgrounds ---
        stage2_start_time = time.time()
        logger.info("--- Starting Stage 2: Realistic Backgrounds ---")
        dataset2 = SpectrogramPairDataset(SAMPLES_PER_EPOCH, mode, fb_path, "realistic")
        dl2 = DataLoader(
            dataset2,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        optimizer = optim.Adam(model.parameters(), lr=LR_STAGE2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        stage_boundaries.append(EPOCHS_STAGE1 + 1)
        for ep in range(1, EPOCHS_STAGE2 + 1):
            logger.info(
                f"Starting Stage 2 Epoch {ep}/{EPOCHS_STAGE2} with LR={scheduler.get_last_lr()[0]:.2e}"
            )
            avg_tot, avg_rec, avg_con = train_epoch(
                model,
                dl2,
                optimizer,
                loss_fn,
                scaler,
                device,
                logger,
                ep + EPOCHS_STAGE1,
                EPOCHS_STAGE1 + EPOCHS_STAGE2,
                freeze_head=(ep <= WARMUP_EPOCHS),
            )
            all_losses["total"].append(avg_tot)
            all_losses["recon"].append(avg_rec)
            all_losses["contrast"].append(avg_con)
            all_metrics.append(
                compute_classification_metrics(model, dl2, device, args.margin)
            )
            scheduler.step()
        logger.info(
            f"--- Stage 2 complete. Time: {format_time(time.time() - stage2_start_time)} ---"
        )
        final_model_path = run_dir / f"model_{mode}.pth"
        state_dict_to_save = (
            model._orig_mod.state_dict()
            if hasattr(model, "_orig_mod")
            else model.state_dict()
        )
        torch.save(state_dict_to_save, final_model_path)
        logger.info(f"Model for mode '{mode}' saved to {final_model_path}")
        # --- Stage 1+2 Evaluation and Reporting ---
        logger.info("--- Starting Stage 1+2 Evaluation ---")
        eval_dataloader = DataLoader(
            dataset2,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        match_d, mismatch_d = collect_distances(
            model, eval_dataloader, device, n_samples=EVAL_SAMPLES
        )
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        plt.style.use("dark_background")
        ax_hist.hist(mismatch_d, bins=50, alpha=0.7, label="Mismatch", density=True)
        ax_hist.hist(match_d, bins=50, alpha=0.7, label="Match", density=True)
        ax_hist.axvline(
            x=args.margin, color="red", ls="--", label=f"Margin ({args.margin})"
        )
        ax_hist.set_title(f"Latent Distance Distribution ({mode.upper()})")
        ax_hist.set_xlabel("Euclidean Distance")
        ax_hist.legend()
        ax_hist.grid(alpha=0.3)
        plt.tight_layout()
        fig_hist.savefig(plot_dir / "distance_histogram.png", dpi=120)
        plt.close(fig_hist)
        report_path = run_dir / "report.txt"
        with report_path.open("w") as f:
            f.write(f"=== Model Training Report: {mode.upper()} ===\n")
            f.write(f"Run ID: {session_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Background File: {fb_path.name}\n\n")
            f.write("--- Background Data Quality ---\n")
            for r in bg_qa_results:
                f.write(f"  [{r['verdict']}] {r['label']}: {r['message']}\n")
            if len(bg_qa_results) > 1:
                f.write("\nRanked summary (quietest first):\n")
                f.write(summarize_background_quality(bg_qa_results) + "\n")
            f.write("\n")
            f.write("--- Hyperparameters ---\n")
            f.write(f"Contrastive Alpha: {args.alpha}\n")
            f.write(f"Contrastive Margin (Stage 1/2): {args.margin}\n")
            f.write(f"Reconstruction Weight Factor: {args.weight_factor}\n\n")
            f.write(
                "--- Stage 1+2 Evaluation Metrics (original narrowband pipeline) ---\n"
            )
            separation_gap = mismatch_d.mean() - match_d.mean()
            f.write(
                f"Match distances    : mu={match_d.mean():.4f}, sigma={match_d.std():.4f}\n"
            )
            f.write(
                f"Mismatch distances : mu={mismatch_d.mean():.4f}, sigma={mismatch_d.std():.4f}\n"
            )
            f.write(f"Separation Gap     : {separation_gap:.4f}\n")
            final_stage12_metrics = all_metrics[-1]
            f.write(
                f"Final-epoch Precision/Recall/F1 (at margin={args.margin}): "
                f"P={final_stage12_metrics['precision']:.4f}, "
                f"R={final_stage12_metrics['recall']:.4f}, "
                f"F1={final_stage12_metrics['f1']:.4f}\n"
            )
            # ---------------- Stage 3 (revised) -------------------
            broadened_margin = args.margin
            if args.include_broadened:
                broadened_margin = (
                    args.broadened_margin
                    if args.broadened_margin is not None
                    else args.margin
                )
                use_conditioned = not args.broadened_disable_conditioning
                loss_fn_stage3 = CombinedLoss(
                    alpha=args.alpha,
                    margin=broadened_margin,
                    weight_factor=args.weight_factor,
                )
                logger.info("--- Starting Stage 3: Broadened-Signal Fine-Tuning ---")
                logger.info(
                    f"    Stage-3 training margin: {broadened_margin} "
                    f"(Stage 1/2 margin was {args.margin})"
                )
                logger.info(f"    Conditioned width/SNR sampling: {use_conditioned}")
                if args.broadened_curriculum:
                    caps = [
                        float(w) for w in args.broadened_curriculum_widths_hz.split(",")
                    ]
                else:
                    caps = [20000.0]
                epoch_counts = _split_epochs(args.broadened_epochs, len(caps))
                logger.info(
                    f"    Curriculum width caps (Hz): {caps}  |  epochs per sub-stage: {epoch_counts}"
                )
                stage3_start_time = time.time()
                epoch_offset = EPOCHS_STAGE1 + EPOCHS_STAGE2
                total_stage3_epochs = args.broadened_epochs
                dataset3 = None  # keep a reference to the LAST sub-stage's dataset for final eval
                for sub_idx, (cap, n_ep) in enumerate(zip(caps, epoch_counts)):
                    if n_ep <= 0:
                        continue
                    stage_boundaries.append(epoch_offset + 1)
                    logger.info(
                        f"    -- Sub-stage {sub_idx+1}/{len(caps)}: width cap={cap} Hz, "
                        f"{n_ep} epochs --"
                    )
                    dataset3 = SpectrogramPairDataset(
                        SAMPLES_PER_EPOCH,
                        mode,
                        fb_path,
                        "broadened",
                        station_b_filterbank_path=args.station_b_filterbank,
                        freq_ghz=args.freq_ghz,
                        p_mdwarf=args.p_mdwarf,
                        broadened_snr_range=(
                            args.broadened_snr_min,
                            args.broadened_snr_max,
                        ),
                        broadened_max_width_hz=cap,
                        broadened_min_effective_snr=args.broadened_min_effective_snr,
                        broadened_use_conditioned_sampling=use_conditioned,
                    )
                    if sub_idx == 0 and dataset3.real_data_b is not None:
                        bg_qa_results.append(
                            _run_background_qa(
                                dataset3.real_data_b,
                                f"{mode}:station_B:{Path(args.station_b_filterbank).name}",
                                logger,
                            )
                        )
                    dl3 = DataLoader(
                        dataset3,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=(num_workers > 0),
                    )
                    optimizer = optim.Adam(model.parameters(), lr=args.broadened_lr)
                    scheduler3 = optim.lr_scheduler.StepLR(
                        optimizer, step_size=10, gamma=0.5
                    )
                    for ep in range(1, n_ep + 1):
                        epoch_offset += 1
                        logger.info(
                            f"Starting Stage 3 Epoch {ep}/{n_ep} (sub-stage {sub_idx+1}) "
                            f"with LR={scheduler3.get_last_lr()[0]:.2e}"
                        )
                        avg_tot, avg_rec, avg_con = train_epoch(
                            model,
                            dl3,
                            optimizer,
                            loss_fn_stage3,
                            scaler,
                            device,
                            logger,
                            epoch_offset,
                            EPOCHS_STAGE1 + EPOCHS_STAGE2 + total_stage3_epochs,
                        )
                        all_losses["total"].append(avg_tot)
                        all_losses["recon"].append(avg_rec)
                        all_losses["contrast"].append(avg_con)
                        all_metrics.append(
                            compute_classification_metrics(
                                model, dl3, device, broadened_margin
                            )
                        )
                        scheduler3.step()
                logger.info(
                    f"--- Stage 3 complete. Time: {format_time(time.time() - stage3_start_time)} ---"
                )
                stage3_model_path = run_dir / f"model_{mode}_broadened.pth"
                state_dict_to_save = (
                    model._orig_mod.state_dict()
                    if hasattr(model, "_orig_mod")
                    else model.state_dict()
                )
                torch.save(state_dict_to_save, stage3_model_path)
                logger.info(
                    f"Stage-3 (broadened-aware) model saved to {stage3_model_path}"
                )
                logger.info(
                    "--- Starting Stage 3 Stratified Evaluation (uncapped, honest report) ---"
                )
                summary, raw_distances = collect_distances_by_case(
                    model,
                    mode,
                    dataset3.real_data,
                    dataset3.header,
                    device,
                    real_data_b=dataset3.real_data_b,
                    freq_ghz=args.freq_ghz,
                    p_mdwarf=args.p_mdwarf,
                    broadened_snr_range=(
                        args.broadened_snr_min,
                        args.broadened_snr_max,
                    ),
                    broadened_max_width_hz=20000.0,  # uncapped final evaluation
                    broadened_min_effective_snr=args.broadened_min_effective_snr,
                    broadened_use_conditioned_sampling=use_conditioned,
                    n_per_case=300,
                    batch_size=BATCH_SIZE,
                )
                plot_stratified_distances(
                    raw_distances,
                    broadened_margin,
                    plot_dir / "broadened_case_distances.png",
                )
                # automatic margin recalibration, reusing the
                # distances just collected -- zero extra inference cost.
                best = tu.best_f1_threshold(
                    raw_distances,
                    max_threshold=max(broadened_margin, 1.0),
                    n_candidates=60,
                )
                recommended_margin_path = run_dir / "recommended_margin_stage3.json"
                import json as _json

                with open(recommended_margin_path, "w") as _jf:
                    _json.dump(
                        json_safe({k: v for k, v in best.items() if k != "sweep"}),
                        _jf,
                        indent=2,
                        allow_nan=False,
                    )
                logger.info(
                    f"[Auto margin recalibration] best-F1 threshold={best['threshold']:.3f} "
                    f"(P={best['precision']:.3f}, R={best['recall']:.3f}, F1={best['f1']:.3f}) "
                    f"-- saved to {recommended_margin_path}"
                )
                f.write(
                    "\n--- Stage 3 Evaluation Metrics (broadened-signal pipeline) ---\n"
                )
                f.write(f"Stage-3 training margin           : {broadened_margin}\n")
                f.write(
                    f"Conditioned width/SNR sampling     : {use_conditioned} "
                    f"(min_effective_snr={args.broadened_min_effective_snr})\n"
                )
                f.write(f"Curriculum width caps (Hz)         : {caps}\n")
                f.write(
                    f"Broadened SNR range (intrinsic)    : [{args.broadened_snr_min}, {args.broadened_snr_max}]\n"
                )
                f.write(f"p_mdwarf                           : {args.p_mdwarf}\n")
                f.write(f"Observing frequency (GHz)          : {args.freq_ghz}\n\n")
                for case_name in BROADENED_CASE_PROBS:
                    mean_d = summary["per_case_mean_distance"][case_name]
                    std_d = summary["per_case_std_distance"][case_name]
                    n = summary["per_case_n"][case_name]
                    f.write(
                        f"  {case_name:<32} n={n:>4}  mean_dist={mean_d:.4f}  std={std_d:.4f}\n"
                    )
                f.write(
                    f"\nOverall match mean    : {summary['overall_match_mean']:.4f}\n"
                )
                f.write(
                    f"Overall mismatch mean : {summary['overall_mismatch_mean']:.4f}\n"
                )
                if "separation_gap" in summary:
                    f.write(
                        f"Separation Gap (Stage 3): {summary['separation_gap']:.4f}\n"
                    )
                if summary.get("auc_roc") is not None:
                    f.write(f"AUC-ROC (Stage 3)       : {summary['auc_roc']:.4f}\n")
                final_stage3_metrics = all_metrics[-1]
                f.write(
                    f"Final-epoch Precision/Recall/F1 (Stage 3, at margin={broadened_margin}): "
                    f"P={final_stage3_metrics['precision']:.4f}, "
                    f"R={final_stage3_metrics['recall']:.4f}, "
                    f"F1={final_stage3_metrics['f1']:.4f}\n"
                )
                f.write("\n--- Automatic Post-Hoc Margin Calibration ---\n")
                f.write(f"Recommended decision threshold : {best['threshold']:.4f}\n")
                f.write(f"  Precision at that threshold  : {best['precision']:.4f}\n")
                f.write(f"  Recall at that threshold     : {best['recall']:.4f}\n")
                f.write(f"  F1 at that threshold          : {best['f1']:.4f}\n")
                f.write(
                    f"  (Full sweep saved to {recommended_margin_path.name}; use this value "
                    f"as --margin in validate.py / diagnostics.py / cross_validate_bliss.py "
                    f"for this checkpoint rather than {broadened_margin}.)\n"
                )
                logger.info(f"Stage 3 stratified summary: {summary}")
        # Plot once after all requested curriculum stages are complete.
        plot_loss_curves(
            all_losses, stage_boundaries, mode, plot_dir / "loss_curves.png"
        )
        plot_f1_curve(
            all_metrics,
            stage_boundaries,
            plot_dir / "f1_vs_epoch.png",
            broadened_margin=broadened_margin if args.include_broadened else None,
        )
        logger.info(f"Final-epoch classification metrics: {all_metrics[-1]}")
        logger.info(f"Detailed report written to {report_path}")
        logger.info(
            f"\n{'='*25} FINISHED MODE: {mode.upper()} | TOTAL TIME: {format_time(time.time() - mode_start_time)} {'='*25}"
        )
    print("\n" + "=" * 70)
    print("All specified models have been trained.")
    print(
        f"Total script execution time: {format_time(time.time() - overall_start_time)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a multi-resolution detector model."
    )
    # Baseline training arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=CONFIGS.keys(),
        default=None,
        help="Specify a single mode to train. If not provided, all modes will be trained.",
    )
    parser.add_argument(
        "--filterbank",
        type=str,
        default=None,
        help="Override the station-A filterbank for the selected --mode.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Specify a custom run ID to override automatic naming.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Weight of the contrastive loss."
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.8,
        help="Margin for the Stage-1/2 contrastive loss.",
    )
    parser.add_argument(
        "--weight_factor",
        type=int,
        default=100,
        help="Weighting for signal pixels in reconstruction loss.",
    )
    parser.add_argument(
        "--include_broadened",
        action="store_true",
        help="If set, run an additional opt-in Stage 3 that fine-tunes the "
        "Stage-2 checkpoint on physically-broadened signal pairs.",
    )
    parser.add_argument(
        "--broadened_epochs",
        type=int,
        default=20,
        help="Total number of Stage-3 fine-tuning epochs (split across "
        "curriculum sub-stages if --broadened_curriculum is set).",
    )
    parser.add_argument(
        "--broadened_lr",
        type=float,
        default=5e-5,
        help="Learning rate for Stage-3 fine-tuning.",
    )
    parser.add_argument(
        "--broadened_snr_min", type=float, default=DEFAULT_BROADENED_SNR_RANGE[0]
    )
    parser.add_argument(
        "--broadened_snr_max", type=float, default=DEFAULT_BROADENED_SNR_RANGE[1]
    )
    parser.add_argument(
        "--p_mdwarf",
        type=float,
        default=0.75,
        help="M-dwarf fraction in the Monte Carlo broadening sampler.",
    )
    parser.add_argument(
        "--freq_ghz",
        type=float,
        default=0.150,
        help="Observing frequency (GHz) used for the physical broadening model.",
    )
    parser.add_argument(
        "--station_b_filterbank",
        type=str,
        default=None,
        help="Optional path to a SECOND real filterbank file for genuinely "
        "dual-site training pairs.",
    )
    # Broadened-signal training arguments
    parser.add_argument(
        "--broadened_margin",
        type=float,
        default=None,
        help="Separate contrastive-loss margin for Stage 3. Default: reuse "
        "--margin. Before a full run, calibrate the value on a short "
        "pilot (e.g. --broadened_epochs 5), read "
        "recommended_margin_stage3.json, then re-run the full Stage 3 "
        "with that value passed here.",
    )
    parser.add_argument(
        "--broadened_curriculum",
        action="store_true",
        help="Split Stage 3 into sequential sub-stages with "
        "progressively looser width caps (see "
        "--broadened_curriculum_widths_hz) for optimisation stability. "
        "Conditioned sampling remains active unless explicitly disabled.",
    )
    parser.add_argument(
        "--broadened_curriculum_widths_hz",
        type=str,
        default=",".join(str(w) for w in DEFAULT_CURRICULUM_WIDTHS_HZ),
        help="Comma-separated width caps (Hz) for --broadened_curriculum's "
        "sequential sub-stages.",
    )
    parser.add_argument(
        "--broadened_min_effective_snr",
        type=float,
        default=DEFAULT_MIN_EFFECTIVE_SNR,
        help="Minimum best-case effective post-broadening SNR a "
        "sampled width must be able to clear (at the top of "
        "[--broadened_snr_min, --broadened_snr_max]) to be accepted by "
        "the conditioned sampler. See "
        "lorentzian_signals.sample_one_broadening_conditioned's docstring.",
    )
    parser.add_argument(
        "--broadened_disable_conditioning",
        action="store_true",
        help="Disable conditioned width/SNR sampling for the registered "
        "unconditioned-population ablation.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="DataLoader worker processes."
    )
    args = parser.parse_args()
    if args.filterbank is not None and args.mode is None:
        parser.error("--filterbank requires an explicit --mode")
    main(args)
