#!/usr/bin/env python3
"""Fine-tune Stage 3 on detector-conditioned 10--100 Hz candidate views.

This is intentionally a short Stage-4 adaptation run, not a restart of the
three-stage curriculum.  It reuses the Stage-3 U-Net encoder, skips the
reconstruction decoder, and optimises the actual coincidence objective on
de-chirped + frequency-scrunched views.  Validation frequencies are disjoint
from training frequencies and select both the checkpoint and operating
threshold; the test set is never consulted here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lofts_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from stage4_data import CandidatePairFactory, Stage4PairDataset
from stage4_model import (
    CandidateCoincidenceModel,
    atomic_torch_save,
    contrastive_loss,
    load_stage3_backbone,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is useful for a paper trail.  warn_only avoids failing on an
    # older CUDA kernel with no deterministic implementation.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older PyTorch
        pass


def finite_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> Dict[str, float]:
    if np.unique(labels).size < 2:
        return {"auc_roc": float("nan"), "average_precision": float("nan")}
    return {
        "auc_roc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def select_f1_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> Dict[str, float]:
    """Select an operating point on validation only."""

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        threshold = 0.5
    else:
        denom = precision[:-1] + recall[:-1]
        f1 = np.divide(
            2.0 * precision[:-1] * recall[:-1],
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )
        threshold = float(thresholds[int(np.nanargmax(f1))])
    predicted = probabilities >= threshold
    truth = labels.astype(bool)
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1_value = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": f1_value,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def evaluate(
    model, loader, device, use_amp: bool
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    probabilities_all = []
    losses = []
    for view_a, view_b, labels in tqdm(loader, desc="validation", leave=False):
        view_a = view_a.to(device, non_blocking=True)
        view_b = view_b.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits, _, _, _ = model(view_a, view_b)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        losses.append(float(loss.item()))
        labels_all.append(labels.cpu().numpy())
        probabilities_all.append(torch.sigmoid(logits).cpu().numpy())
    labels_np = np.concatenate(labels_all)
    probabilities_np = np.concatenate(probabilities_all)
    metrics = classification_metrics(labels_np, probabilities_np)
    metrics["bce"] = finite_mean(losses)
    return metrics, labels_np, probabilities_np


def encoder_parameters(model):
    modules = (
        model.backbone.enc1,
        model.backbone.enc2,
        model.backbone.enc3,
        model.backbone.bottleneck,
        model.backbone.projection_head,
    )
    for module in modules:
        yield from module.parameters()


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp: bool,
    contrastive_weight: float,
    contrastive_margin: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": [], "bce": [], "contrastive": []}
    for view_a, view_b, labels in tqdm(loader, desc="training", leave=False):
        view_a = view_a.to(device, non_blocking=True)
        view_b = view_b.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits, z_a, z_b, _ = model(view_a, view_b)
            bce = F.binary_cross_entropy_with_logits(logits, labels)
            con = contrastive_loss(z_a, z_b, labels, margin=contrastive_margin)
            loss = bce + float(contrastive_weight) * con
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        totals["loss"].append(float(loss.item()))
        totals["bce"].append(float(bce.item()))
        totals["contrastive"].append(float(con.item()))
    return {key: finite_mean(value) for key, value in totals.items()}


def plot_history(history: List[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train total")
    axes[0].plot(epochs, [row["train_bce"] for row in history], label="train BCE")
    axes[0].plot(
        epochs, [row["train_contrastive"] for row in history], label="train contrastive"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        epochs, [row["val_auc_roc"] for row in history], marker="o", label="AUC-ROC"
    )
    axes[1].plot(
        epochs,
        [row["val_average_precision"] for row in history],
        marker="o",
        label="AP",
    )
    axes[1].axhline(0.5, color="gray", ls="--", lw=1)
    axes[1].set_ylim(0.4, 1.01)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation metric")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Stage-4 candidate-conditioned fine-tuning")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(args) -> None:
    set_seed(args.seed)
    if args.station_b_is_real and args.station_b is None:
        raise ValueError("--station_b_is_real requires --station_b")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_stage4.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    if log_path.exists():
        log_path.unlink()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = bool(args.amp and device.type == "cuda")
    log("Device: %s; AMP: %s" % (device, use_amp))
    if device.type != "cuda":
        log(
            "WARNING: CUDA is unavailable. Use the Sweden compute node for the full run; "
            "the head node is suitable only for smoke tests."
        )

    common_factory = dict(
        mode=args.mode,
        station_a_filterbank=args.station_a,
        station_b_filterbank=args.station_b,
        station_b_is_proxy=not args.station_b_is_real,
        widths_hz=(args.width_min, args.width_max),
        target_snr_range=(args.snr_min, args.snr_max),
        snr_mode=args.snr_mode,
        integration=args.integration,
        max_abs_drift_hz_s=args.max_abs_drift,
        width_log_error_sigma=args.width_log_error_sigma,
        drift_error_channels_per_tile=args.drift_error_channels_per_tile,
        center_error_channels=args.center_error_channels,
        station_modulation_log_sigma=args.station_modulation_log_sigma,
        remove_static_bandpass=not args.disable_bandpass_removal,
    )
    train_factory = CandidatePairFactory(split="train", **common_factory)
    val_factory = CandidatePairFactory(split="val", **common_factory)
    train_dataset = Stage4PairDataset(train_factory, args.n_train, args.seed)
    val_dataset = Stage4PairDataset(val_factory, args.n_val, args.seed + 10_000)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        # Workers are intentionally re-created for each epoch so the updated
        # deterministic epoch counter is visible in worker copies.
        persistent_workers=False,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    backbone = load_stage3_backbone(args.stage3_checkpoint, device="cpu")
    model = CandidateCoincidenceModel(
        backbone=backbone,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_physics_features=not args.disable_physics_features,
    ).to(device)
    model.set_encoder_trainability("head")

    optimizer = torch.optim.AdamW(
        [
            {"params": list(encoder_parameters(model)), "lr": args.encoder_lr},
            {"params": list(model.pair_head.parameters()), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-7
    )
    scaler = GradScaler(enabled=use_amp)

    run_config = vars(args).copy()
    run_config["device_resolved"] = str(device)
    run_config["station_b_proxy"] = bool(train_factory.station_b_proxy)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)

    best_auc = -np.inf
    best_epoch = -1
    patience_used = 0
    history = []
    best_path = out_dir / "model_stage4_candidate_conditioned.pt"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        if epoch <= args.head_only_epochs:
            level = "head"
        elif args.unfreeze_all_epoch > 0 and epoch >= args.unfreeze_all_epoch:
            level = "all"
        else:
            level = "top"
        model.set_encoder_trainability(level)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            args.contrastive_weight,
            args.contrastive_margin,
            args.grad_clip,
        )
        val_metrics, _, _ = evaluate(model, val_loader, device, use_amp)
        scheduler.step(val_metrics["auc_roc"])
        row = {
            "epoch": epoch,
            "trainability": level,
            "train_loss": train_metrics["loss"],
            "train_bce": train_metrics["bce"],
            "train_contrastive": train_metrics["contrastive"],
            "val_bce": val_metrics["bce"],
            "val_auc_roc": val_metrics["auc_roc"],
            "val_average_precision": val_metrics["average_precision"],
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        log(
            "Epoch %02d | train=%s loss=%.4f bce=%.4f con=%.4f | "
            "val AUC=%.4f AP=%.4f BCE=%.4f"
            % (
                epoch,
                level,
                row["train_loss"],
                row["train_bce"],
                row["train_contrastive"],
                row["val_auc_roc"],
                row["val_average_precision"],
                row["val_bce"],
            )
        )
        if (
            np.isfinite(row["val_auc_roc"])
            and row["val_auc_roc"] > best_auc + args.min_delta
        ):
            best_auc = row["val_auc_roc"]
            best_epoch = epoch
            patience_used = 0
            payload = {
                "format_version": 1,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "use_physics_features": not args.disable_physics_features,
                },
                "preprocessing": {
                    "integration": args.integration,
                    "signed_foff_required": True,
                    "remove_static_bandpass": not args.disable_bandpass_removal,
                    "width_log_error_sigma": args.width_log_error_sigma,
                    "drift_error_channels_per_tile": args.drift_error_channels_per_tile,
                    "center_error_channels": args.center_error_channels,
                    "station_modulation_log_sigma": args.station_modulation_log_sigma,
                },
                "population": {
                    "snr_mode": args.snr_mode,
                    "width_hz": [args.width_min, args.width_max],
                    "target_snr": [args.snr_min, args.snr_max],
                },
                "stage3_source": str(Path(args.stage3_checkpoint).resolve()),
                "best_epoch": best_epoch,
                "best_validation": dict(val_metrics),
                "station_b_proxy": bool(train_factory.station_b_proxy),
                "run_config": run_config,
            }
            atomic_torch_save(payload, str(best_path))
        else:
            patience_used += 1
        if patience_used >= args.early_stopping_patience:
            log(
                "Early stopping after %d epochs without validation AUC improvement."
                % patience_used
            )
            break

    if not best_path.is_file():
        raise RuntimeError(
            "no checkpoint was selected; validation AUC was non-finite. "
            "Increase --n_val and verify that both classes are generated."
        )

    # Load the selected model and calibrate its operating threshold on the
    # fixed validation set.  The test set remains untouched.
    selected = torch.load(str(best_path), map_location=device)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    val_metrics, val_labels, val_probs = evaluate(model, val_loader, device, use_amp)
    operating = select_f1_threshold(val_labels, val_probs)
    selected["validation_operating_point"] = operating
    selected["best_validation"] = val_metrics
    selected["history"] = history
    atomic_torch_save(selected, str(best_path))

    with (out_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (out_dir / "validation_operating_point.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(operating, handle, indent=2, sort_keys=True)
    plot_history(history, out_dir / "stage4_training_curves.png")

    elapsed = time.time() - start_time
    report = (
        "Stage-4 candidate-conditioned training complete\n"
        "best epoch: %d\n"
        "validation AUC-ROC: %.6f\n"
        "validation average precision: %.6f\n"
        "validation-selected threshold: %.6f\n"
        "validation F1 at threshold: %.6f\n"
        "station B background is a synthetic station proxy: %s\n"
        "elapsed seconds: %.1f\n"
        "checkpoint: %s\n"
        % (
            best_epoch,
            val_metrics["auc_roc"],
            val_metrics["average_precision"],
            operating["threshold"],
            operating["f1"],
            train_factory.station_b_proxy,
            elapsed,
            best_path,
        )
    )
    (out_dir / "stage4_training_report.txt").write_text(report, encoding="utf-8")
    log(report.rstrip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-4 fine-tuning on de-chirped + frequency-scrunched candidates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="high_freq", choices=("high_freq",))
    parser.add_argument(
        "--station_a", required=True, help="Accessible 0000.fil background"
    )
    parser.add_argument(
        "--station_b",
        default=None,
        help="Optional second accessible 0000.fil background",
    )
    parser.add_argument(
        "--station_b_is_real",
        action="store_true",
        help="Set only when --station_b is genuinely from the second station; "
        "the supplied second Sweden background must remain a proxy",
    )
    parser.add_argument("--stage3_checkpoint", required=True)
    parser.add_argument(
        "--out_dir", default="training_runs/stage4_candidate_conditioned"
    )
    parser.add_argument("--snr_mode", choices=("detected", "power"), default="detected")
    parser.add_argument(
        "--integration", choices=("boxcar", "lorentzian", "gaussian"), default="boxcar"
    )
    parser.add_argument("--width_min", type=float, default=10.0)
    parser.add_argument("--width_max", type=float, default=100.0)
    parser.add_argument("--snr_min", type=float, default=8.0)
    parser.add_argument("--snr_max", type=float, default=30.0)
    parser.add_argument("--max_abs_drift", type=float, default=4.0)
    parser.add_argument("--width_log_error_sigma", type=float, default=0.15)
    parser.add_argument("--drift_error_channels_per_tile", type=float, default=0.35)
    parser.add_argument("--center_error_channels", type=float, default=0.35)
    parser.add_argument(
        "--station_modulation_log_sigma",
        type=float,
        default=0.06,
        help="Independent smooth station-to-station amplitude variation",
    )
    parser.add_argument("--disable_bandpass_removal", action="store_true")
    parser.add_argument("--n_train", type=int, default=8000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Keep 0 on the memory-constrained LOFTS system",
    )
    parser.add_argument("--head_only_epochs", type=int, default=1)
    parser.add_argument(
        "--unfreeze_all_epoch",
        type=int,
        default=-1,
        help="-1 keeps enc1/enc2 frozen; safer for a short adaptation run",
    )
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--encoder_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--contrastive_weight", type=float, default=0.25)
    parser.add_argument("--contrastive_margin", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--disable_physics_features",
        action="store_true",
        help="Ablation: remove detector-informed central-track features",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, or cpu")
    parser.add_argument(
        "--amp", action="store_true", help="Enable CUDA mixed precision"
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
