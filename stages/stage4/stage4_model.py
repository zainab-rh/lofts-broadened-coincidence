"""Candidate-conditioned Siamese head built on the existing LOFAR U-Net.

Stage 3 optimised ``reconstruction + alpha * contrastive``.  In the supplied
loss curves, reconstruction is ~30--35 while contrastive is ~0.005--0.03, so
the coincidence objective contributes negligibly at the reported alpha.  This
Stage-4 wrapper reuses the trained encoder/projection weights but does not run
or optimise the decoder.  It trains a small symmetric pair head with BCE plus
an explicitly weighted contrastive regulariser.

The symmetric features (absolute difference, elementwise product, Euclidean
distance, cosine similarity) make station order irrelevant by construction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - only for lightweight syntax QA.
    torch = None
    nn = None
    F = None


_BaseModule = nn.Module if nn is not None else object


class CandidateCoincidenceModel(_BaseModule):
    """Siamese coincidence classifier using the existing U-Net encoder."""

    def __init__(
        self,
        backbone=None,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        use_physics_features: bool = True,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for CandidateCoincidenceModel")
        super().__init__()
        if backbone is None:
            from train import UNet

            backbone = UNet(latent_dim=latent_dim)
        self.backbone = backbone
        linear_layers = [
            module
            for module in self.backbone.projection_head.modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_layers:
            raise ValueError("backbone projection_head has no Linear output layer")
        inferred_latent_dim = int(linear_layers[-1].out_features)
        if int(latent_dim) != inferred_latent_dim:
            raise ValueError(
                "latent_dim=%d does not match checkpoint backbone output=%d"
                % (int(latent_dim), inferred_latent_dim)
            )
        self.latent_dim = inferred_latent_dim
        self.use_physics_features = bool(use_physics_features)
        self.n_candidate_statistics = 6
        # For each statistic: |a-b|, a*b, min(a,b), max(a,b), plus one
        # centre-track correlation.  These are station-order invariant and
        # expose the known-candidate matched-filter evidence directly instead
        # of asking a globally pooled encoder to rediscover it.
        physics_dim = (
            4 * self.n_candidate_statistics + 1 if self.use_physics_features else 0
        )
        pair_dim = 2 * self.latent_dim + 2 + physics_dim
        self.pair_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 4, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 4, 32), 1),
        )

    def encode(self, x):
        """Run only the encoder and projection head (decoder is skipped)."""

        e1 = self.backbone.enc1(x)
        e2 = self.backbone.enc2(self.backbone.pool(e1))
        e3 = self.backbone.enc3(self.backbone.pool(e2))
        bottleneck = self.backbone.bottleneck(self.backbone.pool(e3))
        z = self.backbone.projection_head(bottleneck)
        return F.normalize(z, p=2, dim=1)

    @staticmethod
    def symmetric_pair_features(z_a, z_b):
        abs_diff = torch.abs(z_a - z_b)
        product = z_a * z_b
        distance = torch.linalg.vector_norm(z_a - z_b, ord=2, dim=1, keepdim=True)
        cosine = torch.sum(z_a * z_b, dim=1, keepdim=True)
        return torch.cat((abs_diff, product, distance, cosine), dim=1)

    @staticmethod
    def candidate_tracks(x):
        """Return central and local time series on odd or even grids."""
        centre = x.shape[-1] // 2
        if x.shape[-1] % 2 == 0:
            centre_track = torch.mean(x[:, 0, :, centre - 1 : centre + 1], dim=2)
            lo = max(0, centre - 2)
            hi = min(x.shape[-1], centre + 2)
        else:
            centre_track = x[:, 0, :, centre]
            lo = max(0, centre - 1)
            hi = min(x.shape[-1], centre + 2)
        local_track = torch.mean(x[:, 0, :, lo:hi], dim=2)
        return centre_track, local_track

    @classmethod
    def candidate_statistics(cls, x):
        """Six differentiable statistics around the known central track."""

        centre_track, local_track = cls.candidate_tracks(x)
        return torch.stack(
            (
                torch.mean(centre_track, dim=1),
                torch.std(centre_track, dim=1, unbiased=False),
                torch.amax(centre_track, dim=1),
                torch.mean(local_track, dim=1),
                torch.std(local_track, dim=1, unbiased=False),
                torch.mean(torch.abs(local_track), dim=1),
            ),
            dim=1,
        )

    @classmethod
    def physics_pair_features(cls, x_a, x_b):
        stats_a = cls.candidate_statistics(x_a)
        stats_b = cls.candidate_statistics(x_b)
        track_a, _ = cls.candidate_tracks(x_a)
        track_b, _ = cls.candidate_tracks(x_b)
        track_a = track_a - torch.mean(track_a, dim=1, keepdim=True)
        track_b = track_b - torch.mean(track_b, dim=1, keepdim=True)
        correlation = torch.sum(track_a * track_b, dim=1, keepdim=True) / (
            torch.linalg.vector_norm(track_a, dim=1, keepdim=True)
            * torch.linalg.vector_norm(track_b, dim=1, keepdim=True)
            + 1e-6
        )
        return torch.cat(
            (
                torch.abs(stats_a - stats_b),
                stats_a * stats_b,
                torch.minimum(stats_a, stats_b),
                torch.maximum(stats_a, stats_b),
                correlation,
            ),
            dim=1,
        )

    def forward(self, x_a, x_b):
        z_a = self.encode(x_a)
        z_b = self.encode(x_b)
        features = self.symmetric_pair_features(z_a, z_b)
        if self.use_physics_features:
            features = torch.cat(
                (features, self.physics_pair_features(x_a, x_b)), dim=1
            )
        logits = self.pair_head(features).squeeze(1)
        distance = torch.linalg.vector_norm(z_a - z_b, ord=2, dim=1)
        return logits, z_a, z_b, distance

    def set_encoder_trainability(self, level: str) -> None:
        """Control fine-tuning depth without touching the unused decoder.

        ``head``: pair head only.
        ``top``: projection, bottleneck, and enc3.
        ``all``: all encoder blocks plus projection.
        """

        if level not in ("head", "top", "all"):
            raise ValueError("level must be 'head', 'top', or 'all'")
        encoder_modules = (
            self.backbone.enc1,
            self.backbone.enc2,
            self.backbone.enc3,
            self.backbone.bottleneck,
            self.backbone.projection_head,
        )
        for module in encoder_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
        if level in ("top", "all"):
            for module in (
                self.backbone.enc3,
                self.backbone.bottleneck,
                self.backbone.projection_head,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        if level == "all":
            for module in (self.backbone.enc1, self.backbone.enc2):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        for parameter in self.pair_head.parameters():
            parameter.requires_grad = True


def contrastive_loss(z_a, z_b, labels, margin: float = 1.0):
    """Classic contrastive loss for labels 1=match, 0=mismatch."""

    distances = torch.linalg.vector_norm(z_a - z_b, ord=2, dim=1)
    positive = labels * distances.square()
    negative = (1.0 - labels) * F.relu(float(margin) - distances).square()
    return torch.mean(positive + negative)


def strip_state_dict_prefix(
    state_dict: Dict[str, object], prefix: str
) -> Dict[str, object]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _extract_state_dict(payload) -> Dict[str, object]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(isinstance(key, str) for key in payload):
            return payload
    raise ValueError("checkpoint does not contain a recognisable state_dict")


def load_stage3_backbone(checkpoint_path: str, device="cpu"):
    """Load an original Stage-3 UNet checkpoint with strict key checking."""

    if torch is None:
        raise ImportError("PyTorch is required to load checkpoints")
    from train import UNet

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(str(path), map_location=device)
    state = strip_state_dict_prefix(_extract_state_dict(payload), "module.")
    # A Stage-4 checkpoint stores backbone keys beneath "backbone."; reject it
    # here so the user cannot accidentally start from the wrong file.
    if any(key.startswith("backbone.") for key in state):
        raise ValueError("expected Stage-3 UNet weights, received a Stage-4 checkpoint")
    projection_weight = state.get("projection_head.4.weight")
    latent_dim = (
        int(projection_weight.shape[0]) if projection_weight is not None else 512
    )
    model = UNet(latent_dim=latent_dim)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:  # defensive
        raise RuntimeError(
            "strict Stage-3 checkpoint load unexpectedly reported key mismatch"
        )
    return model


def load_stage4_checkpoint(checkpoint_path: str, device="cpu"):
    """Load a Stage-4 model and return ``(model, checkpoint_metadata)``."""

    if torch is None:
        raise ImportError("PyTorch is required to load checkpoints")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(str(path), map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("not a Stage-4 checkpoint (missing model_state_dict)")
    config = payload.get("model_config", {})
    model = CandidateCoincidenceModel(
        latent_dim=int(config.get("latent_dim", 512)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.15)),
        use_physics_features=bool(config.get("use_physics_features", True)),
    )
    state = strip_state_dict_prefix(payload["model_state_dict"], "module.")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, payload


def atomic_torch_save(payload: dict, path: str) -> None:
    """Write a checkpoint atomically on a single filesystem."""

    if torch is None:
        raise ImportError("PyTorch is required to save checkpoints")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(target))
