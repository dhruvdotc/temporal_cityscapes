from __future__ import annotations

"""Shared training and evaluation utilities used by notebooks 1-4.

This module intentionally stays small and pragmatic:
- basic device/model construction helpers
- one training epoch
- mIoU evaluation with optional progress logging
- checkpoint loading
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import time

import torch
import torch.nn as nn

from .metrics import confusion_matrix, miou_from_cm


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_deeplab(num_classes: int = 19, pretrained: bool = True) -> nn.Module:
    """Build a DeepLabV3-ResNet50 model with a Cityscapes-size output head."""
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        deeplabv3_resnet50,
    )

    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights)

    # torchvision classifier layout: Conv -> BN -> ReLU -> Dropout -> Conv
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


@dataclass
class TrainConfig:
    """Small set of knobs for one-epoch training behavior."""

    use_amp: bool = True
    grad_clip_norm: Optional[float] = None


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: Optional[TrainConfig] = None,
) -> float:
    """Run one epoch and return average training loss."""
    cfg = cfg or TrainConfig()

    model.train()
    total_loss = 0.0

    amp_enabled = device.type == "cuda" and cfg.use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(x)["out"]
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if cfg.grad_clip_norm is not None:
            # Gradients must be unscaled before clipping when using AMP.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def evaluate(
    model: nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
    ignore_index: int = 255,
    use_amp: bool = True,
    max_batches: Optional[int] = None,
    progress: bool = False,
    progress_every: int = 1,
    progress_prefix: str = "eval",
) -> Tuple[float, torch.Tensor]:
    """Compute mIoU and per-class IoU over a dataloader."""
    model.eval()
    cm_total = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    amp_enabled = device.type == "cuda" and use_amp
    total_batches: Optional[int]
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None

    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)

    start = time.time()

    with torch.inference_mode():
        for batch_idx, (x, y, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)["out"]

            pred = torch.argmax(logits, dim=1)
            cm_total += confusion_matrix(
                pred,
                y,
                num_classes=num_classes,
                ignore_index=ignore_index,
            ).to(device)

            if progress:
                done = batch_idx + 1
                should_print = done % max(1, progress_every) == 0
                if total_batches is not None:
                    should_print = should_print or done == total_batches

                if should_print:
                    elapsed = time.time() - start
                    if total_batches is None:
                        print(f"[{progress_prefix}] batch {done} | elapsed {elapsed:.1f}s", flush=True)
                    else:
                        pct = 100.0 * done / max(1, total_batches)
                        sec_per_batch = elapsed / max(1, done)
                        eta = max(0.0, (total_batches - done) * sec_per_batch)
                        print(
                            f"[{progress_prefix}] {done}/{total_batches} ({pct:5.1f}%)"
                            f" | elapsed {elapsed:.1f}s | eta {eta:.1f}s",
                            flush=True,
                        )

    miou, iou_per_class = miou_from_cm(cm_total.cpu())
    return miou, iou_per_class


def load_checkpoint(path, model: nn.Module, map_location: str | torch.device = "cpu") -> dict:
    """Load checkpoint weights into ``model`` and return raw checkpoint dict."""
    ckpt = torch.load(path, map_location=map_location)

    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError("Checkpoint missing model weights key (expected model/model_state_dict/state_dict)")

    model.load_state_dict(state_dict)
    return ckpt