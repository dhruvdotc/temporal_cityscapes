# src/metrics.py
from __future__ import annotations

from typing import Tuple

import torch


@torch.no_grad()
def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Build a [C, C] confusion matrix from predicted and target class maps.

    `pred` and `target` may be `[H, W]` or `[N, H, W]` and are expected to be
    class ids. Pixels where `target == ignore_index` are ignored.
    """
    pred_flat = pred.reshape(-1).long()
    target_flat = target.reshape(-1).long()

    # Keep only supervised pixels and valid class ids so bincount/reshape
    # remains stable even if a model outputs out-of-range labels.
    valid = target_flat != ignore_index
    valid &= (target_flat >= 0) & (target_flat < num_classes)
    valid &= (pred_flat >= 0) & (pred_flat < num_classes)

    pred_flat = pred_flat[valid]
    target_flat = target_flat[valid]

    idx = target_flat * num_classes + pred_flat
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


@torch.no_grad()
def miou_from_cm(cm: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """Compute mean IoU and per-class IoU from a confusion matrix."""
    cm = cm.float()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    denom = tp + fp + fn
    iou_per_class = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
    miou = iou_per_class.mean().item()
    return miou, iou_per_class.cpu()