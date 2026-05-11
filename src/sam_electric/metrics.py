from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_with_logits(logits, targets)
    return bce + dice


def prepare_masks_for_loss(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convierte salida SAM y GT a tensores [B, H, W] compatibles.

    HF SAM suele retornar [B, 1, 1, 256, 256] cuando multimask_output=False.
    """
    if pred_masks.ndim == 5:
        # [B, prompt_count, mask_count, H, W]. Este proyecto usa un prompt y una máscara por instancia.
        pred_masks = pred_masks[:, 0, 0, :, :]
    elif pred_masks.ndim == 4:
        # [B, mask_count, H, W]
        pred_masks = pred_masks[:, 0, :, :]
    elif pred_masks.ndim != 3:
        raise ValueError(f"Forma inesperada de pred_masks: {tuple(pred_masks.shape)}")

    gt = gt_masks.unsqueeze(1)
    gt = F.interpolate(gt, size=pred_masks.shape[-2:], mode="nearest").squeeze(1)
    return pred_masks, gt


def binary_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((intersection + eps) / (union + eps))


def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return float((2 * intersection + eps) / (denom + eps))


class SegmentationMetricAccumulator:
    def __init__(self) -> None:
        self.rows = []

    def update(self, category_name: str, iou: float, dice: float, file_name: str = "") -> None:
        self.rows.append(
            {
                "category_name": category_name,
                "file_name": file_name,
                "iou": float(iou),
                "dice": float(dice),
            }
        )

    def summary(self) -> Dict[str, object]:
        if not self.rows:
            return {"overall": {}, "by_category": {}}

        overall = {
            "mean_iou": float(np.mean([r["iou"] for r in self.rows])),
            "mean_dice": float(np.mean([r["dice"] for r in self.rows])),
            "instances": len(self.rows),
        }

        grouped = defaultdict(list)
        for row in self.rows:
            grouped[row["category_name"]].append(row)

        by_category = {}
        for cat, rows in grouped.items():
            by_category[cat] = {
                "mean_iou": float(np.mean([r["iou"] for r in rows])),
                "mean_dice": float(np.mean([r["dice"] for r in rows])),
                "instances": len(rows),
            }

        return {"overall": overall, "by_category": by_category}
