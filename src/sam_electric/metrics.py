from __future__ import annotations

from collections import defaultdict
from typing import Dict

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
    """Convierte salida SAM y GT a tensores [B, H, W] compatibles."""
    if pred_masks.ndim == 5:
        pred_masks = pred_masks[:, 0, 0, :, :]
    elif pred_masks.ndim == 4:
        pred_masks = pred_masks[:, 0, :, :]
    elif pred_masks.ndim != 3:
        raise ValueError(f"Forma inesperada de pred_masks: {tuple(pred_masks.shape)}")

    gt = gt_masks.unsqueeze(1)
    gt = F.interpolate(gt, size=pred_masks.shape[-2:], mode="nearest").squeeze(1)
    return pred_masks, gt


def binary_confusion(pred: np.ndarray, target: np.ndarray) -> Dict[str, int]:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    tp = int(np.logical_and(pred_b, target_b).sum())
    tn = int(np.logical_and(~pred_b, ~target_b).sum())
    fp = int(np.logical_and(pred_b, ~target_b).sum())
    fn = int(np.logical_and(~pred_b, target_b).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_confusion(conf: Dict[str, int], eps: float = 1e-7) -> Dict[str, float]:
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]
    total = tp + tn + fp + fn

    iou_fg = (tp + eps) / (tp + fp + fn + eps)
    iou_bg = (tn + eps) / (tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    pixel_accuracy = (tp + tn + eps) / (total + eps)
    foreground_accuracy = (tp + eps) / (tp + fn + eps)
    background_accuracy = (tn + eps) / (tn + fp + eps)
    mpa = 0.5 * (foreground_accuracy + background_accuracy)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    return {
        "iou": float(iou_fg),
        "iou_background": float(iou_bg),
        "dice": float(dice),
        "pixel_accuracy": float(pixel_accuracy),
        "mpa": float(mpa),
        "foreground_accuracy": float(foreground_accuracy),
        "background_accuracy": float(background_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def binary_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    return metrics_from_confusion(binary_confusion(pred, target), eps=eps)["iou"]


def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    return metrics_from_confusion(binary_confusion(pred, target), eps=eps)["dice"]


class SegmentationMetricAccumulator:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def update(
        self,
        category_name: str,
        file_name: str,
        metrics: Dict[str, float],
        inference_time_ms: float | None = None,
        gpu_memory_mb: float | None = None,
    ) -> None:
        row = {
            "category_name": category_name,
            "file_name": file_name,
            **{k: float(v) for k, v in metrics.items()},
        }
        if inference_time_ms is not None:
            row["inference_time_ms"] = float(inference_time_ms)
        if gpu_memory_mb is not None:
            row["gpu_memory_mb"] = float(gpu_memory_mb)
        self.rows.append(row)

    def _aggregate_rows(self, rows: list[dict]) -> Dict[str, float | int]:
        metric_keys = [
            "iou",
            "iou_background",
            "dice",
            "pixel_accuracy",
            "mpa",
            "foreground_accuracy",
            "background_accuracy",
            "precision",
            "recall",
            "f1",
            "inference_time_ms",
            "gpu_memory_mb",
        ]
        out: Dict[str, float | int] = {"instances": len(rows)}
        for key in metric_keys:
            values = [r[key] for r in rows if key in r and r[key] is not None]
            if values:
                out[f"mean_{key}"] = float(np.mean(values))
                out[f"std_{key}"] = float(np.std(values))
        return out

    def summary(self) -> Dict[str, object]:
        if not self.rows:
            return {"overall": {}, "by_category": {}, "rows": []}

        overall = self._aggregate_rows(self.rows)

        grouped = defaultdict(list)
        for row in self.rows:
            grouped[row["category_name"]].append(row)

        by_category = {cat: self._aggregate_rows(rows) for cat, rows in grouped.items()}

        # En este proyecto de instancias, mIoU se reporta como promedio de IoU por clase.
        cat_ious = [metrics["mean_iou"] for metrics in by_category.values() if "mean_iou" in metrics]
        if cat_ious:
            overall["miou_by_category"] = float(np.mean(cat_ious))

        return {
            "overall": overall,
            "by_category": by_category,
            "rows": self.rows,
        }
