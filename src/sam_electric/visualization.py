from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone una máscara binaria sobre una imagen RGB sin fijar una paleta específica."""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    overlay = image.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)
    overlay[mask.astype(bool)] = (overlay[mask.astype(bool)] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def draw_box(image: np.ndarray, box_xyxy: Iterable[float], label: str | None = None) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    if label:
        cv2.putText(out, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def save_side_by_side(
    image: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    output_path: str | Path,
    title: str = "",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [("Imagen", image)]
    if gt_mask is not None:
        panels.append(("GT", overlay_mask(image, gt_mask)))
    panels.append(("Predicción", overlay_mask(image, pred_mask)))

    fig = plt.figure(figsize=(5 * len(panels), 5))
    if title:
        fig.suptitle(title)
    for idx, (name, panel) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), idx)
        ax.imshow(panel)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_mask_png(mask: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_uint8).save(output_path)
