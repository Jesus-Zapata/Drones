from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def resize_mask_to_image(mask, image):
    """
    Ajusta una máscara al tamaño de la imagen original.
    La máscara puede venir en 256x256 porque SAM trabaja internamente
    con máscaras de baja resolución.
    """
    if mask.ndim > 2:
        mask = np.squeeze(mask)

    image_h, image_w = image.shape[:2]

    if mask.shape[:2] != (image_h, image_w):
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_uint8)
        mask_pil = mask_pil.resize((image_w, image_h), resample=Image.Resampling.NEAREST)
        mask = np.array(mask_pil) > 0
    else:
        mask = mask.astype(bool)

    return mask


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.45):
    """
    Pinta una máscara sobre una imagen.
    Si la máscara no tiene el mismo tamaño que la imagen, la redimensiona.
    """
    image = np.array(image).copy()

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    mask = resize_mask_to_image(mask, image)

    overlay = image.copy()
    color = np.array(color, dtype=np.uint8)

    overlay[mask] = (
        overlay[mask] * (1 - alpha) + color * alpha
    ).astype(np.uint8)

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
