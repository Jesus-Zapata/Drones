from __future__ import annotations

from io import BytesIO
from typing import Literal

import cv2
import numpy as np
from PIL import Image

CorruptionName = Literal["none", "blur", "gaussian_noise", "jpeg_compression"]


def apply_corruption(
    image: Image.Image,
    corruption: str | None = None,
    severity: int = 3,
) -> Image.Image:
    """Aplica degradaciones simples para evaluar robustez.

    severity va de 1 a 5. La función no cambia la máscara, solo la imagen.
    """
    corruption = (corruption or "none").lower()
    severity = int(max(1, min(severity, 5)))

    if corruption in {"none", "", "null"}:
        return image

    arr = np.array(image.convert("RGB"))

    if corruption == "blur":
        kernel = 2 * severity + 1
        out = cv2.GaussianBlur(arr, (kernel, kernel), sigmaX=severity)
        return Image.fromarray(out)

    if corruption == "gaussian_noise":
        sigma = 8 * severity
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        out = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    if corruption == "jpeg_compression":
        quality = max(5, 95 - severity * 15)
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    raise ValueError(
        f"Corrupción no soportada: {corruption}. "
        "Usa: none, blur, gaussian_noise o jpeg_compression."
    )
