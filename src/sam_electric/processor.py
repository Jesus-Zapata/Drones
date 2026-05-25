from __future__ import annotations

from typing import Any

from transformers import SamProcessor


def configure_sam_processor(
    processor: SamProcessor,
    image_size: int = 1024,
    mask_size: int = 256,
) -> SamProcessor:
    """Configura la resolución usada por el processor de Hugging Face SAM.

    image_size controla el lado mayor de la imagen procesada y el padding final.
    mask_size controla el lado mayor de las máscaras internas.
    """
    image_size = int(image_size)
    mask_size = int(mask_size)

    if image_size <= 0 or mask_size <= 0:
        raise ValueError("image_size y mask_size deben ser enteros positivos.")

    image_processor: Any = processor.image_processor
    image_processor.size = {"longest_edge": image_size}
    image_processor.pad_size = {"height": image_size, "width": image_size}
    image_processor.mask_size = {"longest_edge": mask_size}
    image_processor.mask_pad_size = {"height": mask_size, "width": mask_size}
    return processor


def load_configured_processor(
    model_name_or_path: str,
    image_size: int = 1024,
    mask_size: int = 256,
) -> SamProcessor:
    processor = SamProcessor.from_pretrained(model_name_or_path)
    return configure_sam_processor(processor, image_size=image_size, mask_size=mask_size)
