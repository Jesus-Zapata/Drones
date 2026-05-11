from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from sam_electric.utils import ensure_dir
from sam_electric.visualization import save_mask_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera máscaras automáticas con SAM original. Ojo: estas máscaras NO tienen clase; "
            "sirven para pre-etiquetado o revisión humana."
        )
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth de SAM original, por ejemplo sam_vit_b_01ec64.pth")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--output-dir", default="outputs/auto_masks")
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "No se pudo importar segment_anything. Instala requirements.txt o instala el paquete desde el repositorio oficial."
        ) from exc

    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = ensure_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        min_mask_region_area=args.min_mask_region_area,
    )

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])
    metadata = []
    for image_path in tqdm(image_paths, desc="Generando máscaras"):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image_rgb)

        image_out_dir = ensure_dir(output_dir / image_path.stem)
        for idx, item in enumerate(masks):
            mask = item["segmentation"].astype(np.uint8)
            mask_path = image_out_dir / f"mask_{idx:04d}.png"
            save_mask_png(mask, mask_path)
            metadata.append(
                {
                    "file_name": image_path.name,
                    "mask_path": str(mask_path),
                    "area": int(item.get("area", mask.sum())),
                    "bbox_xywh": [float(v) for v in item.get("bbox", [])],
                    "predicted_iou": float(item.get("predicted_iou", 0.0)),
                    "stability_score": float(item.get("stability_score", 0.0)),
                    "label": None,
                }
            )

    with (output_dir / "auto_masks_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Máscaras automáticas guardadas en: {output_dir}")
    print("Recuerda: estas máscaras deben revisarse y asignarse a clases antes de usarlas como ground truth.")


if __name__ == "__main__":
    main()
