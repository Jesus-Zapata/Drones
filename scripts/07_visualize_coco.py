from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from sam_electric.coco import category_maps, xywh_to_xyxy
from sam_electric.utils import ensure_dir
from sam_electric.visualization import draw_box, overlay_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera visualizaciones rápidas de anotaciones COCO.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/coco_preview")
    parser.add_argument("--max-images", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco = COCO(args.annotations)
    id_to_name, _ = category_maps(coco)
    image_dir = Path(args.image_dir)
    output_dir = ensure_dir(args.output_dir)

    for count, image_id in enumerate(coco.getImgIds()[: args.max_images]):
        image_info = coco.loadImgs([image_id])[0]
        image_path = image_dir / image_info["file_name"]
        if not image_path.exists():
            continue
        image = np.array(Image.open(image_path).convert("RGB"))
        preview = image.copy()
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            mask = coco.annToMask(ann)
            label = id_to_name.get(ann["category_id"], "sin_clase")
            preview = overlay_mask(preview, mask, alpha=0.25)
            preview = draw_box(preview, xywh_to_xyxy(ann["bbox"]), label)
        out = output_dir / f"{count:04d}_{Path(image_info['file_name']).stem}.png"
        Image.fromarray(preview).save(out)
    print(f"Visualizaciones guardadas en: {output_dir}")


if __name__ == "__main__":
    main()
