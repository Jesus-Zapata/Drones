from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from sam_electric.utils import ensure_dir, get_device, load_config
from sam_electric.visualization import draw_box, overlay_mask, save_mask_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia con SAM usando cajas o puntos como prompts.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Carpeta del modelo. Por defecto usa training.output_dir/best.")
    parser.add_argument("--image-dir", required=True, help="Carpeta con imágenes nuevas.")
    parser.add_argument("--prompts", required=True, help="JSON con prompts por imagen.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_prompts(path: str | Path) -> Dict[str, List[dict]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for item in data.get("images", []):
        mapping[item["file_name"]] = item.get("prompts", [])
    return mapping


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(cfg["training"]["output_dir"]) / "best")
    output_dir = ensure_dir(args.output_dir or cfg["inference"]["output_dir"])
    threshold = float(cfg["inference"].get("threshold", 0.5))

    device = get_device()
    processor = SamProcessor.from_pretrained(checkpoint)
    model = SamModel.from_pretrained(checkpoint).to(device)
    model.eval()

    image_dir = Path(args.image_dir)
    prompt_map = load_prompts(args.prompts)
    results = []

    for file_name, prompts in tqdm(prompt_map.items(), desc="Inferencia"):
        image_path = image_dir / file_name
        if not image_path.exists():
            print(f"Imagen no encontrada: {image_path}")
            continue
        if not prompts:
            print(f"Sin prompts para: {file_name}")
            continue

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        overlay = image_np.copy()

        for idx, prompt in enumerate(prompts):
            label = prompt.get("label", "sin_clase")
            box = prompt.get("box")
            if box is None:
                raise ValueError("Este ejemplo espera prompts con 'box': [x1, y1, x2, y2].")

            inputs = processor(images=image, input_boxes=[[box]], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)
                pred = outputs.pred_masks
                while pred.ndim > 3:
                    pred = pred.squeeze(1)
                pred = torch.sigmoid(pred.unsqueeze(1))
                pred = F.interpolate(pred, size=(image.height, image.width), mode="bilinear", align_corners=False)
                mask = (pred.squeeze().cpu().numpy() >= threshold).astype(np.uint8)

            mask_path = output_dir / "masks" / Path(file_name).stem / f"{idx:03d}_{label}.png"
            save_mask_png(mask, mask_path)
            overlay = overlay_mask(overlay, mask, alpha=0.35)
            overlay = draw_box(overlay, box, label)

            results.append(
                {
                    "file_name": file_name,
                    "label": label,
                    "box": box,
                    "mask_path": str(mask_path),
                    "mask_area_pixels": int(mask.sum()),
                }
            )

        overlay_path = output_dir / "overlays" / f"{Path(file_name).stem}_overlay.png"
        ensure_dir(overlay_path.parent)
        Image.fromarray(overlay).save(overlay_path)

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
