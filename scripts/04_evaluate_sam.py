from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from sam_electric.dataset import COCOSAMDataset, collate_sam_batch
from sam_electric.metrics import SegmentationMetricAccumulator, binary_dice, binary_iou, prepare_masks_for_loss
from sam_electric.utils import ensure_dir, get_device, load_config
from sam_electric.visualization import save_side_by_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa un checkpoint SAM ajustado.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Carpeta del modelo. Por defecto usa training.output_dir/best.")
    parser.add_argument("--annotations", default=None, help="COCO de evaluación. Por defecto usa val_annotations.")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--max-visualizations", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(cfg["training"]["output_dir"]) / "best")
    annotations = args.annotations or cfg["data"]["val_annotations"]
    image_dir = cfg["data"]["image_dir"]
    threshold = float(cfg["training"].get("threshold", 0.5))

    device = get_device()
    processor = SamProcessor.from_pretrained(checkpoint)
    model = SamModel.from_pretrained(checkpoint).to(device)
    model.eval()

    dataset = COCOSAMDataset(annotations, image_dir, processor, allowed_classes=cfg.get("classes"))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 2)),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        collate_fn=collate_sam_batch,
    )

    accumulator = SegmentationMetricAccumulator()
    vis_dir = ensure_dir(Path("outputs") / "evaluation_visualizations")
    saved = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluando"):
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            gt_masks = batch["ground_truth_mask"].to(device)

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred_logits, _ = prepare_masks_for_loss(outputs.pred_masks, gt_masks)

            # Para métricas visuales se escala la predicción al tamaño original de cada máscara GT.
            for i in range(pred_logits.shape[0]):
                gt = gt_masks[i].detach().cpu().numpy().astype(np.uint8)
                pred = torch.sigmoid(pred_logits[i : i + 1].unsqueeze(1))
                pred = F.interpolate(pred, size=gt.shape, mode="bilinear", align_corners=False)
                pred_np = (pred.squeeze().detach().cpu().numpy() >= threshold).astype(np.uint8)

                iou = binary_iou(pred_np, gt)
                dice = binary_dice(pred_np, gt)
                category = batch["category_name"][i]
                file_name = batch["file_name"][i]
                accumulator.update(category, iou, dice, file_name)

                if args.save_visualizations and saved < args.max_visualizations:
                    image = np.array(Image.open(Path(image_dir) / file_name).convert("RGB"))
                    out = vis_dir / f"{saved:04d}_{Path(file_name).stem}_{category}.png"
                    save_side_by_side(image, gt, pred_np, out, title=f"{category} | IoU={iou:.3f} | Dice={dice:.3f}")
                    saved += 1

    summary = accumulator.summary()
    out_path = Path("outputs") / "evaluation_metrics.json"
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Métricas guardadas en: {out_path}")


if __name__ == "__main__":
    main()
