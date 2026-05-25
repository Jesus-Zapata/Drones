from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO

from sam_electric.coco import xywh_to_xyxy
from sam_electric.metrics import SegmentationMetricAccumulator
from sam_electric.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa un baseline de detección/clasificación con cajas.")
    parser.add_argument("--annotations", required=True, help="Archivo COCO ground truth.")
    parser.add_argument("--predictions", required=True, help="JSON con predicciones tipo COCO detection.")
    parser.add_argument("--output-dir", default="outputs/experiments/E5_baseline")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    return parser.parse_args()


def box_iou_xyxy(a: list[float], b: list[float], eps: float = 1e-7) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float((inter + eps) / (union + eps))


def normalize_pred_box(pred: dict[str, Any]) -> list[float]:
    bbox = [float(v) for v in pred["bbox"]]
    fmt = pred.get("bbox_format", "xywh").lower()
    if fmt == "xyxy":
        return bbox
    return xywh_to_xyxy(bbox)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    coco = COCO(args.annotations)
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    with Path(args.predictions).open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    if isinstance(predictions, dict):
        predictions = predictions.get("annotations", predictions.get("predictions", []))

    predictions = [p for p in predictions if float(p.get("score", 1.0)) >= args.score_threshold]

    gt_by_image_cat: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for ann_id in coco.getAnnIds():
        ann = coco.loadAnns([ann_id])[0]
        gt_by_image_cat[(ann["image_id"], ann["category_id"])].append(
            {
                "ann_id": ann_id,
                "bbox_xyxy": xywh_to_xyxy(ann["bbox"]),
                "matched": False,
            }
        )

    pred_by_image_cat: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        pred_by_image_cat[(int(pred["image_id"]), int(pred["category_id"]))].append(pred)

    rows = []
    matched_ious_by_cat = defaultdict(list)
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    keys = set(gt_by_image_cat) | set(pred_by_image_cat)
    for key in keys:
        image_id, category_id = key
        gt_items = gt_by_image_cat.get(key, [])
        pred_items = sorted(pred_by_image_cat.get(key, []), key=lambda p: float(p.get("score", 1.0)), reverse=True)
        category_name = cats.get(category_id, str(category_id))

        for pred in pred_items:
            pred_box = normalize_pred_box(pred)
            best_iou = 0.0
            best_gt = None
            for gt in gt_items:
                if gt["matched"]:
                    continue
                iou = box_iou_xyxy(pred_box, gt["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt is not None and best_iou >= args.iou_threshold:
                best_gt["matched"] = True
                counts[category_name]["tp"] += 1
                matched_ious_by_cat[category_name].append(best_iou)
                result = "TP"
            else:
                counts[category_name]["fp"] += 1
                result = "FP"

            rows.append(
                {
                    "image_id": image_id,
                    "category_name": category_name,
                    "score": float(pred.get("score", 1.0)),
                    "best_iou": best_iou,
                    "result": result,
                }
            )

        for gt in gt_items:
            if not gt["matched"]:
                counts[category_name]["fn"] += 1
                rows.append(
                    {
                        "image_id": image_id,
                        "category_name": category_name,
                        "score": "",
                        "best_iou": 0.0,
                        "result": "FN",
                    }
                )

    by_category = {}
    overall_tp = overall_fp = overall_fn = 0
    for cat, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-7)
        by_category[cat] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_detection_iou": float(np.mean(matched_ious_by_cat[cat])) if matched_ious_by_cat[cat] else 0.0,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    overall_precision = overall_tp / max(overall_tp + overall_fp, 1)
    overall_recall = overall_tp / max(overall_tp + overall_fn, 1)
    overall_f1 = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-7)
    all_matched_ious = [iou for values in matched_ious_by_cat.values() for iou in values]

    summary = {
        "overall": {
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
            "mean_iou": float(np.mean(all_matched_ious)) if all_matched_ious else 0.0,
            "mean_precision": float(overall_precision),
            "mean_recall": float(overall_recall),
            "mean_f1": float(overall_f1),
            "instances": int(overall_tp + overall_fn),
            "predictions": int(overall_tp + overall_fp),
        },
        "by_category": by_category,
        "metadata": {
            "annotations": args.annotations,
            "predictions": args.predictions,
            "iou_threshold": args.iou_threshold,
            "score_threshold": args.score_threshold,
            "note": "Baseline de detección: las métricas no son equivalentes a segmentación por máscara.",
        },
        "rows": rows,
    }

    metrics_path = output_dir / "evaluation_metrics.json"
    rows_path = output_dir / "evaluation_rows.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "category_name", "score", "best_iou", "result"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
