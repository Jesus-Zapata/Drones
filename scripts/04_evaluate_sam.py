from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel

from sam_electric.dataset import COCOSAMDataset, collate_sam_batch
from sam_electric.hardware import (
    autocast_dtype,
    configure_torch_runtime,
    dataloader_kwargs,
    hardware_report,
)
from sam_electric.metrics import (
    SegmentationMetricAccumulator,
    binary_confusion,
    metrics_from_confusion,
    prepare_masks_for_loss,
)
from sam_electric.processor import load_configured_processor
from sam_electric.utils import ensure_dir, get_device, load_config
from sam_electric.visualization import save_side_by_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa un checkpoint SAM o un SAM base.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Carpeta del modelo fine-tuned. Si no se entrega, usa --model-name.")
    parser.add_argument("--model-name", default=None, help="Modelo base HF. Por defecto usa model.pretrained_name del config.")
    parser.add_argument("--annotations", default=None, help="COCO de evaluación. Por defecto usa evaluation.annotations o test_annotations.")
    parser.add_argument("--prompt-type", choices=["box", "point"], default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--mask-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", default=None, help="Número de workers del DataLoader. Usa auto, all o un entero.")
    parser.add_argument("--cpu-threads", default=None, help="Hilos CPU para Torch/OpenMP. Usa auto, all o un entero.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--max-visualizations", type=int, default=None)
    parser.add_argument("--corruption", default=None, choices=[None, "none", "blur", "gaussian_noise", "jpeg_compression"])
    parser.add_argument("--corruption-severity", type=int, default=3)
    return parser.parse_args()


def _move_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=(device.type == "cuda"))


def build_model_inputs(
    batch: dict,
    device: torch.device,
    channels_last: bool = False,
) -> dict[str, Any]:
    pixel_values = _move_tensor(batch["pixel_values"], device)
    if channels_last and pixel_values.ndim == 4:
        pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)

    inputs: dict[str, Any] = {"pixel_values": pixel_values, "multimask_output": False}
    if "input_boxes" in batch:
        inputs["input_boxes"] = _move_tensor(batch["input_boxes"], device)
    if "input_points" in batch:
        inputs["input_points"] = _move_tensor(batch["input_points"], device)
    if "input_labels" in batch:
        inputs["input_labels"] = _move_tensor(batch["input_labels"], device)
    return inputs


def _save_rows_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.num_workers is not None:
        cfg.setdefault("runtime", {})["num_workers"] = args.num_workers
    if args.cpu_threads is not None:
        cfg.setdefault("runtime", {})["cpu_threads"] = args.cpu_threads

    eval_cfg = cfg.get("evaluation", {})
    model_name_or_path = args.checkpoint or args.model_name or cfg["model"]["pretrained_name"]
    annotations = args.annotations or eval_cfg.get("annotations") or cfg["data"]["test_annotations"]
    image_dir = cfg["data"]["image_dir"]
    prompt_type = args.prompt_type or eval_cfg.get("prompt_type", "box")
    image_size = args.image_size or int(eval_cfg.get("image_size", cfg["model"].get("processor", {}).get("image_size", 1024)))
    mask_size = args.mask_size or int(eval_cfg.get("mask_size", cfg["model"].get("processor", {}).get("mask_size", 256)))
    threshold = args.threshold if args.threshold is not None else float(eval_cfg.get("threshold", 0.5))
    batch_size = args.batch_size or int(eval_cfg.get("batch_size", cfg["training"].get("batch_size", 1)))
    output_dir = ensure_dir(args.output_dir or eval_cfg.get("output_dir", "outputs/evaluation"))
    save_visualizations = bool(args.save_visualizations or eval_cfg.get("save_visualizations", False))
    max_visualizations = args.max_visualizations or int(eval_cfg.get("max_visualizations", 25))

    device = get_device()
    runtime = configure_torch_runtime(cfg, device)
    amp_dtype = autocast_dtype(runtime)

    processor = load_configured_processor(model_name_or_path, image_size=image_size, mask_size=mask_size)
    model = SamModel.from_pretrained(model_name_or_path).to(device)
    if runtime.channels_last:
        model.to(memory_format=torch.channels_last)
    model.eval()

    dataset = COCOSAMDataset(
        annotations,
        image_dir,
        processor,
        allowed_classes=cfg.get("classes"),
        prompt_type=prompt_type,
        mask_size=mask_size,
        corruption=args.corruption,
        corruption_severity=args.corruption_severity,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_sam_batch,
        **dataloader_kwargs(runtime, shuffle=False),
    )

    accumulator = SegmentationMetricAccumulator()
    vis_dir = ensure_dir(output_dir / "visualizations")
    saved = 0

    metadata = {
        "model_name_or_path": model_name_or_path,
        "annotations": annotations,
        "prompt_type": prompt_type,
        "image_size": image_size,
        "mask_size": mask_size,
        "threshold": threshold,
        "batch_size": batch_size,
        "corruption": args.corruption or "none",
        "corruption_severity": args.corruption_severity,
        "device": str(device),
        "instances": len(dataset),
        "runtime": json.loads(hardware_report(runtime)),
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluando"):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=runtime.mixed_precision):
                outputs = model(**build_model_inputs(batch, device, channels_last=runtime.channels_last))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            memory_mb = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            )

            pred_logits, _ = prepare_masks_for_loss(outputs.pred_masks, _move_tensor(batch["ground_truth_mask"], device))
            batch_n = pred_logits.shape[0]
            per_instance_ms = elapsed_ms / max(batch_n, 1)

            for i in range(batch_n):
                gt = batch["ground_truth_mask_original"][i].detach().cpu().numpy().astype(np.uint8)
                pred = torch.sigmoid(pred_logits[i : i + 1].unsqueeze(1))
                pred = F.interpolate(pred, size=gt.shape, mode="bilinear", align_corners=False)
                pred_np = (pred.squeeze().detach().cpu().numpy() >= threshold).astype(np.uint8)

                metrics = metrics_from_confusion(binary_confusion(pred_np, gt))
                accumulator.update(
                    category_name=batch["category_name"][i],
                    file_name=batch["file_name"][i],
                    metrics=metrics,
                    inference_time_ms=per_instance_ms,
                    gpu_memory_mb=memory_mb,
                )

                if save_visualizations and saved < max_visualizations:
                    image = np.array(Image.open(Path(image_dir) / batch["file_name"][i]).convert("RGB"))
                    out = vis_dir / f"{saved:04d}_{Path(batch['file_name'][i]).stem}_{batch['category_name'][i]}.png"
                    save_side_by_side(
                        image,
                        gt,
                        pred_np,
                        out,
                        title=(
                            f"{batch['category_name'][i]} | "
                            f"IoU={metrics['iou']:.3f} | Dice={metrics['dice']:.3f} | "
                            f"mPA={metrics['mpa']:.3f}"
                        ),
                    )
                    saved += 1

    summary = accumulator.summary()
    summary["metadata"] = metadata

    metrics_path = output_dir / "evaluation_metrics.json"
    rows_path = output_dir / "evaluation_rows.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _save_rows_csv(summary.get("rows", []), rows_path)

    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Detalle por instancia guardado en: {rows_path}")


if __name__ == "__main__":
    main()
