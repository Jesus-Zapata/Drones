from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
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
from sam_electric.metrics import bce_dice_loss, prepare_masks_for_loss
from sam_electric.processor import load_configured_processor
from sam_electric.utils import ensure_dir, get_device, load_config, set_seed, worker_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning de SAM para segmentación de elementos eléctricos.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None, help="Ruta a checkpoint guardado con save_pretrained.")
    parser.add_argument("--model-name", default=None, help="Modelo HF base, por ejemplo facebook/sam-vit-base.")
    parser.add_argument("--prompt-type", choices=["box", "point"], default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--mask-size", type=int, default=None)
    parser.add_argument("--num-workers", default=None, help="Número de workers del DataLoader. Usa auto, all o un entero.")
    parser.add_argument("--cpu-threads", default=None, help="Hilos CPU para Torch/OpenMP. Usa auto, all o un entero.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def freeze_parts(model: SamModel, freeze_vision_encoder: bool, freeze_prompt_encoder: bool) -> None:
    if freeze_vision_encoder:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
    if freeze_prompt_encoder:
        for param in model.prompt_encoder.parameters():
            param.requires_grad = False


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


def make_adamw_optimizer(params: list[torch.nn.Parameter], lr: float, weight_decay: float, device: torch.device, fused: bool) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    if fused and device.type == "cuda":
        try:
            return torch.optim.AdamW(params, fused=True, **kwargs)
        except TypeError:
            pass
        except RuntimeError:
            pass
    return torch.optim.AdamW(params, **kwargs)


def train_one_epoch(
    model: SamModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    mixed_precision: bool,
    amp_dtype: torch.dtype,
    gradient_accumulation_steps: int,
    channels_last: bool,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="Entrenando", leave=False), start=1):
        gt_masks = _move_tensor(batch["ground_truth_mask"], device)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
            outputs = model(**build_model_inputs(batch, device, channels_last=channels_last))
            pred_masks, gt_resized = prepare_masks_for_loss(outputs.pred_masks, gt_masks)
            loss = bce_dice_loss(pred_masks, gt_resized)
            loss = loss / max(gradient_accumulation_steps, 1)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % gradient_accumulation_steps == 0 or step == len(loader):
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu()) * max(gradient_accumulation_steps, 1)
        steps += 1

    return total_loss / max(steps, 1)


def evaluate_loss(
    model: SamModel,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validando", leave=False):
            gt_masks = _move_tensor(batch["ground_truth_mask"], device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
                outputs = model(**build_model_inputs(batch, device, channels_last=channels_last))
                pred_masks, gt_resized = prepare_masks_for_loss(outputs.pred_masks, gt_masks)
                loss = bce_dice_loss(pred_masks, gt_resized)
            total_loss += float(loss.detach().cpu())
            steps += 1

    return total_loss / max(steps, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    if args.num_workers is not None:
        cfg.setdefault("runtime", {})["num_workers"] = args.num_workers
    if args.cpu_threads is not None:
        cfg.setdefault("runtime", {})["cpu_threads"] = args.cpu_threads

    device = get_device()
    runtime = configure_torch_runtime(cfg, device)
    amp_dtype = autocast_dtype(runtime)

    model_name = args.resume or args.model_name or cfg["model"]["pretrained_name"]
    image_size = args.image_size or int(cfg["training"].get("image_size", cfg["model"].get("processor", {}).get("image_size", 1024)))
    mask_size = args.mask_size or int(cfg["training"].get("mask_size", cfg["model"].get("processor", {}).get("mask_size", 256)))
    prompt_type = args.prompt_type or str(cfg["training"].get("prompt_type", "box"))

    processor = load_configured_processor(model_name, image_size=image_size, mask_size=mask_size)
    model = SamModel.from_pretrained(model_name)
    freeze_parts(
        model,
        freeze_vision_encoder=bool(cfg["model"].get("freeze_vision_encoder", True)),
        freeze_prompt_encoder=bool(cfg["model"].get("freeze_prompt_encoder", True)),
    )

    model.to(device)
    if runtime.channels_last:
        model.to(memory_format=torch.channels_last)

    classes = cfg.get("classes")
    train_dataset = COCOSAMDataset(
        cfg["data"]["train_annotations"],
        cfg["data"]["image_dir"],
        processor,
        allowed_classes=classes,
        prompt_type=prompt_type,
        mask_size=mask_size,
    )
    val_dataset = COCOSAMDataset(
        cfg["data"]["val_annotations"],
        cfg["data"]["image_dir"],
        processor,
        allowed_classes=classes,
        prompt_type=prompt_type,
        mask_size=mask_size,
    )

    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    epochs = args.epochs or int(cfg["training"]["epochs"])
    lr = args.lr or float(cfg["training"]["learning_rate"])
    mixed_precision = bool(cfg["training"].get("mixed_precision", True)) and runtime.mixed_precision
    gradient_accumulation_steps = int(cfg["training"].get("gradient_accumulation_steps", 1))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_sam_batch,
        **dataloader_kwargs(runtime, shuffle=True, worker_init_fn=worker_seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_sam_batch,
        **dataloader_kwargs(runtime, shuffle=False),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_adamw_optimizer(
        trainable_params,
        lr=lr,
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
        device=device,
        fused=bool(cfg.get("runtime", {}).get("fused_optimizer", True)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and device.type == "cuda")

    output_dir = ensure_dir(args.output_dir or cfg["training"]["output_dir"])
    history = []
    best_val = float("inf")

    run_metadata = {
        "model_name": model_name,
        "prompt_type": prompt_type,
        "image_size": image_size,
        "mask_size": mask_size,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * max(gradient_accumulation_steps, 1),
        "epochs": epochs,
        "learning_rate": lr,
        "device": str(device),
        "train_instances": len(train_dataset),
        "val_instances": len(val_dataset),
        "trainable_parameters": int(sum(p.numel() for p in trainable_params)),
        "runtime": json.loads(hardware_report(runtime)),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_metadata, ensure_ascii=False, indent=2))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            mixed_precision,
            amp_dtype,
            gradient_accumulation_steps,
            runtime.channels_last,
        )
        val_loss = evaluate_loss(model, val_loader, device, mixed_precision, amp_dtype, runtime.channels_last)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if epoch % int(cfg["training"].get("save_every_epochs", 1)) == 0:
            epoch_dir = output_dir / f"epoch_{epoch:03d}"
            model.save_pretrained(epoch_dir)
            processor.save_pretrained(epoch_dir)

        if val_loss < best_val:
            best_val = val_loss
            best_dir = output_dir / "best"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if bool(cfg.get("runtime", {}).get("empty_cache_each_epoch", True)) and device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Mejor checkpoint guardado en: {output_dir / 'best'}")


if __name__ == "__main__":
    main()
