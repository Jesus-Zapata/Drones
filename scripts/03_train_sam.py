from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from sam_electric.dataset import COCOSAMDataset, collate_sam_batch
from sam_electric.metrics import bce_dice_loss, prepare_masks_for_loss
from sam_electric.utils import ensure_dir, get_device, load_config, set_seed, worker_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning de SAM para segmentación de elementos eléctricos.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None, help="Ruta a un checkpoint guardado con save_pretrained.")
    return parser.parse_args()


def freeze_parts(model: SamModel, freeze_vision_encoder: bool, freeze_prompt_encoder: bool) -> None:
    if freeze_vision_encoder:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
    if freeze_prompt_encoder:
        for param in model.prompt_encoder.parameters():
            param.requires_grad = False


def train_one_epoch(model, loader, optimizer, scaler, device, mixed_precision: bool) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in tqdm(loader, desc="Entrenando", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        gt_masks = batch["ground_truth_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=mixed_precision and device.type == "cuda"):
            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False,
            )
            pred_masks, gt_resized = prepare_masks_for_loss(outputs.pred_masks, gt_masks)
            loss = bce_dice_loss(pred_masks, gt_resized)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        steps += 1

    return total_loss / max(steps, 1)


def evaluate_loss(model, loader, device, mixed_precision: bool) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validando", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            gt_masks = batch["ground_truth_mask"].to(device)
            with torch.cuda.amp.autocast(enabled=mixed_precision and device.type == "cuda"):
                outputs = model(
                    pixel_values=pixel_values,
                    input_boxes=input_boxes,
                    multimask_output=False,
                )
                pred_masks, gt_resized = prepare_masks_for_loss(outputs.pred_masks, gt_masks)
                loss = bce_dice_loss(pred_masks, gt_resized)
            total_loss += float(loss.detach().cpu())
            steps += 1
    return total_loss / max(steps, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    model_name = cfg["model"]["pretrained_name"]
    if args.resume:
        model_name = args.resume

    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name)
    freeze_parts(
        model,
        freeze_vision_encoder=bool(cfg["model"].get("freeze_vision_encoder", True)),
        freeze_prompt_encoder=bool(cfg["model"].get("freeze_prompt_encoder", True)),
    )

    device = get_device()
    model.to(device)

    train_ann = cfg["data"]["train_annotations"]
    val_ann = cfg["data"]["val_annotations"]
    image_dir = cfg["data"]["image_dir"]
    classes = cfg.get("classes")

    train_dataset = COCOSAMDataset(train_ann, image_dir, processor, allowed_classes=classes)
    val_dataset = COCOSAMDataset(val_ann, image_dir, processor, allowed_classes=classes)

    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    epochs = args.epochs or int(cfg["training"]["epochs"])
    lr = args.lr or float(cfg["training"]["learning_rate"])
    num_workers = int(cfg["training"].get("num_workers", 0))
    mixed_precision = bool(cfg["training"].get("mixed_precision", True))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_sam_batch,
        worker_init_fn=worker_seed,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_sam_batch,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and device.type == "cuda")

    output_dir = ensure_dir(cfg["training"]["output_dir"])
    history = []
    best_val = float("inf")

    print(f"Device: {device}")
    print(f"Instancias train: {len(train_dataset)} | val: {len(val_dataset)}")
    print(f"Parámetros entrenables: {sum(p.numel() for p in trainable_params):,}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, mixed_precision)
        val_loss = evaluate_loss(model, val_loader, device, mixed_precision)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        epoch_dir = output_dir / f"epoch_{epoch:03d}"
        if epoch % int(cfg["training"].get("save_every_epochs", 1)) == 0:
            model.save_pretrained(epoch_dir)
            processor.save_pretrained(epoch_dir)

        if val_loss < best_val:
            best_val = val_loss
            best_dir = output_dir / "best"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Mejor checkpoint guardado en: {output_dir / 'best'}")


if __name__ == "__main__":
    main()
