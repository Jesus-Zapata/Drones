from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import SamProcessor

from sam_electric.coco import category_maps, xywh_to_xyxy
from sam_electric.corruptions import apply_corruption


@dataclass
class InstanceRecord:
    image_id: int
    annotation_id: int
    image_path: Path
    file_name: str
    width: int
    height: int
    bbox_xyxy: List[float]
    category_id: int
    category_name: str


def _resize_binary_mask(mask: np.ndarray, size: int) -> np.ndarray:
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    mask_img = mask_img.resize((int(size), int(size)), resample=Image.Resampling.NEAREST)
    return (np.array(mask_img) > 0).astype(np.float32)


def _positive_point_from_mask(mask: np.ndarray, bbox_xyxy: List[float]) -> List[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        return [float(np.median(xs)), float(np.median(ys))]

    x1, y1, x2, y2 = bbox_xyxy
    return [float((x1 + x2) / 2), float((y1 + y2) / 2)]


class COCOSAMDataset(Dataset):
    """Dataset de instancias COCO para fine-tuning y evaluación de SAM.

    Cada muestra corresponde a una instancia anotada. Soporta prompts de caja y de punto.
    La máscara reducida se usa para pérdida; la máscara original se conserva para evaluación.
    """

    def __init__(
        self,
        annotation_path: str | Path,
        image_dir: str | Path,
        processor: SamProcessor,
        allowed_classes: Optional[List[str]] = None,
        prompt_type: str = "box",
        mask_size: int = 256,
        corruption: str | None = None,
        corruption_severity: int = 3,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.coco = COCO(str(self.annotation_path))
        self.id_to_name, self.name_to_id = category_maps(self.coco)
        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        self.prompt_type = prompt_type.lower()
        self.mask_size = int(mask_size)
        self.corruption = corruption
        self.corruption_severity = int(corruption_severity)

        if self.prompt_type not in {"box", "point"}:
            raise ValueError("prompt_type debe ser 'box' o 'point'.")

        self.records = self._build_records()
        if not self.records:
            raise ValueError(
                "No se encontraron anotaciones válidas. "
                "Revisa rutas, clases permitidas y archivo COCO."
            )

    def _build_records(self) -> List[InstanceRecord]:
        records: List[InstanceRecord] = []
        for ann_id in self.coco.getAnnIds():
            ann = self.coco.loadAnns([ann_id])[0]
            cat_name = self.id_to_name.get(ann["category_id"])

            if self.allowed_classes and cat_name not in self.allowed_classes:
                continue
            if ann.get("iscrowd", 0) == 1:
                continue
            if "segmentation" not in ann or "bbox" not in ann:
                continue

            image = self.coco.loadImgs([ann["image_id"]])[0]
            image_path = self.image_dir / image["file_name"]
            if not image_path.exists():
                continue

            records.append(
                InstanceRecord(
                    image_id=image["id"],
                    annotation_id=ann_id,
                    image_path=image_path,
                    file_name=image["file_name"],
                    width=image["width"],
                    height=image["height"],
                    bbox_xyxy=xywh_to_xyxy(ann["bbox"]),
                    category_id=ann["category_id"],
                    category_name=cat_name,
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        image = apply_corruption(image, self.corruption, self.corruption_severity)

        ann = self.coco.loadAnns([record.annotation_id])[0]
        mask_original = self.coco.annToMask(ann).astype(np.uint8)
        mask_resized = _resize_binary_mask(mask_original, self.mask_size)

        if self.prompt_type == "box":
            inputs = self.processor(
                images=image,
                input_boxes=[[record.bbox_xyxy]],
                return_tensors="pt",
            )
        else:
            point = _positive_point_from_mask(mask_original, record.bbox_xyxy)
            inputs = self.processor(
                images=image,
                input_points=[[[point]]],
                input_labels=[[[1]]],
                return_tensors="pt",
            )

        item = {k: v.squeeze(0) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        item["ground_truth_mask"] = torch.from_numpy(mask_resized).float()
        item["ground_truth_mask_original"] = torch.from_numpy(mask_original.astype(np.float32))
        item["category_id"] = torch.tensor(record.category_id, dtype=torch.long)
        item["annotation_id"] = torch.tensor(record.annotation_id, dtype=torch.long)
        item["image_id"] = torch.tensor(record.image_id, dtype=torch.long)
        item["bbox_xyxy"] = torch.tensor(record.bbox_xyxy, dtype=torch.float32)
        item["category_name"] = record.category_name
        item["file_name"] = record.file_name
        item["original_height"] = record.height
        item["original_width"] = record.width
        return item


def collate_sam_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = [
        "pixel_values",
        "input_boxes",
        "input_points",
        "input_labels",
        "original_sizes",
        "reshaped_input_sizes",
        "ground_truth_mask",
        "category_id",
        "annotation_id",
        "image_id",
        "bbox_xyxy",
    ]

    output: Dict[str, Any] = {}
    for key in tensor_keys:
        if key in batch[0]:
            output[key] = torch.stack([sample[key] for sample in batch])

    output["ground_truth_mask_original"] = [sample["ground_truth_mask_original"] for sample in batch]
    output["category_name"] = [sample["category_name"] for sample in batch]
    output["file_name"] = [sample["file_name"] for sample in batch]
    output["original_height"] = [sample["original_height"] for sample in batch]
    output["original_width"] = [sample["original_width"] for sample in batch]
    return output
