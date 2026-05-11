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


class COCOSAMDataset(Dataset):
    """Dataset de instancias COCO para fine-tuning de SAM.

    Cada muestra corresponde a una instancia anotada, no a una imagen completa.
    Se usa la caja de la anotación como prompt y la máscara COCO como objetivo.
    """

    def __init__(
        self,
        annotation_path: str | Path,
        image_dir: str | Path,
        processor: SamProcessor,
        allowed_classes: Optional[List[str]] = None,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.coco = COCO(str(self.annotation_path))
        self.id_to_name, self.name_to_id = category_maps(self.coco)
        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        self.records = self._build_records()

        if not self.records:
            raise ValueError(
                "No se encontraron anotaciones válidas. Revisa rutas, clases permitidas y archivo COCO."
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

            bbox_xyxy = xywh_to_xyxy(ann["bbox"])
            records.append(
                InstanceRecord(
                    image_id=image["id"],
                    annotation_id=ann_id,
                    image_path=image_path,
                    file_name=image["file_name"],
                    width=image["width"],
                    height=image["height"],
                    bbox_xyxy=bbox_xyxy,
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
        ann = self.coco.loadAnns([record.annotation_id])[0]
        mask = self.coco.annToMask(ann).astype(np.float32)

        # Hugging Face SAM espera cajas en formato XYXY, agrupadas por imagen y por prompt.
        inputs = self.processor(
            images=image,
            input_boxes=[[record.bbox_xyxy]],
            return_tensors="pt",
        )

        # Quitamos la dimensión de batch creada por el processor. DataLoader la reconstruye.
        item = {k: v.squeeze(0) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        item["ground_truth_mask"] = torch.from_numpy(mask).float()
        item["category_id"] = torch.tensor(record.category_id, dtype=torch.long)
        item["annotation_id"] = torch.tensor(record.annotation_id, dtype=torch.long)
        item["image_id"] = torch.tensor(record.image_id, dtype=torch.long)
        item["category_name"] = record.category_name
        item["file_name"] = record.file_name
        return item


def collate_sam_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = [
        "pixel_values",
        "input_boxes",
        "original_sizes",
        "reshaped_input_sizes",
        "ground_truth_mask",
        "category_id",
        "annotation_id",
        "image_id",
    ]
    output: Dict[str, Any] = {}
    for key in tensor_keys:
        if key in batch[0]:
            output[key] = torch.stack([sample[key] for sample in batch])

    output["category_name"] = [sample["category_name"] for sample in batch]
    output["file_name"] = [sample["file_name"] for sample in batch]
    return output
