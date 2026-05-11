from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from pycocotools.coco import COCO


def load_coco_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def category_maps(coco: COCO) -> Tuple[Dict[int, str], Dict[str, int]]:
    cats = coco.loadCats(coco.getCatIds())
    id_to_name = {c["id"]: c["name"] for c in cats}
    name_to_id = {c["name"]: c["id"] for c in cats}
    return id_to_name, name_to_id


def xywh_to_xyxy(bbox: Iterable[float]) -> List[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def mask_to_bbox_xywh(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def validate_coco(annotation_path: str | Path, image_dir: str | Path | None = None) -> dict:
    annotation_path = Path(annotation_path)
    data = load_coco_json(annotation_path)

    required_top = {"images", "annotations", "categories"}
    missing_top = required_top - set(data.keys())
    if missing_top:
        raise ValueError(f"Faltan llaves COCO obligatorias: {sorted(missing_top)}")

    coco = COCO(str(annotation_path))
    id_to_name, _ = category_maps(coco)
    ann_ids = coco.getAnnIds()
    img_ids = set(coco.getImgIds())

    errors = []
    warnings = []
    counts_by_cat = Counter()
    empty_masks = 0

    if image_dir is not None:
        image_dir = Path(image_dir)
        for image in data["images"]:
            if not (image_dir / image["file_name"]).exists():
                warnings.append(f"No existe imagen: {image['file_name']}")

    for ann_id in ann_ids:
        ann = coco.loadAnns([ann_id])[0]
        if ann.get("image_id") not in img_ids:
            errors.append(f"Anotación {ann_id}: image_id no existe")
        if ann.get("category_id") not in id_to_name:
            errors.append(f"Anotación {ann_id}: category_id no existe")
        if "bbox" not in ann:
            errors.append(f"Anotación {ann_id}: falta bbox")
        if "segmentation" not in ann:
            errors.append(f"Anotación {ann_id}: falta segmentation")
            continue
        try:
            mask = coco.annToMask(ann)
            if mask.sum() == 0:
                empty_masks += 1
                warnings.append(f"Anotación {ann_id}: máscara vacía")
        except Exception as exc:
            errors.append(f"Anotación {ann_id}: no se pudo decodificar máscara: {exc}")
        counts_by_cat[id_to_name.get(ann.get("category_id"), "desconocida")] += 1

    return {
        "num_images": len(data["images"]),
        "num_annotations": len(data["annotations"]),
        "num_categories": len(data["categories"]),
        "annotations_by_category": dict(counts_by_cat),
        "empty_masks": empty_masks,
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors[:50],
        "warnings": warnings[:50],
    }


def make_coco_subset(data: dict, image_ids: set[int]) -> dict:
    subset = copy.deepcopy(data)
    subset["images"] = [img for img in data["images"] if img["id"] in image_ids]
    subset["annotations"] = [ann for ann in data["annotations"] if ann["image_id"] in image_ids]
    return subset


def split_coco_by_image(
    annotation_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Path, Path, Path]:
    import random

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Usa proporciones válidas: train_ratio > 0, val_ratio >= 0 y train+val < 1")

    data = load_coco_json(annotation_path)
    image_ids = [img["id"] for img in data["images"]]
    random.Random(seed).shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train : n_train + n_val])
    test_ids = set(image_ids[n_train + n_val :])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        out = output_dir / f"instances_{name}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(make_coco_subset(data, ids), f, ensure_ascii=False, indent=2)
        paths.append(out)
    return tuple(paths)  # type: ignore[return-value]


def annotation_stats(annotation_path: str | Path) -> dict:
    coco = COCO(str(annotation_path))
    id_to_name, _ = category_maps(coco)
    by_cat = Counter()
    by_image = defaultdict(int)

    for ann_id in coco.getAnnIds():
        ann = coco.loadAnns([ann_id])[0]
        by_cat[id_to_name[ann["category_id"]]] += 1
        by_image[ann["image_id"]] += 1

    return {
        "images": len(coco.getImgIds()),
        "annotations": len(coco.getAnnIds()),
        "categories": dict(by_cat),
        "avg_instances_per_image": float(np.mean(list(by_image.values()))) if by_image else 0.0,
    }
