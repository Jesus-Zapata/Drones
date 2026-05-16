import argparse
import json
from pathlib import Path


def has_valid_segmentation(annotation):
    segmentation = annotation.get("segmentation")

    if segmentation is None:
        return False

    if segmentation == []:
        return False

    if isinstance(segmentation, list):
        # COCO polygon: [[x1, y1, x2, y2, ...]]
        valid_polygons = []
        for polygon in segmentation:
            if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0:
                valid_polygons.append(polygon)

        annotation["segmentation"] = valid_polygons
        return len(valid_polygons) > 0

    if isinstance(segmentation, dict):
        # COCO RLE
        return "counts" in segmentation and "size" in segmentation

    return False


def normalize_bbox(annotation):
    bbox = annotation.get("bbox")

    if not bbox or len(bbox) != 4:
        return

    annotation["bbox"] = [float(value) for value in bbox]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Archivo COCO original")
    parser.add_argument("--output", required=True, help="Archivo COCO limpio")
    parser.add_argument("--fix-category-name", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    original_annotations = coco.get("annotations", [])

    clean_annotations = []
    removed_annotations = 0

    for ann in original_annotations:
        if not has_valid_segmentation(ann):
            removed_annotations += 1
            continue

        normalize_bbox(ann)
        clean_annotations.append(ann)

    coco["annotations"] = clean_annotations

    if args.fix_category_name:
        for category in coco.get("categories", []):
            if category.get("name") == "Tranformador":
                category["name"] = "Transformador"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print("Limpieza finalizada")
    print(f"Anotaciones originales: {len(original_annotations)}")
    print(f"Anotaciones conservadas: {len(clean_annotations)}")
    print(f"Anotaciones eliminadas: {removed_annotations}")
    print(f"Archivo generado: {output_path}")


if __name__ == "__main__":
    main()