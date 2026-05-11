from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_electric.coco import validate_coco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida un archivo COCO segmentation.")
    parser.add_argument("--annotations", required=True, help="Ruta al JSON COCO.")
    parser.add_argument("--image-dir", default=None, help="Carpeta de imágenes referenciadas por file_name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_coco(args.annotations, args.image_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["num_errors"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
