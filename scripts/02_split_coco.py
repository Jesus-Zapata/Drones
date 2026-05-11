from __future__ import annotations

import argparse

from sam_electric.coco import split_coco_by_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Divide un COCO JSON por imágenes en train/val/test.")
    parser.add_argument("--annotations", required=True, help="JSON COCO original.")
    parser.add_argument("--output-dir", default="data/coco/annotations", help="Carpeta de salida.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = split_coco_by_image(
        annotation_path=args.annotations,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print("Archivos generados:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
