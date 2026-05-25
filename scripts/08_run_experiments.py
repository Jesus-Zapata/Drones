from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta la tabla experimental E1-E5 del proyecto SAM.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="IDs a ejecutar, por ejemplo: --only E1 E2")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime comandos.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_overall_metrics(output_dir: str | Path) -> dict:
    path = Path(output_dir) / "evaluation_metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("overall", {})


def flatten_summary_row(exp: dict[str, Any], output_dir: str | Path, suffix: str = "") -> dict[str, Any]:
    overall = read_overall_metrics(output_dir)
    row = {
        "experiment_id": exp.get("id") + suffix,
        "description": exp.get("description", ""),
        "model": exp.get("model_name", exp.get("type", "")),
        "checkpoint": exp.get("checkpoint"),
        "prompt_type": exp.get("prompt_type", ""),
        "image_size": exp.get("image_size", ""),
        "mask_size": exp.get("mask_size", ""),
        "output_dir": str(output_dir),
    }
    row.update(overall)
    return row


def write_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "experiment_summary.csv"
    md_path = output_dir / "experiment_summary.md"

    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    display_cols = [
        "experiment_id",
        "description",
        "model",
        "prompt_type",
        "mean_iou",
        "miou_by_category",
        "mean_dice",
        "mean_mpa",
        "mean_pixel_accuracy",
        "mean_inference_time_ms",
        "mean_gpu_memory_mb",
        "instances",
    ]
    display_cols = [c for c in display_cols if any(c in r for r in rows)]

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(display_cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(display_cols)) + " |\n")
        for row in rows:
            values = []
            for col in display_cols:
                value = row.get(col, "")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")

    print(f"\nResumen CSV: {csv_path}")
    print(f"Resumen Markdown: {md_path}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    selected = set(args.only) if args.only else None
    rows: list[dict[str, Any]] = []

    for exp in cfg.get("experiments", []):
        exp_id = exp.get("id")
        if selected and exp_id not in selected:
            continue

        exp_type = exp.get("type", "sam")
        if exp_type == "baseline_coco_predictions":
            predictions = Path(exp.get("predictions", ""))
            output_dir = Path(exp.get("output_dir", f"outputs/experiments/{exp_id}"))
            if not predictions.exists():
                print(f"\n{exp_id}: no se ejecuta porque no existe {predictions}")
                rows.append({
                    "experiment_id": exp_id,
                    "description": exp.get("description", ""),
                    "model": exp_type,
                    "status": "skipped_missing_predictions",
                    "output_dir": str(output_dir),
                })
                continue

            cmd = [
                sys.executable,
                "scripts/09_evaluate_baseline_detection.py",
                "--annotations",
                exp["annotations"],
                "--predictions",
                str(predictions),
                "--output-dir",
                str(output_dir),
            ]
            run_command(cmd, args.dry_run)
            rows.append(flatten_summary_row(exp, output_dir))
            continue

        corruptions = exp.get("corruptions")
        if corruptions:
            for corruption in corruptions:
                name = corruption["name"]
                severity = int(corruption.get("severity", 3))
                output_dir = Path(exp["output_dir"]) / f"{name}_s{severity}"
                cmd = [
                    sys.executable,
                    "scripts/04_evaluate_sam.py",
                    "--config",
                    args.config,
                    "--model-name",
                    exp.get("model_name", cfg["model"]["pretrained_name"]),
                    "--annotations",
                    exp["annotations"],
                    "--prompt-type",
                    exp.get("prompt_type", "box"),
                    "--image-size",
                    str(exp.get("image_size", 1024)),
                    "--mask-size",
                    str(exp.get("mask_size", 256)),
                    "--batch-size",
                    str(exp.get("batch_size", cfg.get("evaluation", {}).get("batch_size", 1))),
                    "--output-dir",
                    str(output_dir),
                    "--corruption",
                    name,
                    "--corruption-severity",
                    str(severity),
                    "--save-visualizations",
                ]
                if exp.get("checkpoint"):
                    cmd.extend(["--checkpoint", exp["checkpoint"]])
                run_command(cmd, args.dry_run)
                rows.append(flatten_summary_row(exp, output_dir, suffix=f"_{name}_s{severity}"))
            continue

        output_dir = Path(exp["output_dir"])
        cmd = [
            sys.executable,
            "scripts/04_evaluate_sam.py",
            "--config",
            args.config,
            "--model-name",
            exp.get("model_name", cfg["model"]["pretrained_name"]),
            "--annotations",
            exp["annotations"],
            "--prompt-type",
            exp.get("prompt_type", "box"),
            "--image-size",
            str(exp.get("image_size", 1024)),
            "--mask-size",
            str(exp.get("mask_size", 256)),
            "--batch-size",
            str(exp.get("batch_size", cfg.get("evaluation", {}).get("batch_size", 1))),
            "--output-dir",
            str(output_dir),
            "--save-visualizations",
        ]
        if exp.get("checkpoint"):
            cmd.extend(["--checkpoint", exp["checkpoint"]])
        run_command(cmd, args.dry_run)
        rows.append(flatten_summary_row(exp, output_dir))

    write_table(rows, Path("outputs/experiments"))


if __name__ == "__main__":
    main()
