import argparse
from pathlib import Path

import pandas as pd


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"No se encontró ninguna de estas columnas: {candidates}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def summarize_experiment(exp_dir: Path) -> pd.DataFrame | None:
    rows_path = exp_dir / "evaluation_rows.csv"

    if not rows_path.exists():
        return None

    df = pd.read_csv(rows_path)

    category_col = pick_column(
        df,
        ["category_name", "category", "class_name", "label", "class"],
    )

    iou_col = pick_column(df, ["iou", "mean_iou"])
    dice_col = pick_column(df, ["dice", "mean_dice"])

    optional_cols = {
        "mpa": ["mpa", "mean_mpa"],
        "pixel_accuracy": ["pixel_accuracy", "mean_pixel_accuracy"],
        "inference_time_ms": ["inference_time_ms", "mean_inference_time_ms"],
        "gpu_memory_mb": ["gpu_memory_mb", "mean_gpu_memory_mb"],
    }

    agg_dict = {
        iou_col: ["count", "mean", "std", "min", "max"],
        dice_col: ["mean", "std", "min", "max"],
    }

    selected_optional = {}

    for metric_name, candidates in optional_cols.items():
        for col in candidates:
            if col in df.columns:
                selected_optional[metric_name] = col
                agg_dict[col] = ["mean", "std"]
                break

    grouped = df.groupby(category_col).agg(agg_dict).reset_index()

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    rename_map = {
        category_col: "category",
        f"{iou_col}_count": "instances",
        f"{iou_col}_mean": "mean_iou",
        f"{iou_col}_std": "std_iou",
        f"{iou_col}_min": "min_iou",
        f"{iou_col}_max": "max_iou",
        f"{dice_col}_mean": "mean_dice",
        f"{dice_col}_std": "std_dice",
        f"{dice_col}_min": "min_dice",
        f"{dice_col}_max": "max_dice",
    }

    for metric_name, col in selected_optional.items():
        rename_map[f"{col}_mean"] = f"mean_{metric_name}"
        rename_map[f"{col}_std"] = f"std_{metric_name}"

    grouped = grouped.rename(columns=rename_map)
    grouped.insert(0, "experiment_id", exp_dir.name)

    numeric_cols = grouped.select_dtypes(include=["float", "float64", "float32"]).columns
    grouped[numeric_cols] = grouped[numeric_cols].round(4)

    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-dir",
        default="outputs/experiments",
        help="Carpeta donde están los resultados por experimento.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/experiments/experiment_summary_by_class.csv",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/experiments/experiment_summary_by_class.md",
    )

    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    summaries = []

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        summary = summarize_experiment(exp_dir)

        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise RuntimeError(
            f"No se encontraron archivos evaluation_rows.csv en {experiments_dir}"
        )

    final_df = pd.concat(summaries, ignore_index=True)

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(output_csv, index=False)
    output_md.write_text(final_df.to_markdown(index=False), encoding="utf-8")

    print(f"Resumen por clase guardado en: {output_csv}")
    print(f"Resumen markdown guardado en: {output_md}")


if __name__ == "__main__":
    main()