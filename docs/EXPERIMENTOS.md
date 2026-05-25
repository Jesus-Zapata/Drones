# Flujo experimental E1-E5

Este ajuste deja el repositorio listo para generar la tabla experimental del documento escrito.

## 1. Entrenar el modelo fine-tuned

```bash
python scripts/03_train_sam.py --config configs/config.yaml
```

Por defecto entrena `facebook/sam-vit-base` con prompt de caja y guarda el mejor modelo en:

```text
outputs/checkpoints/sam-electric/best
```

Para bajar el consumo de memoria puedes reducir la resolución:

```bash
python scripts/03_train_sam.py --config configs/config.yaml --image-size 768 --batch-size 1
```

Para probar otro modelo:

```bash
python scripts/03_train_sam.py --config configs/config.yaml --model-name facebook/sam-vit-large --image-size 1024 --batch-size 1 --output-dir outputs/checkpoints/sam-vit-large
```

## 2. Ejecutar la tabla experimental

```bash
python scripts/08_run_experiments.py --config configs/config.yaml
```

Esto ejecuta:

| Experimento | Modelo | Prompt | Métricas |
| --- | --- | --- | --- |
| E1 | SAM ViT-B base sin fine-tuning | Caja | IoU, Dice, mIoU, mPA, pixel accuracy, tiempo, memoria |
| E2 | SAM ViT-B fine-tuned | Caja | IoU, Dice, mIoU, mPA, pixel accuracy, tiempo, memoria |
| E3 | SAM ViT-B fine-tuned | Punto | IoU, Dice, mIoU, mPA, pixel accuracy, tiempo, memoria |
| E4 | Mejor modelo | Caja con imágenes degradadas | Robustez ante blur, ruido y compresión |
| E5 | Modelo actual o baseline | Detección/clasificación | Precisión, recall, F1 e IoU de cajas |

Los resultados consolidados quedan en:

```text
outputs/experiments/experiment_summary.csv
outputs/experiments/experiment_summary.md
```

Cada experimento también genera:

```text
evaluation_metrics.json
evaluation_rows.csv
visualizations/
```

## 3. Cambiar resolución por capacidad del modelo

La resolución se controla en `configs/config.yaml` por experimento:

```yaml
image_size: 1024
mask_size: 256
```

Regla práctica:

- `image_size: 512` o `768`: menor memoria, más rápido, puede afectar objetos pequeños.
- `image_size: 1024`: valor estándar de SAM.
- `image_size > 1024`: probar solo si hay GPU suficiente y si los objetos pequeños lo justifican.

Si usas `facebook/sam-vit-large` o `facebook/sam-vit-huge`, empieza con `batch_size: 1`.

## 4. Baseline E5

Para E5 se espera un archivo de predicciones en:

```text
outputs/baseline/predictions_coco.json
```

Formato esperado:

```json
[
  {
    "image_id": 1,
    "category_id": 1,
    "bbox": [120, 80, 90, 90],
    "bbox_format": "xywh",
    "score": 0.87
  }
]
```

También se permite `bbox_format: "xyxy"` si la caja viene como `[x1, y1, x2, y2]`.

Si todavía no tienes predicciones del modelo actual, E5 se marca como omitido y no bloquea E1-E4.

## 5. Lectura para el documento escrito

Usa `mean_iou`, `miou_by_category`, `mean_dice`, `mean_mpa`, `mean_pixel_accuracy` y `mean_inference_time_ms` para la tabla principal. Para análisis de errores, revisa `evaluation_rows.csv` y las imágenes de `visualizations/`.

## Optimización recomendada para GPU T4 + 14 GB de RAM

Este patch deja un perfil `runtime.profile: t4_14gb_ram` en `configs/config.yaml`. La configuración busca aprovechar la T4 con `mixed_precision`, `pin_memory`, `persistent_workers`, `prefetch_factor`, `channels_last` y optimizador AdamW fusionado cuando PyTorch lo soporte.

Configuración inicial recomendada:

```bash
python scripts/03_train_sam.py --config configs/config.yaml
```

Con esta configuración se usa:

- `image_size: 1024`
- `mask_size: 256`
- `batch_size: 2`
- `gradient_accumulation_steps: 2`
- batch efectivo de 4
- `num_workers: auto`, limitado a máximo 4 para no saturar los 14 GB de RAM

Si aparece error de memoria en GPU, baja primero el batch y conserva resolución:

```bash
python scripts/03_train_sam.py --config configs/config.yaml --batch-size 1
```

Si sigue fallando, baja resolución:

```bash
python scripts/03_train_sam.py --config configs/config.yaml --batch-size 1 --image-size 768 --mask-size 256
```

Si la GPU está por debajo de 80 % de uso y la RAM no está llena, sube workers manualmente:

```bash
python scripts/03_train_sam.py --config configs/config.yaml --num-workers 6 --cpu-threads auto
```

No conviene subir workers sin revisar RAM. Cada worker carga imágenes, máscaras COCO y procesamiento previo. En equipos con 14 GB, 3 o 4 workers suele ser mejor que usar todos los cores como workers.

Para correr la tabla experimental con la misma optimización:

```bash
python scripts/08_run_experiments.py --config configs/config.yaml
```

Para probar una resolución diferente en evaluación:

```bash
python scripts/04_evaluate_sam.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/sam-electric/best \
  --image-size 768 \
  --mask-size 256 \
  --batch-size 2 \
  --output-dir outputs/evaluation_768
```

Los archivos `run_metadata.json` y `evaluation_metrics.json` guardan el resumen del hardware usado, incluyendo GPU, memoria, workers, batch, resolución y precisión mixta. Esto sirve para justificar los experimentos en el documento escrito.
