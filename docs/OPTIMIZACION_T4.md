# Optimización para NVIDIA T4 y CPU con 14 GB de RAM

Este ajuste prioriza tres cosas:

1. Mantener la GPU ocupada usando `mixed_precision`, `pin_memory`, `prefetch_factor` y `persistent_workers`.
2. No saturar la RAM de 14 GB con demasiados procesos de carga.
3. Permitir cambiar resolución y batch por línea de comandos para cada modelo o experimento.

## Comandos recomendados

Ver el perfil detectado:

```bash
python scripts/10_profile_hardware.py --config configs/config.yaml
```

Entrenar con el perfil T4:

```bash
python scripts/03_train_sam.py --config configs/config.yaml
```

Entrenar con resolución menor si aparece error de memoria:

```bash
python scripts/03_train_sam.py --config configs/config.yaml --batch-size 1 --image-size 768 --mask-size 256
```

Forzar más carga de CPU si la GPU está esperando datos y la RAM alcanza:

```bash
python scripts/03_train_sam.py --config configs/config.yaml --num-workers 6 --cpu-threads auto
```

Ejecutar la tabla experimental:

```bash
python scripts/08_run_experiments.py --config configs/config.yaml
```

## Parámetros principales

- `runtime.num_workers`: procesos paralelos del DataLoader. En 14 GB se limita por defecto a máximo 4.
- `runtime.cpu_threads`: hilos usados por Torch/OpenMP. En `auto` usa todos los disponibles.
- `training.batch_size`: batch real en GPU.
- `training.gradient_accumulation_steps`: acumula gradientes para aumentar el batch efectivo sin aumentar memoria.
- `training.image_size`: lado mayor de la imagen procesada por SAM.
- `training.mask_size`: resolución interna de la máscara.

## Orden de ajuste ante problemas

Si hay `CUDA out of memory`:

1. Baja `batch_size` a 1.
2. Mantén `gradient_accumulation_steps` en 2 o 4.
3. Si sigue fallando, baja `image_size` a 768.
4. Si aún falla, usa `image_size` 640.

Si la GPU se mantiene con bajo uso:

1. Sube `num_workers` a 6.
2. Revisa que la RAM no pase de 85 %.
3. Si la RAM se llena, vuelve a `num_workers` 4 o 3.
