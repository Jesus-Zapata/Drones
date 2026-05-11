# Datos esperados

Este proyecto espera un dataset de segmentación de instancias en formato COCO.

Estructura recomendada:

```text
data/coco/
  images/
    img_001.jpg
    img_002.jpg
  annotations/
    instances_train.json
    instances_val.json
    instances_test.json   # opcional
```

Cada anotación COCO debe tener:

- `image_id`
- `category_id`
- `bbox` en formato COCO `[x, y, width, height]`
- `segmentation`, preferiblemente polígonos o RLE válido
- `area`
- `iscrowd`, normalmente `0`

Las categorías mínimas esperadas son:

```json
[
  {"id": 1, "name": "aislador"},
  {"id": 2, "name": "cortacircuitos"},
  {"id": 3, "name": "transformador"}
]
```

Puedes usar otros IDs, siempre que los `category_id` de las anotaciones coincidan con `categories`.
