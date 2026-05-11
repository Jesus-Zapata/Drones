# Fine-tuning de SAM para redes eléctricas con imágenes de drones

Este proyecto implementa una base en Python para ajustar **Segment Anything Model (SAM)** en imágenes de redes eléctricas capturadas por drones. El objetivo es mejorar la segmentación de elementos como:

- aisladores
- cortacircuitos
- transformadores

El enfoque usa **SAM como segmentador guiado por prompts**. Durante entrenamiento se usa la caja `bbox` de cada anotación COCO como prompt, y la máscara de la instancia como objetivo. La clase del objeto se conserva desde la anotación COCO para asociar cada máscara con `aislador`, `cortacircuitos` o `transformador`.

## 1. Qué debes hacer antes de entrenar

Antes del código, la parte crítica es el dataset. SAM puede generar buenas máscaras, pero no sabe por sí solo que una máscara corresponde a un aislador, un cortacircuitos o un transformador. Para que el proyecto aprenda tu dominio, debes preparar imágenes anotadas manualmente.

### 1.1 Recolectar imágenes

Usa imágenes reales de drones, tomadas en diferentes condiciones:

- alturas y ángulos distintos
- buena y mala iluminación
- fondos variados: cielo, vegetación, postes, cables, estructuras
- objetos parcialmente tapados
- distintas resoluciones y distancias
- ejemplos donde los objetos sean pequeños dentro de la imagen

Evita entrenar solo con imágenes perfectas. En producción, el modelo verá casos difíciles.

### 1.2 Depurar imágenes

Antes de etiquetar:

- elimina imágenes totalmente borrosas
- elimina duplicados o imágenes casi iguales
- conserva casos difíciles, pero visibles
- separa imágenes por activo, zona o fecha para evitar fuga de datos entre entrenamiento y validación

Una división inicial razonable:

```text
70 % train
15 % validation
15 % test
```

Si tienes pocas imágenes, empieza con `80 % train`, `10 % val`, `10 % test`, pero no evalúes con imágenes que el modelo ya vio.

### 1.3 Definir clases

Define pocas clases al inicio:

```text
aislador
cortacircuitos
transformador
```

Usa siempre los mismos nombres. No mezcles `corta circuito`, `cortacircuito`, `cortacircuitos`, etc. El archivo `configs/config.yaml` espera estos nombres.

### 1.4 Etiquetar manualmente

Sí, debes etiquetar manualmente una muestra representativa. Para cada imagen, marca cada objeto como una **instancia individual**. Si hay tres aisladores visibles, deben quedar tres máscaras separadas, cada una con clase `aislador`.

Recomendaciones:

- etiqueta solo la parte visible del objeto
- no incluyas fondo dentro de la máscara
- si el objeto está cortado, etiqueta la parte visible
- si dos objetos se tocan, sepáralos como instancias distintas
- revisa bordes en zoom, porque los objetos eléctricos pueden ser pequeños
- usa polígonos o pincel/brush cuando la forma sea irregular

### 1.5 Herramientas recomendadas

Puedes usar cualquiera de estas:

- **CVAT**: buena opción para proyectos de visión computacional y exportación COCO.
- **Roboflow**: útil para organizar datasets, versionar y exportar en COCO Segmentation.
- **Label Studio**: permite etiquetado manual y asistencia con modelos, pero debes validar bien la exportación a COCO segmentation.

Para este proyecto, el formato recomendado es:

```text
COCO Instance Segmentation
```

No basta con tener cajas. Necesitas máscaras de segmentación.

### 1.6 Exportar anotaciones

Exporta en formato COCO segmentation. La estructura mínima esperada es:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "drone_001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [120, 80, 90, 90],
      "segmentation": [[120,80, 210,80, 210,170, 120,170]],
      "area": 8100,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "aislador"},
    {"id": 2, "name": "cortacircuitos"},
    {"id": 3, "name": "transformador"}
  ]
}
```

La segmentación puede estar como polígonos o RLE válido de COCO.

## 2. Limitaciones importantes de SAM

SAM no es un clasificador. SAM genera máscaras a partir de prompts como puntos, cajas o máscaras previas. Por eso:

- SAM puede decir “aquí hay una máscara”.
- SAM no necesariamente puede decir “esto es un aislador”.
- Para asociar clase necesitas anotaciones con `category_id` en entrenamiento.
- En inferencia automática necesitas una etapa previa o adicional que proponga y clasifique objetos.

En este proyecto hay dos rutas de inferencia:

1. **Inferencia con prompts conocidos**: entregas una caja y una clase estimada, y SAM devuelve la máscara.
2. **Máscaras automáticas sin clase**: SAM genera muchas máscaras, pero luego alguien o algún modelo adicional debe clasificarlas.

Para una solución productiva, lo más práctico suele ser:

```text
Detector/clasificador de objetos eléctricos → cajas con clase → SAM → máscara precisa por instancia
```

Ejemplo:

```text
YOLO / GroundingDINO / modelo propio de detección → SAM fine-tuned
```

## 3. Estructura del proyecto

```text
sam-electric-finetuning/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    config.yaml
  data/
    README.md
    raw/
      images/
    coco/
      images/
      annotations/
        instances_train.json
        instances_val.json
        instances_test.json
  examples/
    prompts_example.json
    sample_coco_categories.json
  scripts/
    01_validate_coco.py
    02_split_coco.py
    03_train_sam.py
    04_evaluate_sam.py
    05_infer_with_prompts.py
    06_generate_auto_masks.py
    07_visualize_coco.py
  src/
    sam_electric/
      coco.py
      dataset.py
      metrics.py
      utils.py
      visualization.py
  outputs/
    checkpoints/
    figures/
```

## 4. Instalación

Crea un ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

Instala dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Si usas GPU, instala PyTorch según tu versión de CUDA desde la guía oficial de PyTorch.

## 5. Preparar datos

Copia tus imágenes en:

```text
data/coco/images/
```

Copia tus anotaciones en:

```text
data/coco/annotations/instances_train.json
data/coco/annotations/instances_val.json
data/coco/annotations/instances_test.json
```

Si exportaste un solo JSON COCO, divídelo así:

```bash
python scripts/02_split_coco.py \
  --annotations data/coco/annotations/instances_all.json \
  --output-dir data/coco/annotations \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

## 6. Validar anotaciones

```bash
python scripts/01_validate_coco.py \
  --annotations data/coco/annotations/instances_train.json \
  --image-dir data/coco/images
```

Haz lo mismo para validación:

```bash
python scripts/01_validate_coco.py \
  --annotations data/coco/annotations/instances_val.json \
  --image-dir data/coco/images
```

## 7. Visualizar anotaciones

Antes de entrenar, revisa visualmente el COCO:

```bash
python scripts/07_visualize_coco.py \
  --annotations data/coco/annotations/instances_train.json \
  --image-dir data/coco/images \
  --output-dir outputs/coco_preview \
  --max-images 30
```

Abre las imágenes generadas en `outputs/coco_preview/` y revisa que las máscaras sí correspondan a la clase correcta.

## 8. Entrenamiento

Por defecto se usa:

```text
facebook/sam-vit-base
```

Para entrenar:

```bash
python scripts/03_train_sam.py --config configs/config.yaml
```

El mejor modelo queda en:

```text
outputs/checkpoints/sam-electric/best/
```

El entrenamiento congela el encoder visual y el prompt encoder, y ajusta principalmente el `mask_decoder`. Esto reduce costo y riesgo de sobreajuste.

Si tienes más GPU y más datos, puedes probar `facebook/sam-vit-large` o `facebook/sam-vit-huge` en `configs/config.yaml`.

## 9. Evaluación

```bash
python scripts/04_evaluate_sam.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/sam-electric/best \
  --save-visualizations
```

El script calcula:

- IoU promedio
- Dice promedio
- métricas por clase
- visualizaciones con máscara real y predicha

Los resultados quedan en:

```text
outputs/evaluation_metrics.json
outputs/evaluation_visualizations/
```

## 10. Inferencia sobre imágenes nuevas

Para inferencia con clase, necesitas entregar prompts. El ejemplo está en:

```text
examples/prompts_example.json
```

Formato:

```json
{
  "images": [
    {
      "file_name": "drone_001.jpg",
      "prompts": [
        {
          "label": "aislador",
          "box": [120, 80, 210, 170]
        }
      ]
    }
  ]
}
```

Ejecuta:

```bash
python scripts/05_infer_with_prompts.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/sam-electric/best \
  --image-dir data/raw/images \
  --prompts examples/prompts_example.json \
  --output-dir outputs/inference
```

Salida:

```text
outputs/inference/masks/
outputs/inference/overlays/
outputs/inference/results.json
```

## 11. Generar máscaras automáticas sin clase

Este script usa el SAM original de Meta para generar máscaras automáticas. Sirve para pre-etiquetado, no para inferencia final con clases.

Primero descarga un checkpoint `.pth` de SAM original. Luego:

```bash
python scripts/06_generate_auto_masks.py \
  --image-dir data/raw/images \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --output-dir outputs/auto_masks
```

Importante: estas máscaras salen sin clase. Debes revisarlas y asignarles `aislador`, `cortacircuitos` o `transformador` antes de usarlas como ground truth.

## 12. Recomendación de flujo real

Para un primer MVP:

1. Selecciona 300 a 500 imágenes representativas.
2. Etiqueta instancias de las tres clases.
3. Exporta COCO segmentation.
4. Valida y visualiza las anotaciones.
5. Entrena con `sam-vit-base`.
6. Evalúa por clase.
7. Revisa errores: objetos pequeños, oclusiones, fondos complejos.
8. Aumenta datos donde el modelo falle.

Para una solución más completa:

1. Entrena un detector de objetos eléctricos.
2. Usa el detector para producir cajas y clases.
3. Usa SAM ajustado para obtener máscaras precisas.
4. Evalúa la cadena completa con imágenes nunca vistas.

## 13. Notas para imágenes de drones

En redes eléctricas, muchos objetos son pequeños frente al tamaño total de la imagen. Si el modelo falla, considera:

- recortar la imagen en tiles
- entrenar con recortes alrededor de estructuras eléctricas
- aumentar resolución de entrada si el hardware lo permite
- mejorar balance entre clases
- etiquetar más ejemplos de objetos pequeños
- separar validación por zona o circuito para medir generalización real
