| experiment_id | description | model | prompt_type | mean_iou | miou_by_category | mean_dice | mean_mpa | mean_pixel_accuracy | mean_inference_time_ms | mean_gpu_memory_mb | instances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | SAM ViT-B base sin fine-tuning con prompt de caja | facebook/sam-vit-base | box | 0.0953 | 0.0741 | 0.1433 | 0.5554 | 0.9842 | 135.3107 | 2218.1388 | 99 |
| E2 | SAM ViT-B fine-tuned con prompt de caja | facebook/sam-vit-base | box | 0.6516 | 0.6344 | 0.7752 | 0.8993 | 0.9955 | 134.0552 | 2218.1388 | 99 |
| E3 | SAM ViT-B fine-tuned con prompt de punto positivo | facebook/sam-vit-base | point | 0.2204 | 0.2138 | 0.3321 | 0.6518 | 0.9875 | 133.7763 | 2218.1393 | 99 |
| E4_blur_s3 | Robustez del mejor modelo con imágenes degradadas | facebook/sam-vit-base | box | 0.6518 | 0.6332 | 0.7753 | 0.8993 | 0.9955 | 135.7430 | 2218.1388 | 99 |
| E4_gaussian_noise_s3 | Robustez del mejor modelo con imágenes degradadas | facebook/sam-vit-base | box | 0.6470 | 0.6337 | 0.7702 | 0.8959 | 0.9954 | 131.6803 | 2218.1388 | 99 |
| E4_jpeg_compression_s3 | Robustez del mejor modelo con imágenes degradadas | facebook/sam-vit-base | box | 0.6494 | 0.6312 | 0.7731 | 0.8990 | 0.9954 | 133.2872 | 2218.1388 | 99 |
| E5 | Baseline externo, por ejemplo Custom Vision, YOLO o detector actual | baseline_coco_predictions |  |  |  |  |  |  |  |  |  |
