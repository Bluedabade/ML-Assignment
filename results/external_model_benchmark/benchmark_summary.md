# Same-test-set external pretrained comparison

Test: 248 images, 1,636 objects (acne 1,415; wrinkle 89; dark_spot 15; enlarged_pore 117). COCO JSON and YOLO labels agree.

## Shared COCO evaluation

COCOeval bbox; IoU 0.50:0.05:0.95; 101 recall points; area=all; maxDets=[1,10,300]. Values are fractions.

| Model | mAP50 | mAP50-95 |
|---|---:|---:|
| YOLO11n Clean V2 | 0.045104 | 0.010928 |
| SKN-1 | 0.397378 | 0.142915 |

| Model | Class | AP50 | AP50-95 |
|---|---|---:|---:|
| YOLO11n Clean V2 | acne | 0.140629 | 0.034199 |
| YOLO11n Clean V2 | wrinkle | 0.000209 | 0.000024 |
| YOLO11n Clean V2 | dark_spot | 0.000000 | 0.000000 |
| YOLO11n Clean V2 | enlarged_pore | 0.039577 | 0.009491 |
| SKN-1 | acne | 0.358380 | 0.105854 |
| SKN-1 | wrinkle | 0.267751 | 0.096503 |
| SKN-1 | dark_spot | 0.545743 | 0.208694 |
| SKN-1 | enlarged_pore | 0.417638 | 0.160608 |

## Secondary operating point

Same class-aware greedy matching: confidence >=0.25, IoU >=0.50. Not the historical Ultralytics P/R definition.

| Model | Micro Precision | Micro Recall | Micro F1 |
|---|---:|---:|---:|
| YOLO11n Clean V2 | 0.376147 | 0.050122 | 0.088457 |
| SKN-1 | 0.384808 | 0.470660 | 0.423426 |

Per-class P/R and TP/FP/FN: shared_evaluation/secondary_precision_recall.json.

## Inference protocol

### skn1

```json
{
  "model_id": "skn-1/2",
  "model_type": "YOLONasForObjectDetectionOnnx",
  "backend": "ONNX CPU",
  "execution_providers": [
    "CPUExecutionProvider"
  ],
  "cache_paths": [
    "D:\\Projects\\ML-Assignment-roboflow-env\\model_cache\\models-cache\\v2-skn-1-2-e167c1aeda1e8b7931e14b7a6855a6c0\\7ee34fce76456d9ff4cfe6e7049aad1e"
  ],
  "confidence_floor": 0.001,
  "nms_iou": 0.3,
  "max_detections": 300,
  "class_agnostic_nms": false,
  "preprocessing": {
    "image_pre_processing": {
      "auto_orient": {
        "enabled": true
      },
      "static_crop": null,
      "contrast": null,
      "grayscale": null
    },
    "network_input": {
      "training_input_size": {
        "height": 640,
        "width": 640
      },
      "dataset_version_resize_dimensions": null,
      "dynamic_spatial_size_supported": false,
      "dynamic_spatial_size_mode": null,
      "color_mode": "rgb",
      "resize_mode": "stretch",
      "padding_value": null,
      "input_channels": 3,
      "scaling_factor": 255,
      "normalization": null
    },
    "forward_pass": {
      "static_batch_size": 1,
      "max_dynamic_batch_size": null
    },
    "post_processing": {
      "type": "nms",
      "fused": false,
      "nms_parameters": null
    },
    "model_initialization": null,
    "class_names_operations": null
  },
  "input_shape": [
    1,
    3,
    640,
    640
  ],
  "processed_images": 248,
  "total_inference_seconds": 673.7906862000127,
  "mean_inference_seconds": 2.716897928225858,
  "raw_class_counts": {
    "Acne": 35635,
    "Black Heads": 4,
    "Dark Circles": 961,
    "Dark Spots": 146,
    "Dry Skin": 15422,
    "Oily Skin": 9857,
    "Open Pores": 3080,
    "Red Skin": 184,
    "Wrinkles": 8581
  },
  "canonical_prediction_count": 47442,
  "out_of_scope_count": 26428,
  "class_mapping": {
    "Acne": "acne",
    "Wrinkles": "wrinkle",
    "Dark Spots": "dark_spot",
    "Open Pores": "enlarged_pore"
  },
  "packages": {
    "inference-models": "0.39.1",
    "onnxruntime": "1.22.1",
    "torch": "2.14.1",
    "numpy": "2.4.6"
  }
}
```

### yolo11n

```json
{
  "model_path": "runs\\detect\\results\\training_v2\\yolo11n_clean\\weights\\best.pt",
  "checkpoint_sha256": "aab326fca99a0ed8a97d9758ad3b8a5582d8921a0cce58fc4a945de2cd46ce88",
  "classes": {
    "0": "acne",
    "1": "wrinkle",
    "2": "dark_spot",
    "3": "enlarged_pore"
  },
  "device": 0,
  "gpu": "NVIDIA GeForce GTX 1650 Ti",
  "conf": 0.001,
  "imgsz": 640,
  "nms_iou": 0.7,
  "max_det": 300,
  "agnostic_nms": false,
  "rect": true,
  "multi_label": false,
  "processed_images": 248,
  "total_inference_seconds": 10.887480699981097,
  "mean_inference_seconds": 0.043901131854762486,
  "preprocessing": "Normal Ultralytics predict letterbox and normalization; no custom transforms",
  "packages": {
    "ultralytics": "8.4.163",
    "torch": "2.14.0+cu130",
    "torchvision": "0.29.0+cu130"
  }
}
```

## Limitations and historical context

SKN-1 has higher AP50 and AP50-95 for all four classes on this test set under the recorded prediction/evaluation protocol. Its micro recall and F1 at confidence 0.25 are higher; micro precision is close. This does not establish universal superiority.

SKN-1 external training provenance is not fully known relative to this project. Original training images may overlap this test set. This comparison is NOT guaranteed leakage-free.
Published Roboflow metrics are not used. Historical Clean-V2 Ultralytics P=0.321 R=0.0878 mAP50=0.0568 mAP50-95=0.0158 are context only. COCO interpolation, maxDet, inference/NMS and aggregation differ.
SKN-1 ONNX CPU timing must not be used to claim speed superiority/inferiority relative to YOLO CUDA timing.
Timing covers model calls, excludes image reads/model loading and includes first-call initialization. Processes initially overlapped; this is not a controlled hardware-speed benchmark.
Historical per-class Ultralytics mAP50: acne=0.138, wrinkle=0.000198, dark_spot=0.0514, enlarged_pore=0.0376. Existing outputs were not overwritten.
Model-specific preprocessing/NMS preserved: SKN stretch 640 and NMS 0.3; YOLO letterbox imgsz=640 and NMS 0.7. Out-of-scope SKN labels can consume its 300-detection cap; they are not remapped.
Ultralytics predict uses default single-label-per-candidate NMS, whereas detect val enables multi_label=True. This prediction protocol can discard secondary class scores (including dark_spot) and need not reproduce historical val AP. SKN NMS also selects each candidate's highest-scoring class before suppression.
Minority support is limited, especially dark_spot (15 objects). No universal superiority or medical diagnostic claim is supported.
SKN-1 dataset/project license shown: CC BY 4.0. Separate model-weight licensing not independently established.
No training or fine-tuning. Test data and existing outputs remain unchanged.
