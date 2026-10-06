"""Reuse Phase 5B AutoModel/ONNX CPU inference; no cloud prediction."""
import os
import time
from pathlib import Path
from common import annotate, fail, run

MAPPING = {'Acne':'acne', 'Wrinkles':'wrinkle', 'Dark Spots':'dark_spot', 'Open Pores':'enlarged_pore'}
CACHE = Path('D:/Projects/ML-Assignment-roboflow-env/model_cache')

def predict(args):
    key = os.getenv('ROBOFLOW_API_KEY')
    if not key:
        fail('MISSING_ROBOFLOW_API_KEY')
    if not CACHE.is_dir() or not any(CACHE.rglob('weights.onnx')):
        fail('MISSING_LOCAL_MODEL_CACHE')
    os.environ['INFERENCE_HOME'] = str(CACHE)
    os.environ['DEFAULT_DEVICE'] = 'cpu'
    os.environ['ONNXRUNTIME_EXECUTION_PROVIDERS'] = 'CPUExecutionProvider'
    import cv2
    from inference_models import AutoModel
    model = AutoModel.from_pretrained('skn-1/2', api_key=key, backend='onnx', device='cpu',
        onnx_execution_providers=['CPUExecutionProvider'], verbose=False,
        allow_untrusted_packages=False, allow_local_code_packages=False)
    if model._session.get_providers() != ['CPUExecutionProvider']:
        fail('UNEXPECTED_EXECUTION_PROVIDER')
    image = cv2.imread(str(args.image))
    if image is None:
        fail('INVALID_IMAGE')
    start = time.perf_counter()
    pred = model(image, confidence=args.confidence, iou_threshold=0.3,
                 max_detections=300, class_agnostic_nms=False)[0]
    elapsed = time.perf_counter() - start
    canonical, other = [], []
    for cid, confidence, box in zip(pred.class_id.tolist(), pred.confidence.tolist(), pred.xyxy.tolist()):
        label = model.class_names[int(cid)]
        row = dict(raw_class=label, canonical_class=MAPPING.get(label), confidence=float(confidence),
                   **dict(zip(('x1','y1','x2','y2'), map(float, box))))
        (canonical if label in MAPPING else other).append(row)
    destination = args.output / 'skn1_annotated.jpg'
    annotate(args.image, canonical, destination)
    return dict(model='skn-1/2', backend='ONNX CPU / CPUExecutionProvider',
                confidence=args.confidence, inference_seconds=elapsed, predictions=canonical,
                out_of_scope_predictions=other, raw_classes=model.class_names, annotated_image=str(destination))

if __name__ == '__main__':
    run(predict)
