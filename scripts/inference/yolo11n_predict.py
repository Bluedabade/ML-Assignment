"""Single-image inference in the project's existing YOLO environment."""
import os
import time
from common import ROOT, DISPLAY, annotate, run

def predict(args):
    checkpoint = ROOT / 'runs/detect/results/training_v2/yolo11n_clean/weights/best.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError()
    for name, subfolder in [('YOLO_CONFIG_DIR', 'yolo'), ('MPLCONFIGDIR', 'matplotlib')]:
        cache = args.output / 'runtime' / subfolder
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(cache)
    import torch
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    if set(model.names.values()) != set(DISPLAY):
        raise ValueError('unexpected class mapping')
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 0:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = model.predict(str(args.image), imgsz=640, conf=args.confidence, iou=0.7,
                           max_det=300, agnostic_nms=False, device=device, save=False, verbose=False)[0]
    if device == 0:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    predictions = []
    for cid, confidence, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
        cls = model.names[int(cid)]
        predictions.append(dict(**{'class': cls}, canonical_class=cls, confidence=float(confidence),
                                **dict(zip(('x1','y1','x2','y2'), map(float, box)))))
    destination = args.output / 'yolo11n_annotated.jpg'
    annotate(args.image, predictions, destination)
    return dict(model='YOLO11n Clean V2', backend='CUDA' if device == 0 else 'CPU',
                confidence=args.confidence, inference_seconds=elapsed, predictions=predictions,
                annotated_image=str(destination))

if __name__ == '__main__':
    run(predict)
