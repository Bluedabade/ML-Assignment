"""Shared JSON protocol and canonical-only image rendering."""
import argparse
import contextlib
import io
import json
import logging
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DISPLAY = {'acne': 'Acne', 'wrinkle': 'Wrinkle', 'dark_spot': 'Dark Spot', 'enlarged_pore': 'Enlarged Pore'}
COLORS = {'acne': '#ff5555', 'wrinkle': '#ffcc33', 'dark_spot': '#aa77ff', 'enlarged_pore': '#33cccc'}

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0.05 <= args.confidence <= 0.90:
        parser.error('confidence must be between 0.05 and 0.90')
    if not args.image.is_file():
        parser.error('input image not found')
    args.output.mkdir(parents=True, exist_ok=True)
    return args

def annotate(image_path, predictions, destination):
    from PIL import Image, ImageDraw
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for p in predictions:
        cls = p['canonical_class']
        box = [p[k] for k in ('x1', 'y1', 'x2', 'y2')]
        draw.rectangle(box, outline=COLORS[cls], width=max(2, image.width // 300))
        label = f"{DISPLAY[cls]} {p['confidence']:.1%}"
        x, y = max(0, box[0]), max(0, box[1] - 14)
        draw.text((x, y), label, fill=COLORS[cls], stroke_width=1, stroke_fill='black')
    image.save(destination, quality=95)

def run(predict):
    args = arguments()
    # Suppress third-party logs, which may contain signed authentication URLs.
    logging.disable(logging.CRITICAL)
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            start = time.perf_counter()
            payload = predict(args)
            payload['total_seconds'] = time.perf_counter() - start
        print(json.dumps(payload, ensure_ascii=True))
    except Exception as exc:
        # Never serialize exception messages or captured framework logs.
        codes = {'FileNotFoundError': 'MISSING_MODEL_OR_CACHE', 'PermissionError': 'ACCESS_REQUIRED'}
        code = getattr(exc, 'safe_code', codes.get(type(exc).__name__, 'LOCAL_INFERENCE_FAILED'))
        print(json.dumps({'error': code}))
        raise SystemExit(1)

def fail(code):
    error = RuntimeError()
    error.safe_code = code
    raise error
