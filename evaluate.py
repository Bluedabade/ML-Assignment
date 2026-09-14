"""OUTPUT STAGE: evaluate one saved model once on the held-out test set."""
import argparse
from pathlib import Path
import tensorflow as tf
from config import ARTIFACTS_DIR, BATCH_SIZE, CLASS_NAMES, IMG_SIZE, PROCESSED_DATA_DIR, RESULTS_DIR
from src.dataset import load_datasets
from src.metrics import evaluate_and_save
import src.models  # Registers the serializable preprocessing layer before loading.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mobilenetv2", "efficientnetb0", "resnet50"], required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    path = args.model_path or ARTIFACTS_DIR / args.model / "best_model.keras"
    if not path.exists(): raise SystemExit(f"Model not found: {path}. Train it first.")
    _, _, test = load_datasets(args.data_dir, args.img_size, args.batch_size)
    metrics = evaluate_and_save(tf.keras.models.load_model(path), test, CLASS_NAMES, RESULTS_DIR / args.model)
    print(metrics)
