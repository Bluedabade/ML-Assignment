"""Predict one main skin class and display all five probabilities."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from config import ARTIFACTS_DIR, IMG_SIZE
from src.utils import load_class_names
import src.models  # Registers the serializable preprocessing layer before loading.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", required=True,
                        help="Model name (mobilenetv2/efficientnetb0/resnet50) or .keras path")
    parser.add_argument("--class-map", type=Path, default=ARTIFACTS_DIR / "class_names.json")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    args = parser.parse_args()
    model_path = ARTIFACTS_DIR / args.model.lower() / "best_model.keras"
    if args.model.lower() not in {"mobilenetv2", "efficientnetb0", "resnet50"}:
        model_path = Path(args.model)
    for path in (args.image, model_path, args.class_map):
        if not path.exists(): raise SystemExit(f"Not found: {path}")
    names = load_class_names(args.class_map)
    image = tf.keras.utils.load_img(args.image, target_size=(args.img_size, args.img_size))
    batch = np.expand_dims(tf.keras.utils.img_to_array(image), 0)
    probabilities = tf.keras.models.load_model(model_path).predict(batch, verbose=0)[0]
    winner = int(np.argmax(probabilities))
    print(f"Predicted class:\n{names[winner]}\n\nConfidence:\n{probabilities[winner]*100:.2f}%\n")
    print("All class probabilities:")
    for name, probability in zip(names, probabilities): print(f"{name:12} {probability*100:6.2f}%")
    print("\nEducational preliminary result only; not a medical diagnosis.")
