"""Train a frozen transfer-learning head, optionally followed by fine tuning."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from config import (ARTIFACTS_DIR, BATCH_SIZE, CLASS_NAMES, FINE_TUNE_EPOCHS,
                    FINE_TUNE_LEARNING_RATE, IMG_SIZE, INITIAL_EPOCHS, PROCESSED_DATA_DIR,
                    LEARNING_RATE, PLOTS_DIR, RESULTS_DIR, SEED)
from src.dataset import load_datasets
from src.metrics import evaluate_and_save, plot_history
from src.models import BACKBONES, build_model, enable_fine_tuning
from src.utils import save_json, set_seed

def callbacks(path: Path):
    import tensorflow as tf
    return [
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=.2, min_lr=1e-7),
    ]

def train_one(name: str, args) -> None:
    import tensorflow as tf
    set_seed(args.seed)
    train_data, val_data, test_data = load_datasets(args.data_dir, args.img_size, args.batch_size, args.seed)
    model_dir, result_dir = ARTIFACTS_DIR / name, RESULTS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "best_model.keras"
    model, backbone = build_model(name, args.img_size, weights=args.weights,
                                  learning_rate=args.learning_rate)
    if args.smoke_test:
        print(f"\nSMOKE TEST ({name}): two training batches, one validation batch")
        started = time.perf_counter()
        smoke = model.fit(train_data.take(2), validation_data=val_data.take(1), epochs=1)
        smoke_dir = RESULTS_DIR / "smoke_test" / name
        smoke_history = {key: list(values) for key, values in smoke.history.items()}
        save_json(smoke_dir / "history.json", smoke_history)
        plot_history(smoke_history, smoke_dir / "training_history.png",
                     accuracy_output=PLOTS_DIR / f"{name}_smoke_accuracy.png",
                     loss_output=PLOTS_DIR / f"{name}_smoke_loss.png")
        print(f"Smoke test passed in {time.perf_counter() - started:.1f}s.")
        print("No checkpoint, test metrics, or comparable run_summary.json was created.")
        return
    training_callbacks = callbacks(checkpoint)
    print(f"\nSTAGE 1 - TRANSFER LEARNING ({name}, frozen backbone)")
    started = time.perf_counter()
    first = model.fit(train_data, validation_data=val_data, epochs=args.epochs,
                      callbacks=training_callbacks)
    history = {key: list(values) for key, values in first.history.items()}
    fine_start = None
    if args.fine_tune_epochs > 0:
        print("\nSTAGE 2 - FINE TUNING (small learning rate)")
        fine_start = len(history["loss"])
        enable_fine_tuning(model, backbone, args.unfreeze_last, args.fine_tune_lr)
        second = model.fit(train_data, validation_data=val_data,
                           initial_epoch=fine_start, epochs=fine_start + args.fine_tune_epochs,
                           callbacks=training_callbacks)
        for key, values in second.history.items(): history.setdefault(key, []).extend(values)
    elapsed = time.perf_counter() - started
    save_json(result_dir / "history.json", history)
    plot_history(history, result_dir / "training_history.png", fine_start,
                 PLOTS_DIR / f"{name}_accuracy.png", PLOTS_DIR / f"{name}_loss.png")
    best_model = tf.keras.models.load_model(checkpoint)
    metrics = evaluate_and_save(best_model, test_data, CLASS_NAMES, result_dir)
    metadata = {"model": name, "parameters": best_model.count_params(), "training_time_seconds": elapsed,
                "best_validation_accuracy": max(history["val_accuracy"]),
                "final_training_accuracy": history["accuracy"][-1],
                "final_validation_accuracy": history["val_accuracy"][-1],
                "epochs_completed": len(history["loss"]), "test_metrics": metrics}
    metadata["learning_rate"] = args.learning_rate
    metadata["fine_tune_learning_rate"] = args.fine_tune_lr
    save_json(result_dir / "run_summary.json", metadata)
    print(f"Saved best model: {checkpoint}")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*BACKBONES, "all"], default="mobilenetv2")
    parser.add_argument("--epochs", type=int, default=INITIAL_EPOCHS, help="Stage 1 epochs (default: 10)")
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--unfreeze-last", type=int, default=30)
    parser.add_argument("--fine-tune-lr", type=float, default=FINE_TUNE_LEARNING_RATE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
    parser.add_argument("--smoke-test", action="store_true",
                        help="Train only a few batches and create non-final history/plots")
    args = parser.parse_args(); args.weights = None if args.weights == "none" else args.weights
    if args.epochs < 1 or args.fine_tune_epochs < 0: parser.error("Epoch counts must be positive (fine tuning may be 0).")
    return args

if __name__ == "__main__":
    arguments = parse_args()
    for model_name in (BACKBONES if arguments.model == "all" else [arguments.model]): train_one(model_name, arguments)
