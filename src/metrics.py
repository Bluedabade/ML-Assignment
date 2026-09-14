import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def _decorate_history_axis(axis, fine_tune_start: int | None) -> None:
    if fine_tune_start:
        axis.axvline(fine_tune_start + .5, ls="--", color="gray", label="Fine tuning")
    axis.grid(alpha=.25)
    axis.legend()

def plot_history(history: dict, output: Path, fine_tune_start: int | None = None,
                 accuracy_output: Path | None = None, loss_output: Path | None = None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["accuracy"], label="Training Accuracy")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    axes[0].set(title="Model Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].plot(epochs, history["loss"], label="Training Loss")
    axes[1].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[1].set(title="Model Loss", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        _decorate_history_axis(axis, fine_tune_start)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)

    if accuracy_output:
        accuracy_output.parent.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(epochs, history["accuracy"], label="Training Accuracy")
        axis.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
        axis.set(title="Model Accuracy", xlabel="Epoch", ylabel="Accuracy")
        _decorate_history_axis(axis, fine_tune_start)
        fig.tight_layout(); fig.savefig(accuracy_output, dpi=160); plt.close(fig)
    if loss_output:
        loss_output.parent.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(epochs, history["loss"], label="Training Loss")
        axis.plot(epochs, history["val_loss"], label="Validation Loss")
        axis.set(title="Model Loss", xlabel="Epoch", ylabel="Loss")
        _decorate_history_axis(axis, fine_tune_start)
        fig.tight_layout(); fig.savefig(loss_output, dpi=160); plt.close(fig)

def evaluate_and_save(model, dataset, class_names: list[str], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    loss, accuracy = model.evaluate(dataset, verbose=1)
    truth = np.concatenate([y.numpy() for _, y in dataset])
    probabilities = model.predict(dataset, verbose=1)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(truth, predicted, target_names=class_names,
                                   labels=range(len(class_names)), zero_division=0, output_dict=True)
    text_report = classification_report(truth, predicted, target_names=class_names,
                                        labels=range(len(class_names)), zero_division=0)
    (output_dir / "classification_report.txt").write_text(text_report, encoding="utf-8")
    matrix = confusion_matrix(truth, predicted, labels=range(len(class_names)))
    plt.figure(figsize=(8, 7)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close()
    metrics = {"test_loss": float(loss), "accuracy": float(accuracy),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"], "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {n: {k: report[n][k] for k in ("precision", "recall", "f1-score", "support")} for n in class_names}}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
