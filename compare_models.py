"""Compare only real saved run results and select primarily by validation accuracy."""
import csv
import json
from pathlib import Path
from config import RESULTS_DIR

MODELS = ["mobilenetv2", "efficientnetb0", "resnet50"]

def main() -> None:
    rows = []
    for name in MODELS:
        path = RESULTS_DIR / name / "run_summary.json"
        if not path.exists():
            print(f"Skipping {name}: no measured run at {path}"); continue
        run = json.loads(path.read_text(encoding="utf-8")); test = run["test_metrics"]
        rows.append({"Model": name, "Accuracy": test["accuracy"],
                     "Precision": test["macro_precision"], "Recall": test["macro_recall"],
                     "F1-score": test["macro_f1"], "Weighted F1": test["weighted_f1"],
                     "Best Validation Accuracy": run["best_validation_accuracy"],
                     "Overfit Gap": run["final_training_accuracy"] - run["final_validation_accuracy"],
                     "Parameters": run["parameters"], "Training Time": run["training_time_seconds"]})
    if not rows: raise SystemExit("No completed runs found. Train at least one model first; no results were invented.")
    output = RESULTS_DIR / "model_comparison.csv"; output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    columns = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "Parameters", "Training Time"]
    widths = {c: max(len(c), *(len(f"{r[c]:.4f}") if isinstance(r[c], float) else len(str(r[c])) for r in rows)) for c in columns}
    print(" | ".join(c.ljust(widths[c]) for c in columns)); print("-+-".join("-"*widths[c] for c in columns))
    for row in rows: print(" | ".join((f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])).ljust(widths[c]) for c in columns))
    best = max(rows, key=lambda r: (r["Best Validation Accuracy"], r["F1-score"], -r["Overfit Gap"], -r["Training Time"]))
    print(f"\nBest Model: {best['Model']}")
    print(f"Reason: highest validation-led ranking; validation accuracy={best['Best Validation Accuracy']:.4f}, test macro F1={best['F1-score']:.4f}, overfit gap={best['Overfit Gap']:.4f}.")

if __name__ == "__main__": main()
