# Facial Skin Condition Object Detection

Student Machine Learning / Computer Vision assignment for detecting facial skin
conditions with bounding boxes, class names, and confidence scores.

The September 30 draft is deliberately a **four-class object detector**:

| ID | Class |
|---:|---|
| 0 | `acne` |
| 1 | `wrinkle` |
| 2 | `dark_spot` |
| 3 | `enlarged_pore` |

`normal_skin` is future work. The available data has no real normal-skin bounding
boxes, so this repository does not fabricate full-face or classification-derived
boxes.

## Data sources

- `datasets/raw/roboflow_skin_problem_clean3/`: COCO object-detection source used
  for the draft derived dataset.
- `datasets/raw/kaggle_acne_wrinkle_spots/`: classification only; not used for
  draft detection training.
- `datasets/raw/kaggle_oily_dry_normal/`: classification only; not used for draft
  detection training.

Everything under `datasets/raw/` is immutable. Derived YOLO data is generated at
`datasets/processed/skin_detection/`.

The instructor workflow is preserved in
`reference/chapter 8_2 YoLo Dog Breeds.ipynb`. The reference file must not be
edited. The student-facing notebook uses relative paths and follows the same
setup, preparation, evaluation, comparison, and inference progression.

## Repository layout

```text
configs/       class mapping and future YOLO11n/YOLO11s run settings
datasets/raw/  immutable source datasets
datasets/processed/skin_detection/  reproducible derived YOLO dataset
docs/          annotation policy and preparation report
notebooks/     guided draft notebook
reference/     instructor notebook (read-only)
results/       generated visual checks and future training outputs
scripts/       preparation, validation, and visualization tools
```

## Setup after cloning

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare and inspect the dataset

Run from the repository root:

```bash
python scripts/prepare_detection_dataset.py
python scripts/validate_dataset.py
python scripts/visualize_annotations.py
```

Preparation is deterministic (`seed = 42`), converts only approved COCO classes,
clips out-of-bound boxes in the derived copy, removes exact duplicate leakage,
and retains documented negative examples. See
`docs/ANNOTATION_POLICY.md` and `docs/DATA_PREPARATION_REPORT.md`.

## Draft Model Results (completed Phase 3)

These are preliminary Draft test results from the completed 30-epoch experiments.

| Model | Precision | Recall | mAP50 | mAP50-95 | Parameters | Training hours | best.pt (approx.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 0.386 | 0.130 | 0.0860 | 0.0242 | 2,582,932 | 0.506 | 5.5 MB |
| YOLO11s | 0.404 | 0.112 | 0.0837 | 0.0230 | 9,414,348 | 0.939 | 19.2 MB |

YOLO11s had higher test Precision. YOLO11n had higher test Recall and slightly
higher test mAP50 and mAP50-95. YOLO11n is smaller and trained faster in this
experiment. These observations do not establish that either model is universally
better.

Training artifacts, including best/last weights, curves, results, predictions,
and run settings, are preserved under `runs/detect/results/training/yolo11n/`
and `runs/detect/results/training/yolo11s/`. Test artifacts are under
`runs/detect/results/evaluation/yolo11n_test/` and `yolo11s_test/`.
Smoke-test artifacts are also retained under `runs/detect/results/smoke_test/`.
The saved training `args.yaml` files record 30 epochs; the current editable
`configs/yolo11n.yaml` has 2 epochs and is preserved unchanged in this checkpoint.

Known limitations remain: severe class imbalance, very few dark_spot objects,
currently poor wrinkle performance, and many negative images produced by filtering
excluded source classes. The Draft remains four classes (0 acne, 1 wrinkle,
2 dark_spot, 3 enlarged_pore); normal_skin is excluded from detection training.
Raw datasets must remain unchanged. Phase 4 has not started.

## Training workflow reference

The editable configurations are `configs/yolo11n.yaml` and `configs/yolo11s.yaml`.
For the settings actually used in completed experiments, consult each saved
training `args.yaml`. No training or Phase 4 work was run for this checkpoint.
