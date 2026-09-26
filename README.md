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

## Future training workflow (Phase 3; do not run yet)

The prepared configurations are `configs/yolo11n.yaml` and
`configs/yolo11s.yaml`. In Phase 3, both pretrained models will use the same
dataset, split, image size, seed, optimizer, and initial hyperparameters.

The comparison will report Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95,
training time, inference latency/FPS, parameter count, and model size. It will
also include training/validation curves, confusion matrices, F1 curves, and
prediction examples. Training is intentionally not executed in Phase 2.

