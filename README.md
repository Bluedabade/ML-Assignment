# Skin Condition Classification

## Project Overview

This university Machine Learning assignment builds a reproducible facial/skin image classifier with TensorFlow, Keras, and transfer learning. One input image receives one main class. The project compares MobileNetV2, EfficientNetB0, and ResNet50 under the same split, seed, augmentation policy, initial epochs, and evaluation metrics.

```text
Dataset inspection -> Dataset preparation -> Train/Validation/Test split
-> Resize -> Normalize -> Data augmentation
-> Transfer learning -> Fine-tuning -> Evaluation
-> Prediction on unseen images -> Model comparison
```

This is image classification, not object detection or segmentation. Results are educational and must not be treated as medical diagnoses.

## Classes

The model produces exactly five softmax probabilities in this saved order:

1. `normal`
2. `wrinkle`
3. `acne`
4. `dark_spot`
5. `large_pores`

## Project Structure

```text
ML-Assignment/
|-- config.py                         Central paths and hyperparameters
|-- check_environment.py              Python, TensorFlow, CPU/GPU check
|-- train.py                          Transfer learning and fine-tuning
|-- evaluate.py                       Held-out test evaluation
|-- compare_models.py                 Comparison of completed real runs
|-- predict.py                        Five-class prediction for one image
|-- requirements.txt                  Pinned Python dependencies
|-- setup.ps1 / setup.sh              Optional environment helpers
|-- scripts/
|   |-- audit_dataset.py              Integrity, label and annotation audit
|   `-- prepare_dataset.py            Deduplicated 70/15/15 split
|-- src/                              Dataset, preprocessing, models, metrics
|-- notebooks/skin_transfer_learning.ipynb
|-- data/raw/                         Source datasets (ignored by Git)
|-- data/processed/                   Generated split (ignored by Git)
|-- artifacts/                        Class map and ignored model weights
`-- reports/                          Audit, plots, metrics and comparisons
```

## Dataset Sources / Dataset Layout

Datasets are downloaded separately because image collections are too large for normal Git history. Place them here:

```text
data/raw/
|-- kaggle_acne_wrinkle_spots/
|-- kaggle_oily_dry_normal/
`-- roboflow_skin_problem_clean3/
```

For compatibility, the scripts also recognize those exact folders at repository root. They inspect internal layout rather than assuming it. The current sources include folder-classification datasets and a Roboflow COCO object-detection export. Oily and dry are extra classes and are excluded. A COCO image is eligible only when all its annotations unambiguously map to one target.

Unknown labels, extra labels, multi-target images, conflicting duplicates, missing files, and unreadable images are never silently assigned. See `reports/excluded_images.csv`.

## Known Dataset Limitation

Only **27 trustworthy `large_pores` images** remain after conservative filtering in the current audit, compared with 1,195 normal images. The prepared split contains only 18 training, 4 validation, and 5 test `large_pores` images.

This severe imbalance may produce unstable `large_pores` Recall/F1 and makes its five-image test score especially uncertain. Do not claim balanced performance or interpret aggregate accuracy as equal performance across classes. More reviewed `large_pores` data is strongly recommended.

Other limitations include differing source definitions, quality, framing, source bias, and conditions that co-occur on one face. Exact SHA-256 hashes find identical files, not resized or visually similar duplicates.

## Setup

Python 3.11 is recommended and was used with the validated TensorFlow 2.16.1 environment.

Windows PowerShell:

```powershell
git clone https://github.com/Bluedabade/ML-Assignment.git
cd ML-Assignment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_environment.py
```

macOS/Linux:

```bash
git clone https://github.com/Bluedabade/ML-Assignment.git
cd ML-Assignment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_environment.py
```

Optional helpers perform the same environment steps:

```powershell
.\setup.ps1
```

```bash
sh setup.sh
```

## Dataset Audit

```bash
python scripts/audit_dataset.py
```

The audit checks source structure, original labels, COCO/YOLO formats, image formats, unreadable files, exact duplicates, conflicting labels, extra classes, ambiguous multi-condition images, target distribution, and missing targets. It writes:

- `reports/dataset_audit.json`
- `reports/eligible_images.csv`
- `reports/excluded_images.csv`
- `reports/class_distribution.png`

If a target is unavailable, preparation stops with a message such as `MISSING TARGET CLASS: large_pores`.

## Dataset Preparation

Validate sources without copying images:

```bash
python scripts/prepare_dataset.py --validate-only
```

Create the fixed-seed, per-class 70%/15%/15% split:

```bash
python scripts/prepare_dataset.py
```

Intentionally replace an existing generated split:

```bash
python scripts/prepare_dataset.py --clean
```

Preparation writes `reports/dataset_manifest.csv`, `reports/dataset_summary.json`, and `artifacts/class_names.json`. Exact duplicates are resolved before splitting, preventing cross-split leakage.

## Training

Stage 1 freezes the ImageNet backbone. Stage 2 optionally unfreezes its last layers at a smaller learning rate. Validation is used each epoch; test remains unseen until final evaluation.

```bash
python train.py --model mobilenetv2 --epochs 10
python train.py --model efficientnetb0 --epochs 10
python train.py --model resnet50 --epochs 10
```

Useful overrides:

```bash
python train.py --model mobilenetv2 --epochs 10 --batch-size 16 --learning-rate 0.001
python train.py --model mobilenetv2 --fine-tune-epochs 5 --fine-tune-lr 0.00001 --unfreeze-last 30
python train.py --model all --epochs 10
```

Keras downloads ImageNet weights on first use. `--weights none` is only for offline constructor/smoke tests, not final assignment training.

## Evaluation

Training automatically evaluates the best checkpoint once on held-out test data. To repeat evaluation:

```bash
python evaluate.py --model mobilenetv2
python evaluate.py --model efficientnetb0
python evaluate.py --model resnet50
```

Evaluation saves test loss/accuracy, per-class Precision/Recall/F1, macro Precision/Recall/F1, weighted F1, a classification report, and a confusion matrix.

## Model Comparison

```bash
python compare_models.py
```

Only models with completed `run_summary.json` files are included; missing and smoke-only runs are skipped. No results are invented. The table includes Model, Accuracy, Precision, Recall, F1-score, parameters, and training time. Selection is validation-led rather than based on test accuracy alone.

## Predicting an Image

After training:

```bash
python predict.py --model mobilenetv2 --image path/to/unseen_face.jpg
```

A direct checkpoint path is also accepted:

```bash
python predict.py --model artifacts/mobilenetv2/best_model.keras --image path/to/unseen_face.jpg
```

The command loads `artifacts/class_names.json`, prints all five probabilities, and identifies the predicted class and confidence.

## Output Files

```text
artifacts/class_names.json
artifacts/<model>/best_model.keras                 Ignored by Git
reports/plots/<model>_accuracy.png
reports/plots/<model>_loss.png
reports/results/<model>/training_history.png       Combined figure
reports/results/<model>/history.json
reports/results/<model>/metrics.json
reports/results/<model>/classification_report.txt
reports/results/<model>/confusion_matrix.png
reports/results/<model>/run_summary.json
reports/results/model_comparison.csv
```

Model result files appear only after real training/evaluation. No final metrics are included unless a full run actually completed.

## Training Accuracy/Loss Graphs

Each completed run produces:

- **Model Accuracy:** Training Accuracy versus Validation Accuracy by epoch.
- **Model Loss:** Training Loss versus Validation Loss by epoch.

A dashed line marks fine-tuning when enabled. A vertically stacked combined figure is also saved. Test data is never used for epoch curves.

## CPU / GPU

Training works on CPU but is slower, especially for ResNet50. No GPU is hardcoded. Check availability with:

```bash
python check_environment.py
```

GPU setup varies by OS, hardware, drivers, and TensorFlow version. Modern Windows users commonly use WSL2/Linux for supported NVIDIA workflows. If no GPU is found, the scripts safely use CPU.

## Notebook

`notebooks/skin_transfer_learning.ipynb` mirrors the presentation in numbered beginner-friendly sections. It uses repository-relative paths and works in local Jupyter, VS Code, or Colab after cloning.

## Troubleshooting

- **PowerShell blocks activation:** run `Set-ExecutionPolicy -Scope Process RemoteSigned`, then activate again.
- **TensorFlow import error:** confirm Python 3.11, activate `.venv`, and run `pip install -r requirements.txt`.
- **Incorrect/missing dataset folders:** use the exact names under `data/raw/`, then run the audit.
- **Missing target:** inspect the audit and add a trustworthy source. Preparation intentionally stops.
- **No GPU:** CPU is valid. Consult current official TensorFlow/WSL2 documentation if GPU training is needed.
- **Out of memory:** reduce `--batch-size` to 16, 8, or 4.
- **Existing processed data:** use `--validate-only`, or `--clean` only when deliberately regenerating.
- **No comparison results:** complete real training/evaluation. Smoke tests do not count.

## Before Pushing to GitHub

Datasets, processed images, virtual environments, model weights, caches, secrets, and OS files are ignored. Verify every commit:

```bash
git status
git diff
git diff --cached
```

Never force-add datasets or `.keras` files. Repository: https://github.com/Bluedabade/ML-Assignment.git
