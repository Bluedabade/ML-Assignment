# Included prepared dataset

This directory contains the exact reviewed snapshot used by the assignment:

- `train/`: 1,712 images
- `validation/`: 366 images
- `test/`: 372 images
- total: 2,450 images

Class order is explicitly defined in `config.py` and
`artifacts/class_names.json`:

1. normal
2. wrinkle
3. acne
4. dark_spot
5. large_pores

The split was created deterministically by `scripts/prepare_dataset.py` before
publication. `reports/dataset_manifest.csv` records each image's source,
content hash, class, and split. See `DATASET_LICENSES.md` for source attribution
and redistribution licenses.

`large_pores` has only 27 eligible images (18 train, 4 validation, 5 test), so
its per-class Precision, Recall, F1-score, and confusion-matrix results must be
interpreted cautiously.
