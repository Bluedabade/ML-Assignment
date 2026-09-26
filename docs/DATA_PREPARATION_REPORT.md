# Data Preparation Report

## Scope

This report describes the deterministic Phase 2 conversion from
`datasets/raw/roboflow_skin_problem_clean3/` to
`datasets/processed/skin_detection/`. Raw files were not modified. The Kaggle
classification datasets were not used.

Preparation seed: `42`  
Canonical classes: `acne`, `wrinkle`, `dark_spot`, `enlarged_pore`

## Processed dataset statistics

| Split | Images | Negative images | Objects |
|---|---:|---:|---:|
| train | 1,785 | 692 | 9,810 |
| val | 322 | 176 | 811 |
| test | 248 | 102 | 1,636 |
| **Total** | **2,355** | **970** | **12,257** |

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| acne | 7,495 | 737 | 1,415 | 9,647 |
| wrinkle | 1,815 | 22 | 89 | 1,926 |
| dark_spot | 57 | 19 | 15 | 91 |
| enlarged_pore | 443 | 33 | 117 | 593 |

## Negative images

Every retained image has a matching label file. Negative examples have an empty
YOLO label file and are not labeled as `normal_skin`.

| Split | A: originally no objects | B: all source objects excluded | Total |
|---|---:|---:|---:|
| train | 5 | 687 | 692 |
| val | 0 | 176 | 176 |
| test | 4 | 98 | 102 |
| **Total** | **9** | **961** | **970** |

## Duplicate handling

SHA-256 detected 12 exact-duplicate groups containing 24 source files. Seven
groups crossed split boundaries. Six of those seven had different approved
annotation signatures.

The deterministic policy was:

1. Cross-split duplicates: retain one record using `test > val > train`, without
   merging annotations. Seven lower-priority files were excluded.
2. Same-split duplicates with identical approved annotations: retain the
   lexicographically first filename. One duplicate file was excluded.
3. Same-split duplicates with conflicting approved annotations: exclude the
   entire group for human review. Four groups/eight files were excluded.

In total, 16 duplicate source files were not copied. The processed dataset has
zero remaining exact duplicate groups and zero cross-split hash leakage. Full
group membership and actions are stored in
`datasets/processed/skin_detection/preparation_report.json`.

## Bounding-box corrections

One source box slightly exceeded the lower image boundary and was clipped only
in the derived annotation:

- split: train
- image: `acne-prone-skin_116_jpeg.rf.c88e79ffcb3facb3ccd5c1af5261af58.jpg`
- source annotation ID: 1151 (`Acne`)
- original COCO box: `[227, 237, 394.611, 403.43]`
- clipped COCO box: `[227.0, 237.0, 394.611, 403.0]`

No malformed, non-positive, or otherwise invalid approved annotations were
discarded.

## Excluded source categories

- dataset/supercategory entry (source category ID 0; no objects)
- `Blackheads`
- `Whiteheads`
- `Dry-Skin`
- `Oily-Skin`
- `Eyebags`
- `Skin-Redness`

Blackheads and whiteheads were not merged into acne.

## Validation

`python scripts/validate_dataset.py` completed with `PASS` after a repeat
preparation run. It found:

- 0 unreadable images
- 0 malformed label lines
- 0 out-of-range class IDs
- 0 invalid normalized coordinates
- 0 images without labels
- 0 labels without images
- 0 duplicate filenames
- 0 remaining exact duplicate images
- 0 cross-split exact duplicate leakage

## Limitations before training

- Acne dominates the object distribution.
- Dark spots have only 91 boxes in total.
- Validation has only 22 wrinkle and 33 enlarged-pore objects.
- There are 970 negative images, most created because all source objects belong
  to excluded categories; this is intentional but should be monitored.
- Visual checks show dense, overlapping small boxes and possible source-label
  inconsistency. Human review is recommended before treating metrics as final.
- `normal_skin` is absent because no real bounding-box annotations are available.

