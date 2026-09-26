# Annotation Policy

## Draft scope

The September 30 draft is a four-class object detector:

| ID | Canonical class | Roboflow source class |
|---:|---|---|
| 0 | `acne` | `Acne` |
| 1 | `wrinkle` | `Wrinkles` |
| 2 | `dark_spot` | `Dark-Spots` |
| 3 | `enlarged_pore` | `Englarged-Pores` |

The source spelling `Englarged-Pores` is retained in immutable raw metadata and
mapped only in the derived dataset.

## Exclusions

`Blackheads`, `Whiteheads`, `Dry-Skin`, `Oily-Skin`, `Eyebags`,
`Skin-Redness`, and the unused dataset/supercategory entry are excluded. In
particular, blackheads and whiteheads are not automatically merged into acne.

`normal_skin` is not included because there are no real normal-skin bounding-box
annotations. Classification folder labels are not object-detection labels, and
the project does not fabricate full-face boxes.

## Bounding boxes

COCO boxes are clipped to image boundaries only in the derived copy. Each
change is recorded in `datasets/processed/skin_detection/preparation_report.json`.
Boxes that remain non-positive or malformed after clipping are rejected and
reported. YOLO output uses normalized `class_id x_center y_center width height`.

## Empty images

An empty YOLO label file is an explicit negative example, not `normal_skin`.
The preparation report distinguishes:

- A: the source image originally had no COCO objects;
- B: source objects existed, but every object belonged to an excluded class.

## Exact duplicates

SHA-256 is used for deterministic exact-image matching.

- Across splits, priority is `test > val > train`. The highest-priority copy is
  retained and labels are never merged from another copy.
- Within one split, the lexicographically first file is retained only when the
  approved annotation signatures are identical.
- If same-split copies have conflicting approved annotations, the whole group
  is excluded from the derived dataset and listed for human review.
- Cross-split annotation conflicts are recorded explicitly. The label attached
  to the selected highest-priority source record is used without merging.

No oversampling, undersampling, synthetic data, offline augmentation, or split
reshuffling is performed in Phase 2.

