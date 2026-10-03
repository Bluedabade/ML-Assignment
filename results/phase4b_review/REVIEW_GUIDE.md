# Phase 4B dataset quality review

Baseline: `a2ccb68`. No dataset, annotation, split, baseline checkpoint or training result was changed. No training was run. V2 has NOT been created.

## Open previews

- `grids/wrinkle/train/`: 25 training images, prioritizing geometric flags.
- `grids/wrinkle/val/`: all 8 validation wrinkle images.
- `grids/wrinkle/test/`: 25 test images, descriptive audit only. Do not use these to choose policy.
- `wrinkle/<split>/`: full-size versions; box labels include original source annotation IDs.
- `grids/negatives/<original categories>/`: 100 training negatives sampled round-robin across category combinations with seed 42. Not prevalence-weighted.
- `negatives/<original categories>/`: full-size images showing REMOVED source boxes, not current training labels. Current labels are empty.
- `duplicate_pairs/`: 30 closest cross-split candidate pairs for human comparison. Full candidate list is `near_duplicate_candidates.json`.

Review at full resolution where grids are insufficient. Boxes may overlap visually without reaching the IoU threshold. No AI annotation corrections or final human decisions have been recorded.

## Findings

Train has 214 wrinkle-positive images / 1,815 objects, versus validation 8 / 22. Training previews mix narrow individual lines, larger region boxes and boxes around cheek/nasolabial areas. Some boxes on low-texture forehead areas need human inspection. This suggests policy inconsistency; it does not prove every flagged box is wrong or establish the cause of poor detection.

336 training wrinkle boxes are tiny, 263 thin, 29 near boundaries. No same-image wrinkle pairs have IoU >=0.8. No wrinkle boxes exceed 25% image area, but one training box is 98.4% of image width. Full distributions and source IDs are in `summary.json` and `wrinkle_boxes.json`.

Training negatives: 692 total, 5 originally empty and 687 empty after filtering. Exclusive combinations: Dry-Skin 351; Oily-Skin 135; Eyebags 101; Skin-Redness 52; Blackheads 27; Whiteheads 8; Blackheads+Whiteheads 11; Blackheads+Eyebags 1; Blackheads+Oily-Skin 1. Thus 48 are POTENTIAL CONFLICTING NEGATIVES, and 46 meet N1's exclusive Blackheads/Whiteheads rule. Original labels do not prove absence of unannotated target conditions. These are not automatically normal_skin.

Raw exact duplicates: 12 groups, including 7 cross-split groups. Existing baseline processing already handled these; processed exact duplicate groups are zero. New screening found 127 near-duplicate candidate pairs, 63 cross-split. These are not confirmed duplicates or independent groups. Crops, subject overlap and other variants may escape dHash screening.

## Screening definitions (not removal rules)

- Tiny: minimum side <4 pixels OR normalized area <0.0001.
- Thin: max(width/height, height/width) >=10.
- Large: normalized area >0.25.
- Boundary: gap from any image edge <=1% of that dimension.
- Overlap: same-image wrinkle IoU >=0.8.
- Near duplicate: 64-bit dHash Hamming distance <=4 AND grayscale 32x32 RMSE <=0.08.

## Conservative proposed V2 policy

Wrinkle: retain all boxes until human review identifies clear errors by source annotation ID. Do not automatically remove tiny/thin boxes, expand boxes or invent new boxes. A region-level rewrite would change the task and needs explicit approval. Validation/test annotations stay unchanged. Any proposed test correction requires a separate approval and stops implementation.

| Negative policy | Removed train images | Remaining train images | Remaining train negatives | Trade-off |
|---|---:|---:|---:|---|
| N0 keep all | 0 | 1785 | 692 | Preserves baseline and hard-negative coverage; possible label conflicts remain |
| N1 exclude negatives whose ONLY categories are Blackheads/Whiteheads | 46 | 1739 | 646 | Small, objective change; still loses useful hard negatives and leaves mixed/conflicting examples |
| N2 exclude all filter-created negatives | 687 | 1098 | 5 | Removes possible missing-target negatives but almost eliminates negative training coverage |
| N3 controlled retained subset | Not fixed | Not fixed | Not fixed | Requires human-reviewed eligibility and a predeclared retention rule; count not invented |

Recommend N1 as the first *proposed* negative-policy comparison, with all positive labels unchanged, only after human review/approval and resolution of cross-split duplicate candidates. N1 is not guaranteed to improve detection. Do not select policies using test evidence. Retained images keep their original splits. N1 changes train membership only; val/test membership and annotations remain identical.

All N0/N1/N2 object counts and class-positive image counts remain unchanged:

| Split | Acne images/objects | Wrinkle images/objects | Dark spot images/objects | Enlarged pore images/objects |
|---|---:|---:|---:|---:|
| train | 895 / 7495 | 214 / 1815 | 32 / 57 | 227 / 443 |
| val | 132 / 737 | 8 / 22 | 12 / 19 | 15 / 33 |
| test | 131 / 1415 | 30 / 89 | 9 / 15 | 56 / 117 |

Classes can coexist; class-positive image counts must not be summed to get total images. These estimates assume no wrinkle corrections or duplicate exclusions; those changes require actual reviewed decisions before counts can be recomputed.

## Record human decisions

Use filenames/splits and source annotation IDs from the JSON manifests. Record reviewer, decision, reason and affected annotation IDs separately from original data. For duplicate candidates record CONFIRMED_DUPLICATE / NOT_DUPLICATE / UNCERTAIN. Do not edit raw or baseline files. No decision here authorizes training.

`summary.json`, `negative_training_images.json`, `wrinkle_boxes.json`, `wrinkle_overlap_pairs.json`, `raw_exact_duplicate_groups.json`, `near_duplicate_candidates.json`, `review_manifest.json`, and `class_image_counts.json` provide machine-readable evidence. `protected_files_sha256.json` records before/after verified hashes for 10,961 protected files in raw, baseline processed data and runs.

Baseline validator: PASS (zero unreadable images, malformed labels, missing pairings or exact duplicates). V2 validation: NOT_RUN, because V2 was not created. Structural PASS does not establish annotation correctness or absence of near/subject leakage.

Status: HUMAN_REVIEW_PENDING / V2_NOT_CREATED / TRAINING_NOT_RUN.
