# Phase 4B.1 human review

No dataset changes, V2 creation or training have been performed.

## Duplicate review

Open `D001.jpg` through `D063.jpg` and record decisions in `duplicate_review.csv`. `top30_contact_sheet.jpg` is an overview; use the full-size pair images for decisions. Rank is ascending 32x32 grayscale RMSE, then dHash distance, then filenames. This is a review-order heuristic, not duplicate probability. Each image shows its actual target annotations; NONE means the current derived label is empty.

All `human_decision` values start PENDING and `likely_same_source` values start UNKNOWN. Source IDs are COCO IDs scoped to each source split. Source filenames are filenames in our raw export, not proof of the original photographer/collection identity. Different IDs or filenames do not rule out duplicates.

Human decision vocabulary:

- A_EXACT_SAME_SOURCE: exact/same-source duplicate.
- B_SAME_SOURCE_VARIANT: crop, resize or encoding variant of the same source image.
- C_SAME_SUBJECT_DIFFERENT_IMAGE: same person/session, materially different image.
- D_VISUALLY_SIMILAR_ONLY: merely similar.
- UNCERTAIN: evidence insufficient; leave unresolved rather than guessing.

Only confirmed A/B normally qualify as leakage duplicates. C/D must not be removed automatically. For C, flag subject/session leakage separately for future protocol review. Write reviewer name, date and evidence in notes; update likely_same_source only after human judgment. No script applies these decisions.

## Proposed default priority (NOT implemented)

`test > val > train`:

| Confirmed A/B pair | Proposed retained copy | Proposed excluded copy |
|---|---|---|
| train/test | test | train |
| train/val | val | train |
| val/test | test | val, ONLY after separate explicit approval |

Resolve human-confirmed connected groups globally, not independently per pair: one image may occur in several candidate pairs. Do not merge annotations from another split. Rank/candidate links alone do not confirm a connected duplicate group.

## Wrinkle review

Open `../final_wrinkle_review/wrinkle_review.csv` and match its filename/split/annotation ID to `image_manifest.json` and W-numbered images. A box selected for several reasons occupies one CSV row with combined reason flags, and each source image has one preview.

Selection: 20 smallest normalized-area boxes; 20 greatest max(w/h,h/w); 10 widest normalized-width boxes; 10 smallest normalized edge gaps. Ties use split, filename and annotation ID. Sixty selection slots produce 54 distinct boxes in 33 images. CSV aspect_ratio is w/h, not the symmetric ratio used for thinness selection. All displayed boxes' measurements are available in `all_displayed_wrinkle_box_metadata.json`.

Human decisions may be KEEP / CLEAR_ERROR / UNCERTAIN. Record a specific reason and source annotation ID. Geometric extremes are not inherently wrong. One selected test box is descriptive review only; it must not drive training policy. Any proposed test annotation correction requires separate explicit approval and stops dataset implementation.

## Strict N1 review

`../final_n1_review/n1_candidate_images.csv` contains only 46 TRAIN images whose removed categories are exclusively Blackheads, Whiteheads or both. Proposed action is EXCLUDE_FROM_TRAIN_IN_V2_N1; this has NOT been applied. Two mixed cases are listed separately in `mixed_cases_NOT_N1.json` and are not N1 candidates. Individual N-numbered previews follow CSV row order; `all46_contact_sheet.jpg` shows all candidates.

## Proposed V2 Experiment 1 (NOT created)

Assuming human review identifies no major wrinkle corrections:

1. Keep all positive annotations unchanged on retained images.
2. Keep validation/test membership and annotations unchanged in this first experiment.
3. Exclude only human-confirmed A/B leakage copies from TRAIN where possible, retaining the val/test representative. Do not move images between splits.
4. Apply strict N1 to training negatives, not N2.
5. Union exclusions by filename/split: N1 and duplicate exclusions may overlap and must not be subtracted twice.
6. Keep epochs, imgsz, augmentations and hyperparameters unchanged for the first ablation. Do not use test metrics to choose cleanup policy.

Removing a duplicate training image also removes its annotations; "unchanged positives" means no box edits on retained images. Log those unavoidable object-count reductions explicitly. If confirmed leakage is only val/test, hold it for a separately approved protocol change: TRAIN-only cleanup cannot remove that leakage. Do not claim the test is independent while such leakage remains unresolved.

This package prepares review only. Human approval and a machine-readable change manifest are required before any V2 creation. No training is authorized by a completed review.
