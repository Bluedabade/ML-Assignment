# D031-D063 human review

Open the six contact sheets at 100% zoom and scroll vertically. Each sheet has ONE pair per row, without shrinking source images. A is on the left, B on the right. No facial region is cropped. Individual D031.jpg through D063.jpg are for detailed inspection. Review previews add overlays only; source images and annotations remain untouched.

Each side shows target classes/counts, source COCO image ID and source annotation IDs. IDs are scoped to their original source split and do not prove image identity. ANNOTATION DIFFERENCE means stored target class/count or normalized box geometry differs; it is not a judgment of annotation correctness. Tiny coordinate differences can trigger the geometry flag. Different annotation IDs alone do not trigger the flag.

## Human decision legend

- A_DUPLICATE: effectively the exact same image; negligible encoding differences.
- B_VARIANT: same source image, resized, recompressed, cropped, brightness-adjusted or re-exported.
- C_SAME_SESSION: same person/session but genuinely different photo/frame/pose.
- D_SIMILAR: only visually similar; different source image.
- UNCERTAIN: insufficient evidence.

Only A_DUPLICATE and B_VARIANT should normally be treated as cross-split leakage. dHash/RMSE alone must not decide a label. C/D must not be removed automatically. This phase writes NO decisions to the original CSV: D031-D063 remain PENDING / UNKNOWN.

The current discussion proposes D001-D030 are likely B_VARIANT/same-source variants, but this is not a formal recorded decision. Their existing CSV rows and preview files are preserved without updates. Explicit approval is required before writing decisions.

## Candidate graph (not confirmed duplicate groups)

candidate_duplicate_groups.csv contains connected components using ONLY D031-D063 candidate edges. repeated_file_candidates.csv lists exact filename/split nodes occurring in more than one pair. Filenames include split to avoid accidental merging. These are review aids to prevent inconsistent pair-by-pair deletion, not permission to remove files. Relationships to D001-D030 are outside this report's scope and must be reconciled before any later group-level cleanup.

No raw/baseline images, annotations, split membership, existing review outputs or training results are changed. No V2 or training is authorized.
