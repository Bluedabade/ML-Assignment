# Dataset licenses and attribution

The committed `data/processed/` snapshot contains only images selected by the
project's deterministic audit/preparation pipeline. Provenance for every image
is recorded in `reports/dataset_manifest.csv`.

## Acne-Wrinkles-Spots-Classification

- Source: https://www.kaggle.com/datasets/ranvijaybalbir/acne-wrinkles-spots-classification
- Publisher: ranvijaybalbir
- License displayed by Kaggle: CC0 1.0 Public Domain Dedication
- License: https://creativecommons.org/publicdomain/zero/1.0/

## Oily, Dry and Normal Skin Types Dataset

- Source: https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset
- Publisher: Shakya Dissanayake
- License displayed by Kaggle: Apache License 2.0
- A copy is included at `LICENSES/Apache-2.0.txt`.

## Skin-Problem-Detection-Relabel-Clean3

- Source: https://universe.roboflow.com/parin-kittipongdaja-vwmn3/skin-problem-detection-relabel-clean3
- Provider: Roboflow user `parin-kittipongdaja-vwmn3`
- Export/version: v2, exported February 5, 2024
- License declared in the source README and COCO metadata: CC BY 4.0
- License: https://creativecommons.org/licenses/by/4.0/

Changes made by this project: images were conservatively selected according to
their labels, exact duplicates were removed, ambiguous images were excluded,
and the remaining images were renamed and assigned to a deterministic
70%/15%/15% classification split. WebP-encoded source files were re-encoded as
high-quality JPEG so TensorFlow's standard image loader can decode them. No
claim of endorsement by the source publishers is made.
