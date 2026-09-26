"""Validate the derived four-class YOLO detection dataset."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CLASS_NAMES = {0: "acne", 1: "wrinkle", 2: "dark_spot", 3: "enlarged_pore"}
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(dataset_root: Path) -> tuple[dict, list[str], list[str]]:
    summary = {}
    errors: list[str] = []
    warnings: list[str] = []
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    global_names: dict[str, list[str]] = defaultdict(list)

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"Missing split directories for {split}")
            continue

        images = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
        image_by_stem = {path.stem: path for path in images}
        label_by_stem = {path.stem: path for path in labels}

        if len(image_by_stem) != len(images):
            errors.append(f"{split}: duplicate image stems make image-label pairing ambiguous")
        if len(label_by_stem) != len(labels):
            errors.append(f"{split}: duplicate label stems found")

        images_without_labels = sorted(set(image_by_stem) - set(label_by_stem))
        labels_without_images = sorted(set(label_by_stem) - set(image_by_stem))
        if images_without_labels:
            errors.append(f"{split}: {len(images_without_labels)} images without label files")
        if labels_without_images:
            errors.append(f"{split}: {len(labels_without_images)} labels without images")

        object_counts = Counter()
        negative_images = 0
        empty_labels = 0
        unreadable_images = 0
        malformed_lines = 0

        for image_path in images:
            global_names[image_path.name.lower()].append(f"{split}/{image_path.name}")
            hashes[sha256_file(image_path)].append((split, image_path.name))
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:  # Pillow exposes multiple decoder exception types.
                unreadable_images += 1
                errors.append(f"{split}/{image_path.name}: unreadable image ({exc})")

            label_path = label_by_stem.get(image_path.stem)
            if label_path is None:
                continue
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
                negative_images += 1
                continue
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                location = f"{split}/{label_path.name}:{line_number}"
                if len(fields) != 5:
                    malformed_lines += 1
                    errors.append(f"{location}: expected 5 YOLO fields, got {len(fields)}")
                    continue
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except ValueError:
                    malformed_lines += 1
                    errors.append(f"{location}: class ID or coordinates are not numeric")
                    continue
                if class_id not in CLASS_NAMES:
                    errors.append(f"{location}: class ID {class_id} is outside 0..3")
                if not all(0.0 <= value <= 1.0 for value in coordinates):
                    errors.append(f"{location}: coordinates are outside 0..1")
                if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                    errors.append(f"{location}: width and height must be > 0")
                if class_id in CLASS_NAMES:
                    object_counts[class_id] += 1

        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "negative_images": negative_images,
            "empty_labels": empty_labels,
            "unreadable_images": unreadable_images,
            "malformed_lines": malformed_lines,
            "images_without_labels": len(images_without_labels),
            "labels_without_images": len(labels_without_images),
            "objects": sum(object_counts.values()),
            "objects_per_class": {CLASS_NAMES[class_id]: object_counts[class_id] for class_id in CLASS_NAMES},
        }

    duplicate_filenames = {name: paths for name, paths in global_names.items() if len(paths) > 1}
    duplicate_hashes = {digest: paths for digest, paths in hashes.items() if len(paths) > 1}
    cross_split_hashes = {
        digest: paths for digest, paths in duplicate_hashes.items() if len({split for split, _ in paths}) > 1
    }
    if duplicate_filenames:
        errors.append(f"Duplicate filenames found: {len(duplicate_filenames)} groups")
    if duplicate_hashes:
        errors.append(f"Exact duplicate images remain: {len(duplicate_hashes)} groups")
    if cross_split_hashes:
        errors.append(f"Cross-split exact duplicate leakage remains: {len(cross_split_hashes)} groups")

    summary["duplicates"] = {
        "duplicate_filename_groups": len(duplicate_filenames),
        "exact_duplicate_groups": len(duplicate_hashes),
        "cross_split_exact_duplicate_groups": len(cross_split_hashes),
    }
    if not (dataset_root / "data.yaml").is_file():
        errors.append("Missing data.yaml")
    return summary, errors, warnings


def print_summary(summary: dict, errors: list[str], warnings: list[str]) -> None:
    print("\n=== YOLO Dataset Validation ===")
    for split in SPLITS:
        if split not in summary:
            continue
        row = summary[split]
        print(
            f"{split:>5}: images={row['images']}, labels={row['labels']}, "
            f"negative={row['negative_images']}, objects={row['objects']}"
        )
        print("       " + ", ".join(f"{name}={count}" for name, count in row["objects_per_class"].items()))
        print(
            f"       unreadable={row['unreadable_images']}, malformed={row['malformed_lines']}, "
            f"images_without_labels={row['images_without_labels']}, "
            f"labels_without_images={row['labels_without_images']}"
        )
    duplicates = summary.get("duplicates", {})
    print(
        "Duplicates: "
        f"filenames={duplicates.get('duplicate_filename_groups', 0)}, "
        f"exact={duplicates.get('exact_duplicate_groups', 0)}, "
        f"cross_split={duplicates.get('cross_split_exact_duplicate_groups', 0)}"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print("PASS" if not errors else f"FAIL ({len(errors)} error(s))")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "datasets" / "processed" / "skin_detection",
    )
    args = parser.parse_args()
    summary, errors, warnings = validate(args.dataset_root.resolve())
    print_summary(summary, errors, warnings)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
