"""Prepare the four-class skin-condition YOLO dataset from immutable COCO data.

The script never edits datasets/raw. It filters approved categories, resolves exact
duplicate leakage deterministically, clips boxes to image bounds, and writes a new
derived dataset plus machine-readable preparation reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEED = 42
SPLITS = ("train", "val", "test")
SOURCE_SPLITS = {"train": "train", "val": "valid", "test": "test"}
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
CLASS_NAMES = {0: "acne", 1: "wrinkle", 2: "dark_spot", 3: "enlarged_pore"}
SOURCE_TO_CANONICAL = {
    "Acne": 0,
    "Wrinkles": 1,
    "Dark-Spots": 2,
    "Englarged-Pores": 3,
}
EXCLUDED_CLASSES = {
    "Acne-Blackhead-Wrinkles-f6HR-vJCz-Vyox",
    "Blackheads",
    "Whiteheads",
    "Dry-Skin",
    "Oily-Skin",
    "Eyebags",
    "Skin-Redness",
}


@dataclass(frozen=True)
class SourceImage:
    split: str
    source_dir: Path
    path: Path
    image: dict[str, Any]
    annotations: tuple[dict[str, Any], ...]
    category_names: dict[int, str]
    sha256: str

    @property
    def approved_signature(self) -> tuple[tuple[Any, ...], ...]:
        signature = []
        for annotation in self.annotations:
            source_name = self.category_names[int(annotation["category_id"])]
            if source_name not in SOURCE_TO_CANONICAL:
                continue
            bbox = tuple(round(float(value), 6) for value in annotation["bbox"])
            signature.append((SOURCE_TO_CANONICAL[source_name], *bbox))
        return tuple(sorted(signature))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_recreate_output(output_root: Path, root: Path) -> None:
    expected_parent = (root / "datasets" / "processed").resolve()
    resolved = output_root.resolve()
    if resolved.parent != expected_parent or resolved.name != "skin_detection":
        raise RuntimeError(f"Refusing to recreate unexpected output path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    for split in SPLITS:
        (resolved / "images" / split).mkdir(parents=True, exist_ok=True)
        (resolved / "labels" / split).mkdir(parents=True, exist_ok=True)
    (resolved / "annotations").mkdir(parents=True, exist_ok=True)


def load_source_images(raw_root: Path) -> list[SourceImage]:
    records: list[SourceImage] = []
    for output_split, source_split in SOURCE_SPLITS.items():
        source_dir = raw_root / source_split
        annotation_path = source_dir / "_annotations.coco.json"
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {int(row["id"]): str(row["name"]) for row in data["categories"]}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        for image in data["images"]:
            image_path = source_dir / image["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(f"COCO image is missing: {image_path}")
            records.append(
                SourceImage(
                    split=output_split,
                    source_dir=source_dir,
                    path=image_path,
                    image=image,
                    annotations=tuple(annotations_by_image[int(image["id"])]),
                    category_names=categories,
                    sha256=sha256_file(image_path),
                )
            )
    return records


def resolve_duplicates(records: list[SourceImage]) -> tuple[list[SourceImage], dict[str, Any]]:
    groups: dict[str, list[SourceImage]] = defaultdict(list)
    for record in records:
        groups[record.sha256].append(record)

    kept: list[SourceImage] = []
    duplicate_groups = []
    excluded_conflicting_same_split = []
    removed_paths: set[Path] = set()

    def source_ref(item: SourceImage) -> str:
        return f"{SOURCE_SPLITS[item.split]}/{item.path.name}"

    for digest, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: (-SPLIT_PRIORITY[item.split], item.path.name))
        if len(group) == 1:
            kept.append(group[0])
            continue

        splits = {item.split for item in group}
        signatures_identical = len({item.approved_signature for item in group}) == 1
        entry = {
            "sha256": digest,
            "cross_split": len(splits) > 1,
            "annotations_identical_after_filtering": signatures_identical,
            "members": [
                {
                    "split": item.split,
                    "source_file": source_ref(item),
                    "approved_object_count": len(item.approved_signature),
                }
                for item in ordered
            ],
        }

        if len(splits) > 1:
            canonical = ordered[0]
            kept.append(canonical)
            removed_paths.update(item.path for item in ordered[1:])
            entry["action"] = "kept_highest_priority_split_without_merging_annotations"
            entry["kept"] = source_ref(canonical)
        elif signatures_identical:
            canonical = sorted(group, key=lambda item: item.path.name)[0]
            kept.append(canonical)
            removed_paths.update(item.path for item in group if item.path != canonical.path)
            entry["action"] = "kept_lexicographically_first_identical_copy"
            entry["kept"] = source_ref(canonical)
        else:
            removed_paths.update(item.path for item in group)
            entry["action"] = "excluded_entire_same_split_conflict_for_human_review"
            excluded_conflicting_same_split.append(entry)

        duplicate_groups.append(entry)

    kept.sort(key=lambda item: (SPLITS.index(item.split), item.path.name))
    report = {
        "policy": "Cross-split priority test > val > train; no annotation merging. "
        "Within a split, keep the lexicographically first copy only when approved "
        "annotations are identical; exclude the entire conflicting group for review.",
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_files": sum(len(row["members"]) for row in duplicate_groups),
        "cross_split_groups": sum(bool(row["cross_split"]) for row in duplicate_groups),
        "conflicting_annotation_groups": sum(
            not row["annotations_identical_after_filtering"] for row in duplicate_groups
        ),
        "removed_duplicate_files": len(removed_paths),
        "same_split_conflict_groups_excluded": len(excluded_conflicting_same_split),
        "groups": duplicate_groups,
    }
    return kept, report


def clip_coco_bbox(
    bbox: list[float], image_width: float, image_height: float
) -> tuple[list[float] | None, bool]:
    if len(bbox) != 4:
        return None, False
    x, y, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return None, False
    x1 = max(0.0, min(image_width, x))
    y1 = max(0.0, min(image_height, y))
    x2 = max(0.0, min(image_width, x + width))
    y2 = max(0.0, min(image_height, y + height))
    clipped = [x1, y1, x2 - x1, y2 - y1]
    if clipped[2] <= 0 or clipped[3] <= 0:
        return None, False
    changed = any(abs(a - b) > 1e-9 for a, b in zip(clipped, [x, y, width, height]))
    return clipped, changed


def yolo_line(class_id: int, bbox: list[float], width: float, height: float) -> str:
    x, y, box_width, box_height = bbox
    x_center = (x + box_width / 2.0) / width
    y_center = (y + box_height / 2.0) / height
    normalized_width = box_width / width
    normalized_height = box_height / height
    values = (x_center, y_center, normalized_width, normalized_height)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Normalized box is outside 0..1: {values}")
    return f"{class_id} " + " ".join(f"{value:.8f}" for value in values)


def write_data_yaml(output_root: Path) -> None:
    content = """# Paths are relative to this data.yaml file.
train: images/train
val: images/val
test: images/test

names:
  0: acne
  1: wrinkle
  2: dark_spot
  3: enlarged_pore
"""
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def prepare(raw_root: Path, output_root: Path, root: Path) -> dict[str, Any]:
    records = load_source_images(raw_root)
    selected, duplicate_report = resolve_duplicates(records)
    safe_recreate_output(output_root, root)

    stats = {
        split: {
            "images": 0,
            "negative_images": 0,
            "originally_empty": 0,
            "empty_after_filtering": 0,
            "objects": 0,
            "objects_per_class": {name: 0 for name in CLASS_NAMES.values()},
        }
        for split in SPLITS
    }
    bbox_corrections: list[dict[str, Any]] = []
    invalid_annotations: list[dict[str, Any]] = []
    empty_images: list[dict[str, Any]] = []
    filtered_coco = {
        split: {
            "info": {
                "description": "Derived four-class skin detection dataset",
                "seed": SEED,
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": class_id, "name": name, "supercategory": "skin_condition"}
                for class_id, name in CLASS_NAMES.items()
            ],
        }
        for split in SPLITS
    }
    output_names: dict[str, set[str]] = {split: set() for split in SPLITS}
    next_image_id = {split: 1 for split in SPLITS}
    next_annotation_id = {split: 1 for split in SPLITS}

    for record in selected:
        split = record.split
        destination_name = record.path.name
        if destination_name.lower() in output_names[split]:
            raise RuntimeError(f"Duplicate output filename in {split}: {destination_name}")
        output_names[split].add(destination_name.lower())

        width = float(record.image["width"])
        height = float(record.image["height"])
        approved_rows: list[tuple[int, list[float], dict[str, Any]]] = []
        for annotation in record.annotations:
            source_class = record.category_names.get(int(annotation["category_id"]))
            if source_class not in SOURCE_TO_CANONICAL:
                continue
            clipped, changed = clip_coco_bbox(annotation.get("bbox", []), width, height)
            if clipped is None:
                invalid_annotations.append(
                    {
                        "split": split,
                        "file": destination_name,
                        "source_annotation_id": annotation.get("id"),
                        "reason": "malformed_or_non_positive_box_after_clipping",
                        "bbox": annotation.get("bbox"),
                    }
                )
                continue
            if changed:
                bbox_corrections.append(
                    {
                        "split": split,
                        "file": destination_name,
                        "source_annotation_id": annotation.get("id"),
                        "source_class": source_class,
                        "original_bbox": annotation["bbox"],
                        "clipped_bbox": clipped,
                    }
                )
            approved_rows.append((SOURCE_TO_CANONICAL[source_class], clipped, annotation))

        shutil.copy2(record.path, output_root / "images" / split / destination_name)
        label_path = output_root / "labels" / split / f"{Path(destination_name).stem}.txt"
        label_lines = [yolo_line(class_id, bbox, width, height) for class_id, bbox, _ in approved_rows]
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        split_stats = stats[split]
        split_stats["images"] += 1
        split_stats["objects"] += len(approved_rows)
        for class_id, _, _ in approved_rows:
            split_stats["objects_per_class"][CLASS_NAMES[class_id]] += 1
        if not approved_rows:
            split_stats["negative_images"] += 1
            reason = "A_originally_no_coco_objects" if not record.annotations else "B_all_objects_excluded"
            split_stats["originally_empty" if not record.annotations else "empty_after_filtering"] += 1
            empty_images.append(
                {
                    "split": split,
                    "file": destination_name,
                    "reason": reason,
                    "source_object_classes": sorted(
                        {
                            record.category_names.get(int(row["category_id"]), "UNKNOWN")
                            for row in record.annotations
                        }
                    ),
                }
            )

        coco_image_id = next_image_id[split]
        next_image_id[split] += 1
        filtered_coco[split]["images"].append(
            {
                "id": coco_image_id,
                "file_name": destination_name,
                "width": int(width),
                "height": int(height),
                "source_split": split,
                "source_image_id": record.image["id"],
                "sha256": record.sha256,
            }
        )
        for class_id, bbox, source_annotation in approved_rows:
            filtered_coco[split]["annotations"].append(
                {
                    "id": next_annotation_id[split],
                    "image_id": coco_image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": int(source_annotation.get("iscrowd", 0)),
                    "source_annotation_id": source_annotation.get("id"),
                }
            )
            next_annotation_id[split] += 1

    for split, coco in filtered_coco.items():
        path = output_root / "annotations" / f"instances_{split}.json"
        path.write_text(json.dumps(coco, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_data_yaml(output_root)
    report = {
        "seed": SEED,
        "raw_source": str(raw_root.relative_to(root)),
        "output": str(output_root.relative_to(root)),
        "canonical_classes": CLASS_NAMES,
        "source_mapping": SOURCE_TO_CANONICAL,
        "excluded_source_classes": sorted(EXCLUDED_CLASSES),
        "normal_skin": "not_included_no_real_bounding_box_annotations",
        "duplicates": duplicate_report,
        "bbox_corrections": bbox_corrections,
        "invalid_annotations": invalid_annotations,
        "empty_images": empty_images,
        "splits": stats,
    }
    (output_root / "preparation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("\n=== Dataset preparation complete ===")
    print(f"Seed: {report['seed']}")
    for split in SPLITS:
        row = report["splits"][split]
        print(
            f"{split:>5}: images={row['images']}, negatives={row['negative_images']}, "
            f"objects={row['objects']}"
        )
        print("       " + ", ".join(f"{name}={count}" for name, count in row["objects_per_class"].items()))
    duplicates = report["duplicates"]
    print(
        "Duplicates: "
        f"groups={duplicates['exact_duplicate_groups']}, "
        f"cross_split={duplicates['cross_split_groups']}, "
        f"removed_files={duplicates['removed_duplicate_files']}, "
        f"conflicts={duplicates['conflicting_annotation_groups']}"
    )
    print(f"Clipped boxes: {len(report['bbox_corrections'])}")
    print(f"Invalid annotations skipped: {len(report['invalid_annotations'])}")
    print(f"Report: datasets/processed/skin_detection/preparation_report.json")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=root / "datasets" / "raw" / "roboflow_skin_problem_clean3",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "datasets" / "processed" / "skin_detection",
    )
    args = parser.parse_args()
    report = prepare(args.raw_root.resolve(), args.output_root.resolve(), root)
    print_summary(report)


if __name__ == "__main__":
    main()
