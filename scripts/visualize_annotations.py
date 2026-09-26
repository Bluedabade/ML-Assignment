"""Draw deterministic samples of processed YOLO annotations for human review."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = {0: "acne", 1: "wrinkle", 2: "dark_spot", 3: "enlarged_pore"}
COLORS = {0: "#ef4444", 1: "#3b82f6", 2: "#a855f7", 3: "#22c55e"}
SPLITS = ("train", "val", "test")


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, width, height = line.split()
        rows.append((int(class_id), float(x), float(y), float(width), float(height)))
    return rows


def collect_candidates(dataset_root: Path) -> dict[int, list[tuple[str, Path, Path]]]:
    candidates: dict[int, list[tuple[str, Path, Path]]] = defaultdict(list)
    for split in SPLITS:
        for label_path in sorted((dataset_root / "labels" / split).glob("*.txt")):
            image_matches = sorted((dataset_root / "images" / split).glob(f"{label_path.stem}.*"))
            if len(image_matches) != 1:
                continue
            present_classes = {row[0] for row in read_labels(label_path)}
            for class_id in present_classes:
                if class_id in CLASS_NAMES:
                    candidates[class_id].append((split, image_matches[0], label_path))
    return candidates


def draw_sample(image_path: Path, label_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    image_width, image_height = image.size
    line_width = max(2, round(min(image.size) / 250))
    for class_id, x_center, y_center, width, height in read_labels(label_path):
        x1 = (x_center - width / 2.0) * image_width
        y1 = (y_center - height / 2.0) * image_height
        x2 = (x_center + width / 2.0) * image_width
        y2 = (y_center + height / 2.0) * image_height
        color = COLORS[class_id]
        label = CLASS_NAMES[class_id]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_height - 4)
        draw.rectangle((x1, text_y, x1 + text_width + 6, text_y + text_height + 4), fill=color)
        draw.text((x1 + 3, text_y + 2), label, fill="white", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "datasets" / "processed" / "skin_detection",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "results" / "annotation_samples",
    )
    parser.add_argument("--samples-per-class", type=int, default=5)
    args = parser.parse_args()
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1")

    candidates = collect_candidates(args.dataset_root.resolve())
    manifest = {"selection": "lexicographic by split order train, val, test", "classes": {}}
    for class_id, class_name in CLASS_NAMES.items():
        selected = candidates[class_id][: args.samples_per_class]
        class_output = args.output_root.resolve() / class_name
        class_output.mkdir(parents=True, exist_ok=True)
        manifest["classes"][class_name] = []
        for index, (split, image_path, label_path) in enumerate(selected, start=1):
            output_path = class_output / f"{index:02d}_{split}_{image_path.stem}.jpg"
            draw_sample(image_path, label_path, output_path)
            manifest["classes"][class_name].append(
                {
                    "split": split,
                    "source": str(image_path.relative_to(root)),
                    "output": str(output_path.relative_to(root)),
                }
            )
        print(f"{class_name}: wrote {len(selected)} sample(s) to {class_output}")
    (args.output_root.resolve() / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
