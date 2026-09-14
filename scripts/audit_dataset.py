"""INITIAL STAGE: inspect sources, labels, formats, integrity, and ambiguity."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (CLASS_NAMES, COCO_CLASS_MAP, FOLDER_CLASS_MAP, IMAGE_EXTENSIONS,
                    REPORTS_DIR, SOURCE_NAMES, locate_source)

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required. Run: pip install -r requirements.txt") from exc

def relative_path(path: Path) -> str:
    """Store portable repository-relative provenance paths."""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()

def image_info(path: Path) -> tuple[str | None, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest, None
    except (OSError, ValueError) as exc:
        return None, f"corrupted/unreadable image: {exc}"

def folder_records(source_name: str, root: Path):
    eligible, excluded = [], []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        label_dir = path.parent.name
        target = FOLDER_CLASS_MAP.get(label_dir.casefold().replace("-", " "))
        record = {"file": relative_path(path), "source_dataset": source_name,
                  "original_label": label_dir, "target_class": target or ""}
        digest, error = image_info(path)
        record["sha256"] = digest or ""
        if error: record["reason"] = error; excluded.append(record)
        elif not target:
            record["reason"] = f"unmapped/extra folder label: {label_dir}"; excluded.append(record)
        else: eligible.append(record)
    return eligible, excluded

def coco_records(source_name: str, root: Path):
    eligible, excluded = [], []
    json_files = sorted(root.glob("*/_annotations.coco.json"))
    if not json_files:
        return [], [{"file": relative_path(root), "source_dataset": source_name, "original_label": "",
                     "target_class": "", "sha256": "", "reason": "COCO annotation JSON not found"}]
    for annotation_path in json_files:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        names = {item["id"]: item["name"] for item in data.get("categories", [])}
        labels_by_image: dict[int, set[str]] = defaultdict(set)
        for annotation in data.get("annotations", []):
            labels_by_image[annotation["image_id"]].add(names.get(annotation["category_id"], "UNKNOWN"))
        for item in data.get("images", []):
            path = annotation_path.parent / item["file_name"]
            original = sorted(labels_by_image.get(item["id"], set()))
            targets = {COCO_CLASS_MAP[label] for label in original if label in COCO_CLASS_MAP}
            extras = [label for label in original if label not in COCO_CLASS_MAP]
            record = {"file": relative_path(path), "source_dataset": source_name,
                      "original_label": "|".join(original),
                      "target_class": next(iter(targets)) if len(targets) == 1 and not extras else ""}
            if not path.exists():
                record.update(sha256="", reason="annotation references missing image"); excluded.append(record); continue
            digest, error = image_info(path); record["sha256"] = digest or ""
            if error: record["reason"] = error
            elif not original: record["reason"] = "image has no COCO annotations"
            elif extras: record["reason"] = "contains extra/non-target annotations: " + ", ".join(extras)
            elif len(targets) != 1: record["reason"] = "contains multiple target classes: " + ", ".join(sorted(targets))
            else: eligible.append(record); continue
            excluded.append(record)
    return eligible, excluded

def collect_records():
    eligible, excluded, structures = [], [], []
    for name in SOURCE_NAMES:
        root = locate_source(name)
        if root is None:
            structures.append({"source": name, "path": "MISSING", "format": "unknown", "images": 0})
            continue
        images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        coco = list(root.glob("*/_annotations.coco.json"))
        yolo = list(root.rglob("data.yaml")) + list(root.rglob("data.yml"))
        annotation_txt = [p for p in root.rglob("*.txt") if p.name.lower() not in {"readme.dataset.txt", "readme.roboflow.txt"}]
        kind = "COCO object detection" if coco else "folder classification"
        if coco:
            annotation_counts, image_ids_by_label = Counter(), defaultdict(set)
            for coco_path in coco:
                data = json.loads(coco_path.read_text(encoding="utf-8"))
                category_names = {c["id"]: c["name"] for c in data.get("categories", [])}
                for annotation in data.get("annotations", []):
                    label = category_names.get(annotation["category_id"], "UNKNOWN")
                    annotation_counts[label] += 1
                    image_ids_by_label[label].add((coco_path.parent.name, annotation["image_id"]))
            original_classes = {label: {"annotations": annotation_counts[label],
                                        "images": len(image_ids_by_label[label])}
                                for label in sorted(annotation_counts)}
        else:
            original_classes = dict(sorted(Counter(p.parent.name for p in images).items()))
        structures.append({"source": name, "path": relative_path(root), "format": kind,
                           "images": len(images), "coco_json": len(coco),
                           "yolo_yaml": len(yolo), "annotation_txt": len(annotation_txt),
                           "extensions": dict(Counter(p.suffix.lower() for p in images)),
                           "original_classes": original_classes})
        good, bad = coco_records(name, root) if coco else folder_records(name, root)
        eligible.extend(good); excluded.extend(bad)

    # Exact bytes identify leakage candidates. Keep one deterministic copy only.
    by_hash = defaultdict(list)
    for record in eligible: by_hash[record["sha256"]].append(record)
    deduplicated = []
    for digest, records in by_hash.items():
        labels = {r["target_class"] for r in records}
        if len(labels) > 1:
            for r in records: r["reason"] = "duplicate image has conflicting target labels"; excluded.append(r)
        else:
            deduplicated.append(records[0])
            for r in records[1:]: r["reason"] = f"exact duplicate of {records[0]['file']}"; excluded.append(r)
    return deduplicated, excluded, structures

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "source_dataset", "original_label", "target_class", "sha256", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)

def run_audit(make_plot: bool = True):
    eligible, excluded, structures = collect_records()
    counts = Counter(r["target_class"] for r in eligible)
    original = Counter((r["source_dataset"], r["original_label"]) for r in eligible + excluded)
    exclusion_summary = Counter()
    for record in excluded:
        reason = record.get("reason", "")
        if reason.startswith("corrupted/unreadable"):
            exclusion_summary["corrupted_or_unreadable"] += 1
        elif reason.startswith("exact duplicate"):
            exclusion_summary["exact_duplicates"] += 1
        elif reason == "duplicate image has conflicting target labels":
            exclusion_summary["conflicting_duplicates"] += 1
        elif "multiple target classes" in reason:
            exclusion_summary["multiple_target_classes"] += 1
        elif "extra/non-target" in reason or "unmapped/extra" in reason:
            exclusion_summary["extra_or_unmapped_labels"] += 1
        else:
            exclusion_summary["other"] += 1
    for category in ("corrupted_or_unreadable", "exact_duplicates", "conflicting_duplicates",
                     "multiple_target_classes", "extra_or_unmapped_labels", "other"):
        exclusion_summary.setdefault(category, 0)
    summary = {"target_order": CLASS_NAMES, "eligible_by_target": {c: counts[c] for c in CLASS_NAMES},
               "eligible_total": len(eligible), "excluded_total": len(excluded),
               "exclusion_summary": dict(exclusion_summary),
               "sources": structures,
               "original_label_distribution": [{"source": s, "label": l, "count": n} for (s,l),n in sorted(original.items())],
               "warnings": ["COCO images are accepted only when all annotations map to one target class.",
                            "Exact byte hashes detect exact duplicates, not visually similar images.",
                            "Labels from different sources may have different definitions and source bias."]}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dataset_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(REPORTS_DIR / "eligible_images.csv", eligible)
    write_csv(REPORTS_DIR / "excluded_images.csv", excluded)
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5)); plt.bar(CLASS_NAMES, [counts[c] for c in CLASS_NAMES])
        plt.title("Eligible Images by Target Class"); plt.ylabel("Images"); plt.xticks(rotation=20)
        plt.tight_layout(); plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=160); plt.close()
    print("\nDATASET AUDIT REPORT")
    print("=" * 60)
    for source in structures:
        print(f"{source['source']}: {source['format']}, {source['images']} images, {source['path']}")
        if source.get("coco_json"): print(f"  COCO JSON: {source['coco_json']}; YOLO YAML: {source['yolo_yaml']}; annotation .txt: {source['annotation_txt']}")
    print("Eligible target distribution:")
    for class_name in CLASS_NAMES: print(f"  {class_name:12} {counts[class_name]}")
    print(f"Excluded: {len(excluded)} (details: reports/excluded_images.csv)")
    missing = [c for c in CLASS_NAMES if counts[c] == 0]
    for class_name in missing: print(f"MISSING TARGET CLASS: {class_name}")
    return eligible, excluded, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib chart")
    args = parser.parse_args()
    run_audit(not args.no_plot)
