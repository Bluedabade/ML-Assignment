"""Create a reproducible, deduplicated 70/15/15 classification dataset."""
from __future__ import annotations
import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_dataset import run_audit
from config import CLASS_NAMES, PROCESSED_DATA_DIR, REPORTS_DIR, SEED
from src.utils import save_json
from PIL import Image

def split_class(records: list[dict], seed: int):
    items = records[:]; random.Random(seed).shuffle(items); n = len(items)
    n_train = int(n * .70); n_val = int(n * .15)
    return {"train": items[:n_train], "validation": items[n_train:n_train+n_val],
            "test": items[n_train+n_val:]}

def main(output: Path, seed: int, clean: bool, validate_only: bool = False) -> None:
    eligible, _, summary = run_audit(make_plot=True)
    missing = [c for c in CLASS_NAMES if summary["eligible_by_target"][c] == 0]
    if missing:
        for name in missing: print(f"MISSING TARGET CLASS: {name}")
        raise SystemExit("Dataset preparation stopped. Add/review the missing source class and audit again.")
    if validate_only:
        print("Dataset source validation passed: all five target classes have eligible images; no files copied.")
        return
    existing = [path for path in output.iterdir() if path.name != ".gitkeep"] if output.exists() else []
    if existing and not clean:
        raise SystemExit(f"{output} is not empty. Re-run with --clean to replace only this generated directory.")
    if clean and output.exists():
        resolved, root = output.resolve(), ROOT.resolve()
        if root not in resolved.parents: raise SystemExit(f"Refusing to clean outside repository: {resolved}")
        # Empty generated children while keeping the tracked directory itself.
        for child in resolved.iterdir():
            if child.name in {".gitkeep", "README.md"}:
                continue
            if child.is_dir(): shutil.rmtree(child)
            else: child.unlink()
    grouped = defaultdict(list)
    for row in eligible: grouped[row["target_class"]].append(row)
    manifest = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        splits = split_class(grouped[class_name], seed + class_index)
        for split, rows in splits.items():
            destination_dir = output / split / class_name; destination_dir.mkdir(parents=True, exist_ok=True)
            for index, row in enumerate(rows):
                source = ROOT / row["file"]
                destination = destination_dir / f"{row['sha256'][:16]}_{index:05d}{source.suffix.lower()}"
                with Image.open(source) as image:
                    source_format = image.format
                    if source_format in {"JPEG", "PNG", "GIF", "BMP"}:
                        shutil.copy2(source, destination)
                    else:
                        # TensorFlow's standard directory loader cannot decode
                        # formats such as WebP, even when Pillow can. Re-encode
                        # at high JPEG quality while preserving split/class.
                        destination = destination.with_suffix(".jpg")
                        image.convert("RGB").save(destination, format="JPEG",
                                                  quality=95, optimize=True)
                manifest.append({**row, "split": split,
                                 "processed_file": destination.resolve().relative_to(ROOT.resolve()).as_posix()})
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(manifest[0])
    with (REPORTS_DIR / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(manifest)
    counts = Counter((r["split"], r["target_class"]) for r in manifest)
    dataset_summary = {split: {c: counts[(split,c)] for c in CLASS_NAMES}
                       for split in ("train", "validation", "test")}
    save_json(REPORTS_DIR / "dataset_summary.json", dataset_summary)
    save_json(ROOT / "artifacts" / "class_names.json", CLASS_NAMES)
    print("\nPREPARED DATASET (70/15/15 per class)")
    for split, values in dataset_summary.items(): print(f"{split:5}: {values} (total={sum(values.values())})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--clean", action="store_true", help="Replace the generated output directory")
    parser.add_argument("--validate-only", action="store_true", help="Audit and validate without copying images")
    args = parser.parse_args(); main(args.output, args.seed, args.clean, args.validate_only)
