"""Read-only dataset audit; writes only a new Phase 4B human-review package.

No annotation decisions, dataset creation, training, or test-based policy selection.
Run with python -B scripts/audit_phase4b.py to avoid import caches.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import random
import textwrap
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/processed/skin_detection"
RAW = ROOT / "datasets/raw/roboflow_skin_problem_clean3"
OUT = ROOT / "results/phase4b_review"
CLASSES = ["acne", "wrinkle", "dark_spot", "enlarged_pore"]
SPLITS = ["train", "val", "test"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_snapshot():
    paths = []
    for folder in [ROOT / "datasets/raw", DATA, ROOT / "runs"]:
        paths.extend(p for p in folder.rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def distribution(values):
    if not values:
        return None
    q = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    return dict(zip(["min", "p05", "p25", "median", "p75", "p95", "max"], q.tolist()))


def iou(a, b):
    x, y, w, h = a
    xx, yy, ww, hh = b
    inter = max(0, min(x+w, xx+ww)-max(x, xx)) * max(0, min(y+h, yy+hh)-max(y, yy))
    return inter / (w*h + ww*hh - inter) if w*h + ww*hh > inter else 0


def preview(record, annotations, caption, destination):
    with Image.open(record["path"]) as src:
        im = src.convert("RGB")
    draw = ImageDraw.Draw(im)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
        label = f"{ann['class']} id={ann.get('source_annotation_id', ann.get('id'))}"
        draw.text((max(0, x), max(0, y-12)), label, fill="red", stroke_width=1, stroke_fill="white")
    lines = []
    for line in caption:
        lines.extend(textwrap.wrap(line, 85))
    canvas = Image.new("RGB", (max(640, im.width), im.height + 20*len(lines)+12), "white")
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    for n, line in enumerate(lines):
        d.text((8, im.height+6+n*20), line, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)


def grids(paths, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(paths), 12):
        canvas = Image.new("RGB", (1280, 1080), "white")
        for idx, path in enumerate(paths[start:start+12]):
            with Image.open(path) as src:
                tile = ImageOps.contain(src, (320, 360))
            canvas.paste(tile, ((idx % 4)*320, (idx//4)*360))
        canvas.save(folder / f"grid_{start//12+1:02d}.jpg", quality=92)


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite review package: {OUT}")
    before = protected_snapshot()
    OUT.mkdir(parents=True)
    rng = random.Random(42)
    records, boxes, stats = [], [], {}
    source = {}
    for split, rawsplit in zip(SPLITS, ["train", "valid", "test"]):
        coco = json.loads((RAW / rawsplit / "_annotations.coco.json").read_text())
        cats = {a["id"]: a["name"] for a in coco["categories"]}
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(dict(a, **{"class": cats[a["category_id"]]}))
        source[split] = {im["id"]: anns[im["id"]] for im in coco["images"]}
        derived = json.loads((DATA / "annotations" / f"instances_{split}.json").read_text())
        byimage = defaultdict(list)
        for a in derived["annotations"]:
            byimage[a["image_id"]].append(dict(a, **{"class": CLASSES[a["category_id"]]}))
        summary = Counter()
        for im in derived["images"]:
            path = DATA / "images" / split / im["file_name"]
            with Image.open(path) as opened:
                gray = np.asarray(opened.convert("L").resize((9, 8)), dtype=np.int16)
                thumb = np.asarray(opened.convert("L").resize((32, 32)), dtype=np.float32)/255
                width, height = opened.size
            bits = (gray[:, 1:] > gray[:, :-1]).flatten()
            dhash = sum(int(v) << k for k, v in enumerate(bits))
            original = source[split][im["source_image_id"]]
            categories = sorted({a["class"] for a in original})
            label = DATA / "labels" / split / (path.stem + ".txt")
            yolo = [line.split() for line in label.read_text().splitlines() if line.strip()]
            assert len(yolo) == len(byimage[im["id"]]), f"COCO/YOLO count mismatch: {path}"
            for row, ann in zip(yolo, byimage[im["id"]]):
                assert int(row[0]) == ann["category_id"]
                x, y, w, h = ann["bbox"]
                assert np.allclose(list(map(float, row[1:])), [(x+w/2)/width, (y+h/2)/height, w/width, h/height], atol=2e-6)
            r = dict(split=split, filename=path.name, path=str(path), sha256=digest(path),
                     annotations=byimage[im["id"]], original_categories=categories,
                     source_image_id=im["source_image_id"], dhash=dhash, thumb=thumb)
            records.append(r)
            summary["images"] += 1
            summary["negative_images"] += not yolo
            for a in r["annotations"]:
                summary[a["class"]] += 1
                if a["class"] != "wrinkle":
                    continue
                x, y, w, h = a["bbox"]
                flags = []
                if min(w, h) < 4 or w*h/(width*height) < .0001:
                    flags.append("TINY")
                if w*h/(width*height) > .25:
                    flags.append("LARGE")
                if min(x/width, y/height, (width-x-w)/width, (height-y-h)/height) <= .01:
                    flags.append("BOUNDARY")
                if max(w/h, h/w) >= 10:
                    flags.append("THIN_ASPECT")
                boxes.append(dict(split=split, filename=path.name, source_annotation_id=a.get("source_annotation_id"),
                                  width_px=w, height_px=h, area_px=w*h, width_norm=w/width,
                                  height_norm=h/height, area_norm=w*h/(width*height), aspect_ratio=w/h, flags=flags))
        stats[split] = dict(summary)
    overlaps = []
    for r in records:
        wr = [a for a in r["annotations"] if a["class"] == "wrinkle"]
        for i, a in enumerate(wr):
            for b in wr[i+1:]:
                score = iou(a["bbox"], b["bbox"])
                if score >= .8:
                    overlaps.append(dict(split=r["split"], filename=r["filename"], iou=score,
                                         source_annotation_ids=[a.get("source_annotation_id"), b.get("source_annotation_id")]))
    wrstats = {}
    for split in SPLITS:
        rows = [b for b in boxes if b["split"] == split]
        wrstats[split] = dict(images=len({b["filename"] for b in rows}), objects=len(rows),
                             distributions={key: distribution([b[key] for b in rows]) for key in
                                            ["width_px", "height_px", "area_px", "width_norm", "height_norm", "area_norm", "aspect_ratio"]},
                             flags=dict(Counter(flag for b in rows for flag in b["flags"])),
                             highly_overlapping_pairs=sum(a["split"] == split for a in overlaps))
    negatives = [r for r in records if r["split"] == "train" and not r["annotations"] and r["original_categories"]]
    negrows = [dict(filename=r["filename"], source_image_id=r["source_image_id"], original_categories=r["original_categories"],
                    derived_label_status="EMPTY_AFTER_FILTERING", potential_conflicting_negative=bool(set(r["original_categories"]) & {"Blackheads", "Whiteheads"})) for r in negatives]
    composition = Counter(" + ".join(r["original_categories"]) for r in negatives)
    hashes = defaultdict(list)
    for r in records:
        hashes[r["sha256"]].append(f"{r['split']}/{r['filename']}")
    exact = {k: v for k, v in hashes.items() if len(v)>1}
    near = []
    for i, a in enumerate(records):
        for b in records[i+1:]:
            distance = (a["dhash"] ^ b["dhash"]).bit_count()
            if distance <= 4:
                rmse = float(np.sqrt(np.mean((a["thumb"]-b["thumb"])**2)))
                if rmse <= .08:
                    near.append(dict(a=f"{a['split']}/{a['filename']}", b=f"{b['split']}/{b['filename']}",
                                     hamming=distance, gray_rmse=rmse, cross_split=a["split"] != b["split"], decision="PENDING_HUMAN_REVIEW"))
    preview_manifest = []
    flagged = {(b["split"], b["filename"]) for b in boxes if b["flags"]}
    for split in SPLITS:
        candidates = [r for r in records if r["split"] == split and any(a["class"] == "wrinkle" for a in r["annotations"])]
        rng.shuffle(candidates)
        candidates.sort(key=lambda r: (r["split"], r["filename"]) not in flagged)
        paths = []
        for i, r in enumerate(candidates[:25]):
            path = OUT / "wrinkle" / split / f"{i+1:03d}.jpg"
            preview(r, [a for a in r["annotations"] if a["class"] == "wrinkle"],
                    [f"WRINKLE | source split={split} | HUMAN REVIEW PENDING", r["filename"], "TEST: descriptive audit only; do not select policy using test examples" if split == "test" else "Check box policy and errors; flags are not deletion decisions"], path)
            paths.append(path)
            preview_manifest.append(dict(kind="wrinkle", split=split, filename=r["filename"], preview=str(path.relative_to(ROOT)), decision="PENDING"))
        grids(paths, OUT / "grids/wrinkle" / split)
    groups = defaultdict(list)
    for r in negatives:
        groups["__".join(r["original_categories"])].append(r)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected)<100 and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected)<100:
                selected.append((key, groups[key].pop()))
    grouped_paths = defaultdict(list)
    for i, (key, r) in enumerate(selected):
        path = OUT / "negatives" / key / f"{i+1:03d}.jpg"
        preview(r, source["train"][r["source_image_id"]],
                ["TRAIN | current derived label=EMPTY_AFTER_FILTERING", "Removed categories: " + ", ".join(r["original_categories"]), r["filename"], "POTENTIAL CONFLICTING NEGATIVE" if set(r["original_categories"]) & {"Blackheads", "Whiteheads"} else "Not automatically normal_skin"], path)
        grouped_paths[key].append(path)
        preview_manifest.append(dict(kind="negative", split="train", filename=r["filename"], preview=str(path.relative_to(ROOT)), original_categories=r["original_categories"], decision="PENDING"))
    for key, paths in grouped_paths.items():
        grids(paths, OUT / "grids/negatives" / key)
    n1 = sum(set(r["original_categories"]) <= {"Blackheads", "Whiteheads"} for r in negatives)
    estimates = {}
    for policy, removed in [("N0", 0), ("N1", n1), ("N2", len(negatives))]:
        estimates[policy] = {s: dict(stats[s], images=stats[s]["images"]-(removed if s=="train" else 0),
                                    negative_images=stats[s]["negative_images"]-(removed if s=="train" else 0)) for s in SPLITS}
    prep = json.loads((DATA / "preparation_report.json").read_text())
    rawhash = defaultdict(list)
    for p in sorted(RAW.rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            rawhash[digest(p)].append(str(p.relative_to(RAW)))
    rawduplicates = {k: v for k, v in rawhash.items() if len(v)>1}
    after = protected_snapshot()
    assert before == after, "Protected file contents or membership changed"
    save("protected_files_sha256.json", before)
    save("wrinkle_boxes.json", boxes)
    save("wrinkle_overlap_pairs.json", overlaps)
    save("negative_training_images.json", negrows)
    save("near_duplicate_candidates.json", near)
    save("raw_exact_duplicate_groups.json", rawduplicates)
    save("review_manifest.json", preview_manifest)
    report = dict(seed=42, baseline_commit="a2ccb68", thresholds=dict(tiny="min side <4px OR area fraction <0.0001", large="area fraction >0.25", boundary="edge gap <=1%", thin="max(w/h,h/w)>=10", overlap="IoU>=0.8", near="64-bit dHash distance<=4 AND 32x32 grayscale RMSE<=0.08"),
                  baseline_counts=stats, wrinkle=wrstats, negative_composition=dict(composition),
                  conflicting_negatives=sum(r["potential_conflicting_negative"] for r in negrows),
                  n1_excluded_images=n1, n2_excluded_images=len(negatives), policy_estimates=estimates,
                  processed_exact_groups=exact, raw_exact_groups=len(rawduplicates),
                  raw_cross_split_exact_groups=sum(len({p.split('/')[0].split('\\')[0] for p in v})>1 for v in rawduplicates.values()),
                  baseline_duplicate_report=prep["duplicates"], near_candidate_pairs=len(near),
                  cross_split_near_candidates=sum(p["cross_split"] for p in near),
                  preview_counts=dict(Counter(r["kind"] for r in preview_manifest)),
                  protected_file_count=len(before), protected_files_unchanged=True,
                  v2_created=False, training="NOT_RUN", human_review="PENDING",
                  limitations=["dHash is a screening heuristic, not duplicate confirmation; crop/subject leakage may be missed", "No automatic annotation correctness or dominant-class decisions", "Test statistics/previews descriptive only; no test annotation edits proposed"])
    save("summary.json", report)
    print(json.dumps(report, indent=2))


def supplement():
    """Add review aids without re-running or overwriting the audit package."""
    target = OUT / "class_image_counts.json"
    if target.exists():
        raise RuntimeError("Supplement already exists; refusing overwrite")
    counts = {}
    for split in SPLITS:
        data = json.loads((DATA / "annotations" / f"instances_{split}.json").read_text())
        counts[split] = {name: {"images": len({a["image_id"] for a in data["annotations"] if a["category_id"] == idx}),
                               "objects": sum(a["category_id"] == idx for a in data["annotations"])}
                         for idx, name in enumerate(CLASSES)}
    save("class_image_counts.json", counts)
    near = json.loads((OUT / "near_duplicate_candidates.json").read_text())
    selected = sorted([p for p in near if p["cross_split"]], key=lambda p: (p["gray_rmse"], p["a"], p["b"]))[:30]
    folder = OUT / "duplicate_pairs"
    folder.mkdir()
    for idx, pair in enumerate(selected, 1):
        canvas = Image.new("RGB", (1280, 710), "white")
        d = ImageDraw.Draw(canvas)
        for col, key in enumerate(["a", "b"]):
            split, filename = pair[key].split("/", 1)
            with Image.open(DATA / "images" / split / filename) as im:
                canvas.paste(ImageOps.contain(im.convert("RGB"), (640, 640)), (col*640, 0))
            for line, text in enumerate(textwrap.wrap(pair[key], 82)):
                d.text((col*640+5, 645+line*15), text, fill="black")
        d.text((5, 690), f"Pair {idx}: candidate only | distance={pair['hamming']} RMSE={pair['gray_rmse']:.5f} | HUMAN DECISION PENDING", fill="black")
        canvas.save(folder / f"pair_{idx:03d}.jpg", quality=92)
    save("duplicate_preview_manifest.json", selected)
    protected = json.loads((OUT / "protected_files_sha256.json").read_text())
    assert protected == protected_snapshot(), "Protected data changed"
    print("Added class image counts and 30 cross-split duplicate comparison previews; protected files unchanged")


if __name__ == "__main__":
    if sys.argv[1:] == ["--supplement"]:
        supplement()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("Usage: python -B scripts/audit_phase4b.py [--supplement]")
