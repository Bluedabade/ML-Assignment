"""Prepare human-only CSV/image review aids from the existing Phase 4B audit.

Never edits source images, annotations, splits or any existing review output.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import json
import math
import textwrap
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phase4b_review"
DATA = ROOT / "datasets/processed/skin_detection"
RAW = ROOT / "datasets/raw/roboflow_skin_problem_clean3"
DUP = BASE / "final_duplicate_review"
WR = BASE / "final_wrinkle_review"
N1 = BASE / "final_n1_review"
COLORS = ["#ef4444", "#2563eb", "#9333ea", "#16a34a"]
NAMES = ["acne", "wrinkle", "dark_spot", "enlarged_pore"]
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
SMALL = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_write(path, fields, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8-sig", newline="") as f:
        check = list(csv.DictReader(f))
    assert len(check) == len(rows)
    if "human_decision" in fields:
        assert all(row["human_decision"] == "PENDING" for row in check)


def lines(texts, width=83):
    output = []
    for text in texts:
        remaining = str(text)
        while remaining:
            n = min(len(remaining), width)
            while n > 1 and FONT.getlength(remaining[:n]) > 682:
                n -= 1
            if n < len(remaining) and " " in remaining[:n]:
                n = remaining.rfind(" ", 0, n)+1
            output.append(remaining[:n].rstrip())
            remaining = remaining[n:].lstrip()
    return output


def panel(rec, captions, wrinkle_only=False, highlights=None):
    with Image.open(rec["path"]) as src:
        image = src.convert("RGB")
    draw = ImageDraw.Draw(image)
    for ann in rec["annotations"]:
        cid = ann["category_id"]
        if wrinkle_only and cid != 1:
            continue
        x, y, w, h = ann["bbox"]
        aid = ann.get("source_annotation_id", ann["id"])
        color = COLORS[cid]
        draw.rectangle((x, y, x+w, y+h), outline=color, width=2)
        draw.text((max(0, x), max(0, y-15)), f"{NAMES[cid]} #{aid}", fill=color,
                  font=SMALL, stroke_width=1, stroke_fill="white")
    image = ImageOps.contain(image, (700, 700))
    textlines = lines(captions)
    canvas = Image.new("RGB", (700, 700+22*len(textlines)+12), "white")
    canvas.paste(image, ((700-image.width)//2, 0))
    draw = ImageDraw.Draw(canvas)
    for idx, line in enumerate(textlines):
        draw.text((7, 706+idx*22), line, fill="black", font=FONT)
    return canvas


def contact(paths, output, columns, tile_w, tile_h):
    canvas = Image.new("RGB", (columns*tile_w, math.ceil(len(paths)/columns)*tile_h), "#eeeeee")
    for idx, path in enumerate(paths):
        with Image.open(path) as src:
            tile = ImageOps.contain(src, (tile_w-6, tile_h-6))
        canvas.paste(tile, ((idx%columns)*tile_w+3, (idx//columns)*tile_h+3))
    canvas.save(output, quality=93)


def main():
    for folder in [DUP, WR, N1]:
        if folder.exists():
            raise RuntimeError(f"Refusing to overwrite: {folder}")
    prior = {str(p.relative_to(ROOT)): sha(p) for p in BASE.rglob("*") if p.is_file()}
    protected = read(BASE / "protected_files_sha256.json")
    assert all(sha(ROOT/p) == h for p, h in protected.items()), "Baseline changed since audit"
    records = {}
    for split, source_split in [("train", "train"), ("val", "valid"), ("test", "test")]:
        raw = read(RAW / source_split / "_annotations.coco.json")
        original = {im["id"]: im for im in raw["images"]}
        coco = read(DATA / "annotations" / f"instances_{split}.json")
        anns = defaultdict(list)
        for ann in coco["annotations"]:
            anns[ann["image_id"]].append(ann)
        for im in coco["images"]:
            p = DATA / "images" / split / im["file_name"]
            with Image.open(p) as opened:
                dimensions = list(opened.size)
            records[(split, im["file_name"])] = dict(path=p, split=split, filename=im["file_name"],
                source_split=source_split, source_id=im["source_image_id"],
                source_filename=original[im["source_image_id"]]["file_name"],
                dimensions=dimensions, annotations=anns[im["id"]])
    for folder in [DUP, WR, N1]:
        folder.mkdir()
    pairs = sorted([p for p in read(BASE / "near_duplicate_candidates.json") if p["cross_split"]],
                   key=lambda p: (p["gray_rmse"], p["hamming"], p["a"], p["b"]))
    assert len(pairs) == 63
    duprows, metadata, pairpaths = [], [], []
    for idx, pair in enumerate(pairs, 1):
        pid = f"D{idx:03d}"
        panels, meta = [], {}
        for key in ["a", "b"]:
            split, filename = pair[key].split("/", 1)
            rec = records[(split, filename)]
            counts = Counter(NAMES[a["category_id"]] for a in rec["annotations"])
            captions = [f"{pid} image {key.upper()} | split={split} | HUMAN DECISION PENDING",
                        filename, f"dHash distance={pair['hamming']} | RMSE={pair['gray_rmse']:.8f}",
                        f"Image dimensions={rec['dimensions'][0]}x{rec['dimensions'][1]}",
                        f"Original source split={rec['source_split']} | COCO image ID={rec['source_id']}",
                        "Original filename: " + rec["source_filename"],
                        "Target objects: " + (str(dict(counts)) if counts else "NONE (empty derived label)")]
            panels.append(panel(rec, captions))
            meta[key] = {k: rec[k] for k in ["split", "filename", "source_split", "source_id", "source_filename", "dimensions", "annotations"]}
        canvas = Image.new("RGB", (1400, max(p.height for p in panels)), "white")
        for col, p in enumerate(panels):
            canvas.paste(p, (col*700, 0))
        dest = DUP / f"{pid}.jpg"
        canvas.save(dest, quality=94)
        pairpaths.append(dest)
        sa, fa = pair["a"].split("/", 1)
        sb, fb = pair["b"].split("/", 1)
        duprows.append(dict(pair_id=pid, file_a=fa, split_a=sa, file_b=fb, split_b=sb,
                            dhash_distance=pair["hamming"], rmse=pair["gray_rmse"],
                            likely_same_source="UNKNOWN", human_decision="PENDING", notes=""))
        metadata.append(dict(pair_id=pid, preview=dest.name, **meta))
    csv_write(DUP / "duplicate_review.csv", list(duprows[0]), duprows)
    (DUP / "pair_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    contact(pairpaths[:30], DUP / "top30_contact_sheet.jpg", 3, 1000, 760)

    boxes = read(BASE / "wrinkle_boxes.json")
    for box in boxes:
        rec = records[(box["split"], box["filename"])]
        ann = next(a for a in rec["annotations"] if a.get("source_annotation_id") == box["source_annotation_id"] and a["category_id"] == 1)
        x, y, w, h = ann["bbox"]
        iw, ih = rec["dimensions"]
        box["boundary_gap_norm"] = min(x/iw, y/ih, (iw-x-w)/iw, (ih-y-h)/ih)
    tie = lambda b: (b["split"], b["filename"], b["source_annotation_id"])
    groups = {
        "SMALLEST_20": sorted(boxes, key=lambda b: (b["area_norm"], min(b["width_px"], b["height_px"]), tie(b)))[:20],
        "THINNEST_20": sorted(boxes, key=lambda b: (-max(b["aspect_ratio"], 1/b["aspect_ratio"]), tie(b)))[:20],
        "WIDEST_10": sorted(boxes, key=lambda b: (-b["width_norm"], tie(b)))[:10],
        "NEAREST_BOUNDARY_10": sorted(boxes, key=lambda b: (b["boundary_gap_norm"], tie(b)))[:10]}
    chosen, imagegroups = {}, defaultdict(list)
    for reason, group in groups.items():
        for b in group:
            key = (b["split"], b["filename"], b["source_annotation_id"])
            if key not in chosen:
                chosen[key] = dict(b, reasons=[])
            chosen[key]["reasons"].append(reason)
    wrrows = []
    for key, b in sorted(chosen.items()):
        row = dict(image=b["filename"], split=b["split"], annotation_id=b["source_annotation_id"],
                   width_px=b["width_px"], height_px=b["height_px"], normalized_area=b["area_norm"],
                   aspect_ratio=b["aspect_ratio"], reason_flagged=";".join(b["reasons"]), human_decision="PENDING", notes="")
        wrrows.append(row)
        imagegroups[key[:2]].append(b)
    csv_write(WR / "wrinkle_review.csv", list(wrrows[0]), wrrows)
    image_manifest = []
    for idx, (key, selected) in enumerate(sorted(imagegroups.items()), 1):
        rec = records[key]
        captions = [f"W{idx:03d} | split={key[0]} | HUMAN DECISION PENDING", key[1],
                    f"Original source ID={rec['source_id']} | image={rec['dimensions'][0]}x{rec['dimensions'][1]}",
                    "All wrinkle boxes shown. Selected outlier measurements (w/h ratio; area fraction):"]
        captions.extend(f"#{b['source_annotation_id']} w={b['width_px']:.3f}px h={b['height_px']:.3f}px area={b['area_norm']:.8f} ratio={b['aspect_ratio']:.3f} flags={','.join(b['reasons'])}" for b in selected)
        dest = WR / f"W{idx:03d}.jpg"
        panel(rec, captions, wrinkle_only=True).save(dest, quality=94)
        image_manifest.append(dict(image=key[1], split=key[0], preview=dest.name,
                                   selected_annotation_ids=[b["source_annotation_id"] for b in selected]))
    (WR / "image_manifest.json").write_text(json.dumps(image_manifest, indent=2)+"\n")

    negatives = read(BASE / "negative_training_images.json")
    n1 = sorted([n for n in negatives if n["original_categories"] and set(n["original_categories"]) <= {"Blackheads", "Whiteheads"}], key=lambda n: n["filename"])
    assert len(n1) == 46
    nrows, npaths = [], []
    for idx, n in enumerate(n1, 1):
        nrows.append(dict(filename=n["filename"], original_categories=";".join(n["original_categories"]),
                          current_label_status=n["derived_label_status"], proposed_action="EXCLUDE_FROM_TRAIN_IN_V2_N1"))
        rec = records[("train", n["filename"])]
        assert not rec["annotations"]
        dest = N1 / f"N{idx:03d}.jpg"
        panel(rec, [f"N{idx:03d} | TRAIN | PROPOSAL ONLY, NOT EXCLUDED", n["filename"],
                    "Original categories: " + ", ".join(n["original_categories"]), "Current label: EMPTY_AFTER_FILTERING"]).save(dest, quality=92)
        npaths.append(dest)
    csv_write(N1 / "n1_candidate_images.csv", list(nrows[0]), nrows)
    contact(npaths, N1 / "all46_contact_sheet.jpg", 6, 360, 460)
    mixed = [n for n in negatives if set(n["original_categories"]) & {"Blackheads", "Whiteheads"} and n not in n1]
    assert len(mixed) == 2
    (N1 / "mixed_cases_NOT_N1.json").write_text(json.dumps(mixed, indent=2)+"\n")

    assert all(sha(ROOT/p) == h for p, h in protected.items()), "Protected files changed"
    assert all(sha(ROOT/p) == h for p, h in prior.items()), "Existing audit results changed"
    summary = dict(cross_split_pairs=63, rank="ascending grayscale RMSE, then dHash distance, then filenames; heuristic ranking only",
                   top10=duprows[:10], wrinkle_group_slots={k:len(v) for k,v in groups.items()},
                   unique_wrinkle_annotations=len(chosen), unique_wrinkle_images=len(imagegroups),
                   wrinkle_splits=dict(Counter(r["split"] for r in wrrows)), strict_n1=46,
                   excluded_mixed_cases=2, human_decisions="ALL PENDING", likely_same_source="ALL UNKNOWN",
                   automated_duplicate_or_annotation_decisions=0, protected_files_unchanged=True,
                   existing_audit_files_unchanged=True, v2_created=False, training="NOT_RUN")
    (DUP / "review_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


def refresh_wrinkle_previews():
    """Fix caption wrapping only in this phase's newly created wrinkle previews."""
    manifest = read(WR / "image_manifest.json")
    with (WR / "wrinkle_review.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    tables = {s: read(DATA / "annotations" / f"instances_{s}.json") for s in ["train", "val", "test"]}
    displayed = []
    for m in manifest:
        coco = tables[m["split"]]
        im = next(i for i in coco["images"] if i["file_name"] == m["image"])
        anns = [a for a in coco["annotations"] if a["image_id"] == im["id"]]
        path = DATA / "images" / m["split"] / m["image"]
        with Image.open(path) as opened:
            dimensions = list(opened.size)
        rec = dict(path=path, annotations=anns)
        captions = [f"{Path(m['preview']).stem} | split={m['split']} | HUMAN DECISION PENDING", m["image"],
                    f"Original source ID={im['source_image_id']} | image={dimensions[0]}x{dimensions[1]}",
                    "All wrinkle boxes shown; selected outlier measurements below."]
        for r in rows:
            if r["image"] == m["image"] and r["split"] == m["split"]:
                captions.extend([f"#{r['annotation_id']} w={float(r['width_px']):.3f}px h={float(r['height_px']):.3f}px area={float(r['normalized_area']):.8f} ratio(w/h)={float(r['aspect_ratio']):.3f}", "Reason: " + r["reason_flagged"]])
        panel(rec, captions, wrinkle_only=True).save(WR / m["preview"], quality=94)
        for a in anns:
            if a["category_id"] == 1:
                x, y, w, h = a["bbox"]
                displayed.append(dict(image=m["image"], split=m["split"], annotation_id=a.get("source_annotation_id"),
                                      width_px=w, height_px=h, normalized_area=w*h/(dimensions[0]*dimensions[1]), aspect_ratio=w/h,
                                      selected_outlier=a.get("source_annotation_id") in m["selected_annotation_ids"]))
    (WR / "all_displayed_wrinkle_box_metadata.json").write_text(json.dumps(displayed, indent=2)+"\n")
    protected = read(BASE / "protected_files_sha256.json")
    assert all(sha(ROOT/p) == h for p, h in protected.items())
    print("Refreshed 33 NEW wrinkle preview captions; added measurements for all displayed wrinkle boxes. Protected data unchanged.")


if __name__ == "__main__":
    if sys.argv[1:] == ["--refresh-wrinkle-previews"]:
        refresh_wrinkle_previews()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("Usage: python -B scripts/prepare_phase4b1_review.py [--refresh-wrinkle-previews]")
