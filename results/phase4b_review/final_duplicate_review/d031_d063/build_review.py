"""Generate ONLY D031-D063 human review outputs; never write source manifests."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = ROOT / "results/phase4b_review"
SOURCE = OUT.parent
DATA = ROOT / "datasets/processed/skin_detection"
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
TITLE = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)
BOXFONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
NAMES = ["acne", "wrinkle", "dark_spot", "enlarged_pore"]
COLORS = ["#dc2626", "#2563eb", "#9333ea", "#15803d"]


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def wrap(text, width):
    result = []
    text = str(text)
    while text:
        n = len(text)
        while n>1 and FONT.getlength(text[:n]) > width:
            n -= 1
        if n<len(text) and " " in text[:n]:
            n = text.rfind(" ", 0, n)+1
        result.append(text[:n].rstrip())
        text = text[n:].lstrip()
    return result


def panel(meta, label):
    path = DATA / "images" / meta["split"] / meta["filename"]
    with Image.open(path) as src:
        im = src.convert("RGB")
    assert list(im.size) == meta["dimensions"]
    draw = ImageDraw.Draw(im)
    for ann in meta["annotations"]:
        x, y, w, h = ann["bbox"]
        cid = ann["category_id"]
        draw.rectangle([x, y, x+w, y+h], outline=COLORS[cid], width=3)
        text = NAMES[cid]
        tx = min(max(0, x), max(0, im.width-BOXFONT.getlength(text)-2))
        draw.text((tx, max(0, y-15)), text, font=BOXFONT, fill=COLORS[cid], stroke_width=1, stroke_fill="white")
    counts = dict(Counter(NAMES[a["category_id"]] for a in meta["annotations"]))
    ids = [a.get("source_annotation_id", a["id"]) for a in meta["annotations"]]
    captions = [f"{label} | split={meta['split']} | dimensions={im.width}x{im.height}",
                "Filename: " + meta["filename"],
                f"Target classes/counts: {counts or 'NONE'} | total={len(ids)}",
                f"Source COCO image ID={meta['source_id']} (source split={meta['source_split']})",
                "Source filename: " + meta["source_filename"],
                f"Source annotation IDs: {ids or 'NONE'}"]
    width = max(640, im.width)
    lines = [line for text in captions for line in wrap(text, width-20)]
    canvas = Image.new("RGB", (width, im.height+26*len(lines)+20), "white")
    canvas.paste(im, ((width-im.width)//2, 0))
    draw = ImageDraw.Draw(canvas)
    for idx, line in enumerate(lines):
        assert FONT.getlength(line) <= width-20
        draw.text((10, im.height+10+idx*26), line, fill="black", font=FONT)
    return canvas


def signature(meta):
    width, height = meta["dimensions"]
    return sorted((a["category_id"], *(round(v/d, 6) for v, d in zip(a["bbox"], [width,height,width,height]))) for a in meta["annotations"])


def csv_write(name, rows, fields):
    with (OUT / name).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / name).open(encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.DictReader(f))) == len(rows)


def main():
    if any(OUT.glob("D*.jpg")):
        raise RuntimeError("Refusing to overwrite previews")
    prior = {str(p.relative_to(ROOT)):sha(p) for p in BASE.rglob("*") if p.is_file() and not p.is_relative_to(OUT)}
    protected = json.loads((BASE / "protected_files_sha256.json").read_text())
    with (SOURCE / "duplicate_review.csv").open(encoding="utf-8-sig", newline="") as f:
        allrows = list(csv.DictReader(f))
    rows = [r for r in allrows if 31<=int(r["pair_id"][1:])<=63]
    rows.sort(key=lambda r:r["pair_id"])
    assert [r["pair_id"] for r in rows] == [f"D{i:03d}" for i in range(31,64)]
    assert all(r["human_decision"] == "PENDING" and r["likely_same_source"] == "UNKNOWN" for r in rows)
    meta = {p["pair_id"]:p for p in json.loads((SOURCE / "pair_metadata.json").read_text())}
    differences, edges, occurrence = [], defaultdict(set), defaultdict(list)
    for r in rows:
        pid = r["pair_id"]
        m = meta[pid]
        for side in ["a","b"]:
            assert m[side]["split"] == r[f"split_{side}"] and m[side]["filename"] == r[f"file_{side}"]
        changed = signature(m["a"]) != signature(m["b"])
        counts = [Counter(NAMES[a["category_id"]] for a in m[k]["annotations"]) for k in ["a","b"]]
        if changed:
            differences.append(dict(pair_id=pid, classes_counts_a=dict(counts[0]), classes_counts_b=dict(counts[1]),
                                    difference="CLASS_OR_COUNT" if counts[0]!=counts[1] else "BOX_GEOMETRY", human_decision="PENDING"))
        panels = [panel(m[k], f"Image {k.upper()}") for k in ["a","b"]]
        header = f"{pid} | dHash distance={r['dhash_distance']} | RMSE={float(r['rmse']):.8f} | PENDING / UNKNOWN"
        status = "ANNOTATION DIFFERENCE - human interpretation required" if changed else "Target annotation geometry/classes match (not a duplicate decision)"
        w = sum(p.width for p in panels)+30
        canvas = Image.new("RGB", (w, max(p.height for p in panels)+90), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10,10), header, font=TITLE, fill="black")
        draw.text((10,45), status, font=TITLE, fill="#b45309" if changed else "black")
        x=10
        for p in panels:
            canvas.paste(p,(x,85))
            x += p.width+10
        canvas.save(OUT / f"{pid}.jpg", quality=97, subsampling=0)
        a, b = f"{r['split_a']}/{r['file_a']}", f"{r['split_b']}/{r['file_b']}"
        edges[a].add(b)
        edges[b].add(a)
        occurrence[a].append(pid)
        occurrence[b].append(pid)
    sheets=[]
    for start in range(0,33,6):
        subset=rows[start:start+6]
        images=[]
        for r in subset:
            with Image.open(OUT / (r["pair_id"]+".jpg")) as im:
                images.append(im.copy())
        sheet=Image.new("RGB",(max(im.width for im in images),sum(im.height+16 for im in images)),"#dddddd")
        y=0
        for im in images:
            sheet.paste(im,(0,y))
            y+=im.height+16
        name=f"{subset[0]['pair_id']}_{subset[-1]['pair_id']}.jpg"
        sheet.save(OUT/name,quality=97,subsampling=0)
        sheets.append(name)
    visited=set()
    groups=[]
    for seed in sorted(edges):
        if seed in visited:
            continue
        stack=[seed]
        members=set()
        while stack:
            f=stack.pop()
            if f in members:
                continue
            members.add(f)
            stack.extend(edges[f]-members)
        visited.update(members)
        pids=sorted({pid for f in members for pid in occurrence[f]})
        groups.append(dict(group_id=f"G{len(groups)+1:03d}",pair_ids=";".join(pids),files=";".join(sorted(members)),
                           splits=";".join(sorted({f.split('/')[0] for f in members})),num_files=len(members),num_pairs=len(pids)))
    csv_write("candidate_duplicate_groups.csv",groups,["group_id","pair_ids","files","splits","num_files","num_pairs"])
    repeated=[dict(file=f,pair_ids=";".join(sorted(p)),num_pairs=len(p)) for f,p in sorted(occurrence.items()) if len(p)>1]
    csv_write("repeated_file_candidates.csv",repeated,["file","pair_ids","num_pairs"])
    csv_write("annotation_differences.csv",[dict(pair_id=d['pair_id'],difference=d['difference'],classes_counts_a=json.dumps(d['classes_counts_a']),classes_counts_b=json.dumps(d['classes_counts_b']),human_decision="PENDING") for d in differences],
              ["pair_id","difference","classes_counts_a","classes_counts_b","human_decision"])
    assert len([p for p in OUT.glob("D*.jpg") if len(p.stem)==4])==33
    for i in range(31,64):
        with Image.open(OUT/f"D{i:03d}.jpg") as im:
            im.verify()
    for p,h in prior.items():
        assert sha(ROOT/p)==h, f"Existing review file changed: {p}"
    for p,h in protected.items():
        assert sha(ROOT/p)==h, f"Protected file changed: {p}"
    expected=set(protected)
    actual={str(p.relative_to(ROOT)) for folder in [ROOT/'datasets/raw',DATA,ROOT/'runs'] for p in folder.rglob('*') if p.is_file()}
    assert actual==expected, "Protected file membership changed"
    report=dict(pairs=33,contact_sheets=sheets,annotation_difference_pairs=differences,
                candidate_components=len(groups),multi_pair_groups=[g for g in groups if g['num_pairs']>1],repeated_files=repeated,
                d001_d030_status=dict(Counter(r['human_decision'] for r in allrows if int(r['pair_id'][1:])<=30)),
                d031_d063_status="PENDING",likely_same_source="UNKNOWN",source_manifest_unchanged=True,
                existing_review_outputs_unchanged=True,protected_files_unchanged=True,protected_file_count=len(protected),
                v2_created=False,training="NOT_RUN",group_scope="D031-D063 only; candidate graph, not confirmed duplicates",
                annotation_comparison="class + normalized bbox rounded to 6 decimals; IDs ignored; small geometry changes flagged, not judged errors")
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
