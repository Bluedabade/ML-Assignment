"""Deterministic membership-only S1 derivation. Default: read-only preflight."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'datasets/processed/skin_detection'
V2 = ROOT / 'datasets/processed/skin_detection_v2'
REVIEW = ROOT / 'results/phase4b_review'
FINAL = REVIEW / 'final_duplicate_review'
SPLITS = ('train', 'val', 'test')
NAMES = ('acne', 'wrinkle', 'dark_spot', 'enlarged_pore')
EXPECTED = {
    'train': dict(images=1687, negatives=610, objects=[7484,1815,54,440], class_images=[885,214,29,224]),
    'val': dict(images=315, negatives=174, objects=[734,22,16,33], class_images=[129,8,10,15]),
    'test': dict(images=248, negatives=102, objects=[1415,89,15,117], class_images=[131,30,9,56]),
}

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, content):
    with path.open('x', encoding='utf-8') as f:
        json.dump(content, f, indent=2)
        f.write('\n')

def stats(coco):
    anns = defaultdict(list)
    for a in coco['annotations']:
        anns[a['image_id']].append(a)
    objects, positive = Counter(), Counter()
    for im in coco['images']:
        ids = [a['category_id'] for a in anns[im['id']]]
        objects.update(ids)
        positive.update(set(ids))
    return dict(images=len(coco['images']), negatives=sum(not anns[im['id']] for im in coco['images']),
        objects=[objects[i] for i in range(4)], class_images=[positive[i] for i in range(4)])

def protected_check(expected):
    paths = {str(p.relative_to(ROOT)) for root in [ROOT/'datasets/raw', BASE, ROOT/'runs'] for p in root.rglob('*') if p.is_file()}
    assert paths == set(expected), 'Protected membership differs from original audit'
    for path, digest in expected.items():
        assert sha(ROOT/path) == digest, f'Protected hash differs: {path}'

def preflight():
    assert not V2.exists(), 'Refusing to overwrite existing V2'
    protected = load(REVIEW/'protected_files_sha256.json')
    protected_check(protected)
    review_hashes = {str(p.relative_to(ROOT)):sha(p) for p in REVIEW.rglob('*') if p.is_file()}
    pairs = csv_rows(FINAL/'duplicate_review.csv')
    assert [r['pair_id'] for r in pairs] == [f'D{i:03d}' for i in range(1,64)]
    assert all(r['human_decision']=='B_VARIANT' and r['likely_same_source']=='YES' for r in pairs)
    groups = csv_rows(FINAL/'confirmed_duplicate_groups.csv')
    group_nodes = {}
    for g in groups:
        group_nodes[g['group_id']] = {tuple(f.split('/',1)) for f in g['files'].split(';')}
    assert len(groups)==43 and sum(len(n) for n in group_nodes.values())==102
    node_group = {n:gid for gid,nodes in group_nodes.items() for n in nodes}
    assert len(node_group)==102
    for p in pairs:
        assert node_group[(p['split_a'],p['file_a'])] == node_group[(p['split_b'],p['file_b'])]
    duplicates = {f:gid for gid,nodes in group_nodes.items() if any(s in ['val','test'] for s,f in nodes) for s,f in nodes if s=='train'}
    n1_rows = csv_rows(REVIEW/'final_n1_review/n1_candidate_images.csv')
    n1 = {r['filename']:r for r in n1_rows}
    assert len(duplicates)==52 and len(n1)==46
    assert not (duplicates.keys() & n1.keys())
    assert all(set(r['original_categories'].replace(';',',').split(',')) <= {'Blackheads','Whiteheads'} for r in n1_rows)
    held = {gid:nodes for gid,nodes in group_nodes.items() if {'val','test'} <= {s for s,f in nodes}}
    assert len(held)==7 and set(held)=={'G003','G005','G011','G015','G021','G032','G038'}
    val_removals = {f:gid for gid,nodes in held.items() for s,f in nodes if s=='val'}
    assert len(val_removals)==7
    excluded = {'train':set(duplicates)|set(n1), 'val':set(val_removals), 'test':set()}
    combined = csv_rows(FINAL/'proposed_v2_train_removals.csv')
    assert {r['filename'] for r in combined} == excluded['train'] and len(combined)==98
    original, derived, before, after = {}, {}, {}, {}
    changes = []
    for split in SPLITS:
        coco = load(BASE/'annotations'/f'instances_{split}.json')
        assert {c['id']:c['name'] for c in coco['categories']} == dict(enumerate(NAMES))
        images = {im['file_name']:im for im in coco['images']}
        assert excluded[split] <= images.keys()
        image_ids = {im['id'] for im in coco['images']}
        assert len(image_ids)==len(images)==len(coco['images'])
        assert all(a['image_id'] in image_ids for a in coco['annotations'])
        anns = defaultdict(list)
        for a in coco['annotations']: anns[a['image_id']].append(a)
        for filename,im in images.items():
            label = BASE/'labels'/split/(Path(filename).stem+'.txt')
            assert (BASE/'images'/split/filename).is_file() and label.is_file()
            lines = [l.split() for l in label.read_text().splitlines() if l.strip()]
            assert len(lines)==len(anns[im['id']])
            for row,a in zip(lines,anns[im['id']]):
                x,y,w,h=a['bbox']
                assert int(row[0])==a['category_id']
                assert np.allclose(list(map(float,row[1:])),[(x+w/2)/im['width'],(y+h/2)/im['height'],w/im['width'],h/im['height']],atol=2e-6)
            if split=='train' and filename in n1:
                assert not lines, 'N1 image must be negative'
            if filename in excluded[split]:
                gid = duplicates.get(filename) if split=='train' else val_removals[filename]
                reason = 'STRICT_N1_BLACKHEAD_WHITEHEAD_NEGATIVE' if split=='train' and filename in n1 else ('CONFIRMED_CROSS_SPLIT_B_VARIANT' if split=='train' else 'VAL_TEST_LEAKAGE_S1_REMOVE_VAL')
                counterparts = sorted(f'{s}/{f}' for s,f in group_nodes.get(gid,set()) if s in ['val','test'] and f not in excluded[s])
                changes.append(dict(filename=filename,split=split,image_path=f'images/{split}/{filename}',label_path=f'labels/{split}/{label.name}',
                    reason=reason,duplicate_group_id=gid,retained_counterparts=counterparts,
                    retained_test_counterparts=[p for p in counterparts if p.startswith('test/')],
                    original_categories=n1[filename]['original_categories'] if filename in n1 else None,
                    removed_annotation_ids=[a['id'] for a in anns[im['id']]],objects=dict(Counter(NAMES[a['category_id']] for a in anns[im['id']]))))
        retained = [im for im in coco['images'] if im['file_name'] not in excluded[split]]
        ids = {im['id'] for im in retained}
        new = dict(coco, images=retained, annotations=[a for a in coco['annotations'] if a['image_id'] in ids])
        original[split],derived[split]=coco,new
        before[split],after[split]=stats(coco),stats(new)
        assert after[split]==EXPECTED[split], f'COUNT DISAGREEMENT: {split}: {after[split]} != {EXPECTED[split]}'
    retained_nodes = {(s,im['file_name']) for s in SPLITS for im in derived[s]['images']}
    assert all(len({s for s,f in nodes & retained_nodes})<=1 for nodes in group_nodes.values()), 'Confirmed transitive leakage remains'
    required_bytes = sum((BASE/'images'/s/im['file_name']).stat().st_size + (BASE/'labels'/s/(Path(im['file_name']).stem+'.txt')).stat().st_size for s in SPLITS for im in derived[s]['images'])
    assert shutil.disk_usage(BASE).free > required_bytes + 100*1024*1024, 'Insufficient free disk space'
    print('PREFLIGHT_COUNTS_MATCH: '+json.dumps(after))
    return protected,review_hashes,pairs,group_nodes,original,derived,before,after,changes

def near_screen(derived):
    records=[]
    for s in SPLITS:
        for im in derived[s]['images']:
            with Image.open(V2/'images'/s/im['file_name']) as opened:
                gray=np.asarray(opened.convert('L').resize((9,8)),dtype=np.int16)
                thumb=np.asarray(opened.convert('L').resize((32,32)),dtype=np.float32)/255
            bits=(gray[:,1:]>gray[:,:-1]).flatten()
            dhash=sum(int(v)<<k for k,v in enumerate(bits))
            records.append(dict(node=f"{s}/{im['file_name']}",split=s,dhash=dhash,thumb=thumb))
    candidates=[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            distance=(a['dhash']^b['dhash']).bit_count()
            if distance<=4:
                rmse=float(np.sqrt(np.mean((a['thumb']-b['thumb'])**2)))
                if rmse<=.08:
                    candidates.append(dict(a=a['node'],b=b['node'],hamming=distance,gray_rmse=rmse,cross_split=a['split']!=b['split'],decision='PENDING_HUMAN_REVIEW'))
    baseline={frozenset((p['a'],p['b'])) for p in load(REVIEW/'near_duplicate_candidates.json')}
    new_cross=[p for p in candidates if p['cross_split'] and frozenset((p['a'],p['b'])) not in baseline]
    return candidates,new_cross

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create',action='store_true',help='Create V2 only after successful read-only preflight')
    args=parser.parse_args()
    protected,review_hashes,pairs,groups,original,derived,before,after,changes=preflight()
    if not args.create:
        print('READ_ONLY_PREFLIGHT_PASS; V2 not created')
        return
    V2.mkdir()
    for s in SPLITS:
        (V2/'images'/s).mkdir(parents=True)
        (V2/'labels'/s).mkdir(parents=True)
        for im in derived[s]['images']:
            for kind,filename in [('images',im['file_name']),('labels',Path(im['file_name']).stem+'.txt')]:
                shutil.copy2(BASE/kind/s/filename,V2/kind/s/filename)
                assert sha(BASE/kind/s/filename)==sha(V2/kind/s/filename)
    (V2/'annotations').mkdir()
    for s in SPLITS:
        dst=V2/'annotations'/f'instances_{s}.json'
        if s=='test':
            shutil.copy2(BASE/'annotations'/dst.name,dst)
        else:
            save(dst,derived[s])
        assert load(dst)==derived[s]
    shutil.copy2(BASE/'data.yaml',V2/'data.yaml')
    assert 'path:' not in (V2/'data.yaml').read_text(), 'Use YAML-directory-relative paths, not machine paths'
    manifest=dict(baseline_dataset_path='datasets/processed/skin_detection',derived_dataset_path='datasets/processed/skin_detection_v2',
        policy='V2_EXPERIMENT_1_STRATEGY_S1_PRESERVE_TEST',classes=list(NAMES),human_decisions='D001-D063 B_VARIANT / YES',
        changes=changes,counts_before=before,counts_after=after,
        removal_summary=dict(train_duplicate=52,train_strict_n1=46,train_overlap=0,train_unique=98,val_s1=7,test=0),
        annotation_policy='Membership changes only. Retained image/label bytes and annotation records unchanged; IDs not renumbered. Wrinkle and annotation conflicts unchanged. No N2 or synthetic data.',
        seed=42,determinism='No randomization or split reassignment; removal manifests and original retained COCO order preserved.',
        source_metadata_hashes=review_hashes,baseline_protected_hash_manifest='results/phase4b_review/protected_files_sha256.json')
    save(V2/'v2_change_manifest.json',manifest)
    validation=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validate_dataset.py'),'--dataset-root',str(V2)],capture_output=True,text=True,cwd=ROOT)
    print(validation.stdout,flush=True)
    assert validation.returncode==0, validation.stdout+validation.stderr
    from validate_dataset import validate
    validator_summary,errors,warnings=validate(V2)
    assert not errors
    for s in SPLITS:
        row=validator_summary[s]
        assert row['images']==after[s]['images'] and row['negative_images']==after[s]['negatives']
        assert row['objects_per_class']==dict(zip(NAMES,after[s]['objects']))
        assert stats(load(V2/'annotations'/f'instances_{s}.json'))==after[s]
    test_hashes={}
    for kind in ['images','labels']:
        a={p.name:sha(p) for p in (BASE/kind/'test').iterdir() if p.is_file()}
        b={p.name:sha(p) for p in (V2/kind/'test').iterdir() if p.is_file()}
        assert a==b, f'TEST {kind} mismatch'
        test_hashes[kind]=b
    assert sha(BASE/'annotations/instances_test.json')==sha(V2/'annotations/instances_test.json')
    candidates,new_cross=near_screen(derived)
    retained={(s,im['file_name']) for s in SPLITS for im in derived[s]['images']}
    remaining=[gid for gid,nodes in groups.items() if len({s for s,f in nodes & retained})>1]
    assert not remaining, f'CONFIRMED_LEAKAGE_FAIL: {remaining}'
    cross=[p for p in candidates if p['cross_split']]
    save(V2/'near_duplicate_candidates.json',dict(method='Same Phase 4B screening: 64-bit dHash<=4; 32x32 grayscale float32 RMSE<=0.08; Pillow default resize; no automatic decisions',
        all_pairs=candidates,cross_split_pairs=cross,new_cross_split_pairs=new_cross,confirmed_reviewed_leakage_groups_remaining=remaining))
    protected_check(protected)
    for rel,digest in review_hashes.items():
        assert sha(ROOT/rel)==digest, f'Review evidence modified: {rel}'
    current_review={str(p.relative_to(ROOT)) for p in REVIEW.rglob('*') if p.is_file()}
    assert current_review==set(review_hashes)
    report=dict(status='PASS',counts=after,validator=validator_summary,errors=errors,warnings=warnings,
        test_identical=True,test_images=248,test_annotation_count=len(original['test']['annotations']),test_hashes=test_hashes,
        test_coco_sha256=sha(V2/'annotations/instances_test.json'),all_retained_images_and_labels_byte_identical=True,
        all_retained_coco_records_unchanged=True,confirmed_reviewed_leakage_groups_remaining=remaining,
        near_duplicate_pairs=len(candidates),cross_split_near_candidates=len(cross),new_cross_split_near_candidates=len(new_cross),
        protected_files_unchanged=True,protected_file_count=len(protected),review_evidence_unchanged=True,training='NOT_RUN')
    save(V2/'v2_validation_report.json',report)
    print('V2_CREATION_VALIDATION_PASS '+json.dumps({k:v for k,v in report.items() if k not in ['test_hashes','validator']}))

if __name__=='__main__':
    main()
