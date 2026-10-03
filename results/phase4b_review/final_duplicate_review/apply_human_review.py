"""Phase 4B.3: authorized human decisions and proposals; never alters datasets."""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / 'datasets/processed/skin_detection'
NAMES = ['acne', 'wrinkle', 'dark_spot', 'enlarged_pore']

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames

def encode_csv(rows, fields):
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    return f.getvalue().encode('utf-8-sig')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def signature(side):
    w, h = side['dimensions']
    return sorted((a['category_id'], *(round(v/d, 6) for v, d in zip(a['bbox'], [w,h,w,h]))) for a in side['annotations'])

def main():
    protected = json.loads((OUT.parent / 'protected_files_sha256.json').read_text())
    for rel, expected in protected.items():
        assert digest(ROOT / rel) == expected, f'Pre-existing protected change: {rel}'
    snapshot = {str(p.relative_to(ROOT)): digest(p) for p in OUT.parent.rglob('*') if p.is_file()}
    rows, fields = read_csv(OUT / 'duplicate_review.csv')
    original = [dict(r) for r in rows]
    assert [r['pair_id'] for r in rows] == [f'D{i:03d}' for i in range(1,64)]
    metadata = {m['pair_id']: m for m in json.loads((OUT / 'pair_metadata.json').read_text())}
    differences, _ = read_csv(OUT / 'd031_d063/annotation_differences.csv')
    authoritative = {r['pair_id']: r['difference'] for r in differences}
    annotation_rows = []
    flags = {}
    for r in rows:
        m = metadata[r['pair_id']]
        for key, suffix in [('a','a'), ('b','b')]:
            assert m[key]['filename'] == r['file_'+suffix]
            assert m[key]['split'] == r['split_'+suffix]
        counts = [Counter(a['category_id'] for a in m[k]['annotations']) for k in ['a','b']]
        computed = 'CLASS_OR_COUNT' if counts[0] != counts[1] else ('BOX_GEOMETRY' if signature(m['a']) != signature(m['b']) else 'MATCH')
        if int(r['pair_id'][1:]) >= 31:
            assert computed == authoritative.get(r['pair_id'], 'MATCH'), 'Structured comparison disagreement'
            basis = 'd031_d063/annotation_differences.csv + pair_metadata.json'
        else:
            basis = 'pair_metadata.json; existing normalized-bbox comparison (6 decimals)'
        flag = {'CLASS_OR_COUNT':'ANNOTATION_CONFLICT', 'BOX_GEOMETRY':'ANNOTATION_DIFFERENCE', 'MATCH':'MATCH'}[computed]
        flags[r['pair_id']] = flag
        r['human_decision'] = 'B_VARIANT'
        r['likely_same_source'] = 'YES'
        r['annotation_flag'] = flag
        note = f'Human-confirmed B_VARIANT; {flag}; structured comparison only; no dataset changes.'
        r['notes'] = (r['notes'] + ' | ' if r['notes'] else '') + note
        annotation_rows.append(dict(pair_id=r['pair_id'], annotation_flag=flag,
            classes_counts_a=json.dumps({NAMES[k]:v for k,v in counts[0].items()},sort_keys=True),
            classes_counts_b=json.dumps({NAMES[k]:v for k,v in counts[1].items()},sort_keys=True),
            absent_one_side=bool(counts[0]) != bool(counts[1]), comparison_source=basis))
    for pid in ['D033','D035','D044','D052','D060']:
        assert flags[pid] == 'ANNOTATION_CONFLICT'
    parent = {}
    def find(n):
        parent.setdefault(n,n)
        if parent[n] != n:
            parent[n] = find(parent[n])
        return parent[n]
    for r in rows:
        a, b = (r['split_a'],r['file_a']), (r['split_b'],r['file_b'])
        parent[find(b)] = find(a)
    components = defaultdict(set)
    for node in list(parent):
        components[find(node)].add(node)
    components = sorted(components.values(), key=lambda nodes: min(r['pair_id'] for r in rows if (r['split_a'],r['file_a']) in nodes))
    groups, cleanup, removals = [], [], {}
    for idx, nodes in enumerate(components,1):
        gid = f'G{idx:03d}'
        edges = [r['pair_id'] for r in rows if (r['split_a'],r['file_a']) in nodes]
        bysplit = {s: sorted(f for ss,f in nodes if ss==s) for s in ['train','val','test']}
        hold = bool(bysplit['val'] and bysplit['test'])
        status = 'HOLD_VAL_TEST_REVIEW' if hold else 'PROPOSE_TRAIN_EXCLUSION_ONLY'
        groups.append(dict(group_id=gid,pair_ids=';'.join(edges),num_files=len(nodes),files=';'.join(f'{s}/{f}' for s,f in sorted(nodes)),
            splits=';'.join(s for s in bysplit if bysplit[s]),train_files=';'.join(bysplit['train']),val_files=';'.join(bysplit['val']),test_files=';'.join(bysplit['test']),
            has_train=bool(bysplit['train']),has_val=bool(bysplit['val']),has_test=bool(bysplit['test']),
            contains_annotation_difference=any(flags[p]=='ANNOTATION_DIFFERENCE' for p in edges),
            contains_annotation_conflict=any(flags[p]=='ANNOTATION_CONFLICT' for p in edges),proposed_cleanup_status=status))
        for s,f in sorted(nodes):
            if s == 'train' and (bysplit['val'] or bysplit['test']):
                assert f not in removals
                removals[f] = gid
                action = 'PROPOSE_EXCLUDE_FROM_TRAIN'
            else:
                action = 'RETAIN_UNCHANGED_PENDING_VAL_TEST_REVIEW' if hold else 'RETAIN_HELD_OUT_UNCHANGED'
            cleanup.append(dict(group_id=gid,filename=f,split=s,proposed_action=action,
                retained_counterparts=';'.join(f'{ss}/{ff}' for ss,ff in sorted(nodes) if ss in ['val','test'] and (ss,ff)!=(s,f)),
                proposed_cleanup_status=status,notes='Proposal only; no file removed or split changed.'))
    n1, _ = read_csv(OUT.parent / 'final_n1_review/n1_candidate_images.csv')
    n1_names = {r['filename'] for r in n1}
    assert len(n1_names) == len(n1) == 46
    assert all(set(r['original_categories'].replace(';',',').split(',')) <= {'Blackheads','Whiteheads'} for r in n1)
    removal_names = set(removals) | n1_names
    removal_rows = [dict(filename=f,reason_duplicate=f in removals,duplicate_group_id=removals.get(f,''),reason_n1=f in n1_names,
        proposed_action='EXCLUDE_FROM_TRAIN_IN_V2_EXPERIMENT_1') for f in sorted(removal_names)]
    estimates = {}
    for split in ['train','val','test']:
        coco = json.loads((DATA / 'annotations' / f'instances_{split}.json').read_text())
        anns = defaultdict(list)
        for a in coco['annotations']:
            anns[a['image_id']].append(a)
        images = {im['file_name']:im for im in coco['images']}
        if split == 'train':
            assert removal_names <= images.keys()
            assert all(not anns[images[f]['id']] for f in n1_names)
        baseline, after, removed = Counter(), Counter(), Counter()
        for f,im in images.items():
            objects = anns[im['id']]
            label = DATA / 'labels' / split / (Path(f).stem+'.txt')
            lines = [l for l in label.read_text().splitlines() if l.strip()]
            assert Counter(int(l.split()[0]) for l in lines) == Counter(a['category_id'] for a in objects), f'Label/COCO mismatch {split}/{f}'
            counts = Counter({'images':1,'negative_images':int(not objects)})
            counts.update(NAMES[a['category_id']] for a in objects)
            baseline.update(counts)
            (removed if split=='train' and f in removal_names else after).update(counts)
        estimates[split] = {k:{metric:v[metric] for metric in ['images','negative_images',*NAMES]} for k,v in [('baseline',baseline),('estimated_v2',after),('proposed_removed',removed)]}
    wrinkle = OUT.parent / 'final_wrinkle_review/wrinkle_review.csv'
    wr, _ = read_csv(wrinkle)
    assert all(r['human_decision']=='PENDING' for r in wr)
    summary = dict(phase='4B.3',human_decision='B_VARIANT',likely_same_source='YES',pairs=len(rows),
        connected_groups=len(groups),unique_files=len(parent),group_split_composition=dict(Counter(g['splits'] for g in groups)),
        annotation_pair_flags=dict(Counter(flags.values())),duplicate_train_removals=len(removals),
        strict_n1_total=len(n1_names),n1_overlap=len(n1_names & set(removals)),n1_only=len(n1_names-set(removals)),duplicate_only=len(set(removals)-n1_names),
        combined_unique_train_removals=len(removal_names),estimated_counts=estimates,
        val_test_groups=[g for g in groups if g['has_val'] and g['has_test']],wrinkle_review_pending=len(wr),
        evaluation_warning='Confirmed val/test same-source variants make validation and test non-independent. Historical baseline evaluation may be optimistic. Bias magnitude unknown; metrics not recomputed.',
        historical_reports='Earlier review summaries/previews describe pre-decision state; duplicate_review.csv and this phase4b3_summary.json are current.',
        protected_file_count=len(protected),v2_created=False,training='NOT_RUN')
    outputs = {
        'duplicate_review.csv': encode_csv(rows,fields+['annotation_flag']),
        'confirmed_duplicate_groups.csv':encode_csv(groups,list(groups[0])),
        'proposed_duplicate_cleanup.csv':encode_csv(cleanup,list(cleanup[0])),
        'proposed_v2_train_removals.csv':encode_csv(removal_rows,list(removal_rows[0])),
        'confirmed_pair_annotation_flags.csv':encode_csv(annotation_rows,list(annotation_rows[0])),
    }
    assert not (ROOT / 'datasets/processed/skin_detection_v2').exists()
    for name in outputs:
        if name != 'duplicate_review.csv':
            assert not (OUT / name).exists(), f'Output already exists: {name}'
    for name, content in outputs.items():
        (OUT / name).write_bytes(content)
    updated, _ = read_csv(OUT/'duplicate_review.csv')
    for before, after in zip(original,updated):
        assert all(before[k]==after[k] for k in fields if k not in ['notes','human_decision','likely_same_source'])
        assert after['human_decision']=='B_VARIANT' and after['likely_same_source']=='YES'
    for name in outputs:
        rr, _ = read_csv(OUT/name)
        assert rr
    for rel, expected in protected.items():
        assert digest(ROOT / rel) == expected, f'Protected change: {rel}'
    for rel, expected in snapshot.items():
        if rel != str((OUT/'duplicate_review.csv').relative_to(ROOT)):
            assert digest(ROOT/rel) == expected, f'Unexpected review change: {rel}'
    summary['protected_files_unchanged'] = True
    summary['existing_review_metadata_unchanged_except_authorized_csv'] = True
    (OUT/'phase4b3_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
