"""Additional spatial-first matrices, using saved Phase 5B predictions only."""
import os
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
os.environ['MPLCONFIGDIR']=str(OUT/'.matplotlib_cache')
import json,csv,hashlib
from collections import defaultdict,Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
NAMES=['acne','wrinkle','dark_spot','enlarged_pore','background']
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def spatial_pairs(gt_boxes,pred_boxes):
    """IoU descending greedy; ties resolved by GT index then prediction index."""
    g=np.array(gt_boxes,float).reshape(-1,4); p=np.array(pred_boxes,float).reshape(-1,4)
    if not len(g) or not len(p): return []
    inter=np.maximum(np.minimum(g[:,None,2:],p[None,:,2:])-np.maximum(g[:,None,:2],p[None,:,:2]),0).prod(axis=2)
    union=np.maximum(g[:,2:]-g[:,:2],0).prod(axis=1)[:,None]+np.maximum(p[:,2:]-p[:,:2],0).prod(axis=1)[None,:]-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    gi,pi=np.where(iou>=.5)
    candidates=sorted(zip(gi.tolist(),pi.tolist()),key=lambda pair:(-iou[pair[0],pair[1]],pair[0],pair[1]))
    used_g=set(); used_p=set(); matches=[]
    for a,b in candidates:
        if a not in used_g and b not in used_p:
            matches.append((a,b)); used_g.add(a); used_p.add(b)
    assert len(used_g)==len(matches)==len(used_p)
    return matches
def main():
    # Guard against reruns overwriting any previous matrix.
    for slug in ['yolo11n','skn1']:
        for suffix in ['.png','_normalized.png','.csv']:
            assert not (OUT/(slug+'_confusion_matrix_class_confusion'+suffix)).exists()
    baseline_files=[p for p in OUT.parent.rglob('*') if p.is_file() and p!=Path(__file__).resolve() and '.matplotlib_cache' not in p.parts]
    benchmark=ROOT/'results/external_model_benchmark'
    gtpath=ROOT/'datasets/processed/skin_detection_v2/annotations/instances_test.json'
    protected=baseline_files+[p for p in benchmark.rglob('*') if p.is_file()]+[gtpath]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    gt=load(gtpath); assert len(gt['images'])==248 and len(gt['annotations'])==1636
    cats={c['id']:c['name'] for c in gt['categories']}; canonical={n:i for i,n in enumerate(NAMES[:4])}
    support=Counter(canonical[cats[a['category_id']]] for a in gt['annotations'])
    assert [support[i] for i in range(4)]==[1415,89,15,117]
    truth=defaultdict(list)
    for a in gt['annotations']:
        x,y,w,h=a['bbox']; truth[a['image_id']].append((canonical[cats[a['category_id']]],[x,y,x+w,y+h]))
    matrices={}; statistics={}
    for slug in ['yolo11n','skn1']:
        export=load(benchmark/slug/'canonical_predictions.json')
        assert len(export['processed_images'])==248 and set(export['processed_images'])=={im['file_name'] for im in gt['images']}
        preds=defaultdict(list)
        for p in export['predictions']:
            if p['confidence']>=.25:
                assert p['class_name'] in canonical
                preds[p['image_id']].append((canonical[p['class_name']],[p['x1'],p['y1'],p['x2'],p['y2']]))
        m=np.zeros((5,5),int); prediction_total=sum(map(len,preds.values()))
        for im in gt['images']:
            aa=truth[im['id']]; pp=preds[im['id']]
            matches=spatial_pairs([a[1] for a in aa],[p[1] for p in pp])
            used_g={g for g,p in matches}; used_p={p for g,p in matches}
            for g,p in matches: m[pp[p][0],aa[g][0]]+=1
            for g,a in enumerate(aa):
                if g not in used_g: m[4,a[0]]+=1
            for p,row in enumerate(pp):
                if p not in used_p: m[row[0],4]+=1
        assert m[:,:4].sum()==1636
        assert m[:4,:].sum()==prediction_total
        assert m.sum(axis=0)[:4].tolist()==[support[c] for c in range(4)]
        assert m[4,4]==0
        correct=int(np.trace(m[:4,:4])); wrong=int(m[:4,:4].sum()-correct)
        stats=dict(correct_class_spatial_matches=correct,wrong_class_spatial_matches=wrong,unmatched_gt=int(m[4,:4].sum()),unmatched_predictions=int(m[:4,4].sum()),predictions_at_conf_025=prediction_total,total_gt=1636)
        assert correct+wrong+stats['unmatched_gt']==1636
        assert correct+wrong+stats['unmatched_predictions']==prediction_total
        matrices[slug]=m; statistics[slug]=stats
        with (OUT/(slug+'_confusion_matrix_class_confusion.csv')).open('x',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['predicted_class / true_class']+NAMES)
            for i in range(5): w.writerow([NAMES[i]]+m[i].tolist())
    vmax=max(m.max() for m in matrices.values())
    for slug,m in matrices.items():
        for normalized in [False,True]:
            values=m.astype(float)
            if normalized: values=np.divide(values,values.sum(axis=0),out=np.zeros_like(values),where=values.sum(axis=0)>0)
            fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
            ax.imshow(values,cmap='Blues',vmin=0,vmax=1 if normalized else vmax)
            for i in range(5):
                for j in range(5): ax.text(j,i,f'{values[i,j]:.2f}' if normalized else str(m[i,j]),ha='center',va='center',color='white' if values[i,j]>(.5 if normalized else vmax*.5) else '#222',fontsize=11)
            ax.set(xticks=range(5),yticks=range(5),xticklabels=NAMES,yticklabels=NAMES,xlabel='True Class',ylabel='Predicted Class',title=('YOLO11n Clean V2' if slug=='yolo11n' else 'SKN-1')+'\nSpatial-first multiclass confusion'+('\nNormalized by GT column' if normalized else '\nRaw counts'))
            plt.setp(ax.get_xticklabels(),rotation=30,ha='right')
            fig.suptitle('Common test set: conf >=0.25, IoU >=0.50\nOne-to-one, highest-IoU-first matching; class equality not required',fontsize=11)
            filename=slug+'_confusion_matrix_class_confusion'+('_normalized.png' if normalized else '.png')
            fig.savefig(OUT/filename,dpi=180,bbox_inches='tight',facecolor='white'); plt.close(fig)
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    metadata=dict(confidence=.25,iou=.5,images=248,objects=1636,matching='Class-agnostic spatial matching. Greedy descending IoU; skip edges when either endpoint is already used. Ties: GT index then prediction index. Inclusive IoU >=0.50.',normalization='GT columns; nonempty column sums to 1; background column represents unmatched-prediction distribution, not normal_skin/TN.',ultralytics_note='Similar interpretation, not guaranteed byte-identical to an Ultralytics version-specific matching implementation. No inference or metrics rerun.',statistics=statistics,existing_files_preserved=True,input_and_existing_output_sha256=before)
    (OUT/'class_confusion_validation.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(statistics=statistics,matrices={k:v.tolist() for k,v in matrices.items()}),indent=2))
if __name__=='__main__':
    # Wrong class at identical location must match; duplicate predictions remain unmatched.
    assert spatial_pairs([[0,0,10,10]],[[0,0,10,10],[0,0,10,10]])==[(0,0)]
    assert spatial_pairs([],[[0,0,10,10]])==[]
    assert spatial_pairs([[0,0,10,10]],[])==[]
    assert spatial_pairs([[0,0,10,10]],[[0,0,5,10]])==[(0,0)]  # inclusive IoU=0.50
    main()
