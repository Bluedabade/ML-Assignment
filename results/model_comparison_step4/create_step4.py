"""Read-only inputs: genuine training history and saved Phase 5B predictions."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
import csv,json,hashlib,shutil
from collections import defaultdict,Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
BENCH=ROOT/'results/external_model_benchmark'
TRAIN=ROOT/'runs/detect/results/training_v2/yolo11n_clean'
DATA=ROOT/'datasets/processed/skin_detection_v2'
NAMES=['acne','wrinkle','dark_spot','enlarged_pore']
LABELS=NAMES+['background']
MODELS={'yolo11n':'YOLO11n Clean V2','skn1':'SKN-1'}
COLORS=['#2878b5','#e17c05','#6b4395','#168879']
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj): p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def csvwrite(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def savefig(fig,path):
    fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white'); plt.close(fig)
def match_flags(rows,gt):
    truth=defaultdict(list)
    for a in gt['annotations']:
        x,y,w,h=a['bbox']; truth[(a['image_id'],a['category_id'])].append([x,y,x+w,y+h])
    grouped=defaultdict(list)
    for i,p in enumerate(rows): grouped[(p['image_id'],p['class_id'])].append(i)
    flags=np.zeros(len(rows),dtype=bool)
    for key,indices in grouped.items():
        boxes=np.array(truth[key],float).reshape(-1,4); used=np.zeros(len(boxes),bool)
        for i in sorted(indices,key=lambda i:-rows[i]['confidence']):
            if not len(boxes): continue
            p=rows[i]; b=np.array([p['x1'],p['y1'],p['x2'],p['y2']])
            inter=np.maximum(np.minimum(b[2:],boxes[:,2:])-np.maximum(b[:2],boxes[:,:2]),0).prod(axis=1)
            union=np.maximum(b[2:]-b[:2],0).prod()+np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(axis=1)-inter
            iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0); iou[used]=-1
            best=int(iou.argmax())
            if iou[best]>=0.5: flags[i]=True; used[best]=True
    return flags
def matrix(rows,flags,support):
    m=np.zeros((5,5),int)
    for p,tp in zip(rows,flags):
        if p['confidence']>=0.25: m[p['class_id'],p['class_id'] if tp else 4]+=1
    for c in range(4): m[4,c]=support[c]-m[c,c]
    return m
def draw_matrix(ax,m,title,normalized=False,vmax=None):
    values=m.astype(float)
    if normalized:
        values=np.divide(values,values.sum(axis=0),out=np.zeros_like(values),where=values.sum(axis=0)>0)
    im=ax.imshow(values,cmap='Blues',vmin=0,vmax=1 if normalized else vmax)
    for i in range(5):
        for j in range(5):
            txt=f'{values[i,j]:.2f}' if normalized else str(m[i,j])
            ax.text(j,i,txt,ha='center',va='center',fontsize=10,color='white' if values[i,j]>(0.5 if normalized else vmax*0.5) else '#222')
    ax.set(xticks=range(5),yticks=range(5),xticklabels=LABELS,yticklabels=LABELS,xlabel='Ground truth',ylabel='Predicted',title=title)
    plt.setp(ax.get_xticklabels(),rotation=30,ha='right')
    return im
def curve_points(rows,flags,support,thresholds,c=None):
    scores=np.array([p['confidence'] for p in rows]); cls=np.array([p['class_id'] for p in rows])
    sel=np.ones(len(rows),bool) if c is None else cls==c
    s=scores[sel]; t=flags[sel]; order=np.argsort(-s,kind='stable'); s=s[order]; t=t[order]
    cumulative=np.r_[0,np.cumsum(t)]; n=np.searchsorted(-s,-thresholds,side='right')
    tp=cumulative[n]; total=sum(support.values()) if c is None else support[c]
    precision=np.divide(tp,n,out=np.zeros_like(tp,dtype=float),where=n>0); recall=tp/total
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(precision),where=precision+recall>0)
    return precision,recall,f1,tp,n
def main():
    for d in ['learning_curves','confusion_matrices','evaluation_curves','prediction_examples','tables']: (OUT/d).mkdir(exist_ok=True)
    inputs=[DATA/'annotations/instances_test.json',TRAIN/'results.csv',TRAIN/'results.png',TRAIN/'weights/best.pt']+list(BENCH.rglob('*.json'))+list((BENCH/'shared_evaluation').glob('*.csv'))
    before={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    gt=read(DATA/'annotations/instances_test.json'); cats={c['id']:c['name'] for c in gt['categories']}
    assert cats==dict(enumerate(NAMES)); assert len(gt['images'])==248 and len(gt['annotations'])==1636
    support=Counter(a['category_id'] for a in gt['annotations']); assert list(support[c] for c in range(4))==[1415,89,15,117]
    history=list(csv.DictReader((TRAIN/'results.csv').open())); epochs=np.array([int(r['epoch']) for r in history]); assert len(history)==28 and epochs.tolist()==list(range(1,29))
    map95=np.array([float(r['metrics/mAP50-95(B)']) for r in history]); best=int(epochs[map95.argmax()]); assert best==13
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ax,loss in zip(axes[0],['box','cls','dfl']):
        for split in ['train','val']: ax.plot(epochs,[float(r[f'{split}/{loss}_loss']) for r in history],label=split)
        ax.set(title=f'{loss.upper()} loss',xlabel='Epoch',ylabel='Loss'); ax.legend()
    for ax,keys,title in [(axes[1,0],['precision','recall'],'Validation Precision / Recall'),(axes[1,1],['mAP50','mAP50-95'],'Validation mAP')]:
        for k in keys: ax.plot(epochs,[float(r[f'metrics/{k}(B)']) for r in history],label=k)
        ax.set(title=title,xlabel='Epoch',ylabel='Metric'); ax.legend()
    axes[1,2].axis('off'); axes[1,2].text(0,0.85,'Genuine YOLO11n training history\n28 epochs; Best epoch = 13\n\nSKN-1: training history unavailable\nNo estimated training lines',fontsize=14,va='top')
    for ax in list(axes.flat)[:5]: ax.axvline(13,color='#444',ls='--',lw=1); ax.grid(alpha=.2)
    fig.suptitle('YOLO11n Clean V2: Training / Validation Learning Curve',fontsize=18)
    savefig(fig,OUT/'learning_curves/yolo11n_learning_curve.png')
    shutil.copyfile(TRAIN/'results.png',OUT/'learning_curves/yolo11n_results_original.png')
    write(OUT/'learning_curves/training_history_status.json',{'yolo11n':{'status':'AVAILABLE','epochs':28,'best_epoch':13,'source':str((TRAIN/'results.csv').relative_to(ROOT))},'skn1':{'status':'NOT AVAILABLE','reason':'External pretrained model; original epoch-by-epoch training logs were not available in downloaded artifacts.'}})
    overall=list(csv.DictReader((BENCH/'shared_evaluation/overall_metrics.csv').open())); perclass=list(csv.DictReader((BENCH/'shared_evaluation/per_class_metrics.csv').open())); secondary=read(BENCH/'shared_evaluation/secondary_precision_recall.json')
    combined=[]
    for r in overall: combined.append(dict(r,**secondary[r['model']]['micro']))
    csvwrite(OUT/'tables/performance_metrics.csv',combined); shutil.copyfile(BENCH/'shared_evaluation/per_class_metrics.csv',OUT/'tables/per_class_metrics.csv')
    rows_by_model={}; flags_by_model={}; matrices={}; perimage={};
    for slug,label in MODELS.items():
        export=read(BENCH/slug/'canonical_predictions.json'); assert len(export['processed_images'])==248
        rows=export['predictions']; flags=match_flags(rows,gt); m=matrix(rows,flags,support)
        s=secondary[label]['micro']; assert int(np.trace(m[:4,:4]))==s['tp'] and int(m[:4,4].sum())==s['fp'] and int(m[4,:4].sum())==s['fn']
        for c in range(4): assert m[c,c]==secondary[label]['per_class'][NAMES[c]]['tp']
        rows_by_model[slug]=rows; flags_by_model[slug]=flags; matrices[slug]=m
        stats={im['id']:{'tp':0,'fp':0,'fn':sum(a['image_id']==im['id'] for a in gt['annotations'])} for im in gt['images']}
        for p,tp in zip(rows,flags):
            if p['confidence']>=.25: stats[p['image_id']]['tp' if tp else 'fp']+=1; stats[p['image_id']]['fn']-=int(tp)
        perimage[slug]=stats
        csvwrite(OUT/f'confusion_matrices/{slug}_confusion_matrix.csv',[dict(predicted=LABELS[i],**dict(zip(LABELS,map(int,m[i])))) for i in range(5)])
    vmax=max(m.max() for m in matrices.values())
    for slug,m in matrices.items():
        for norm in [False,True]:
            fig,ax=plt.subplots(figsize=(8,7),layout='constrained'); image=draw_matrix(ax,m,MODELS[slug]+('\nColumn normalized' if norm else '\nRaw counts'),norm,vmax); fig.colorbar(image,ax=ax,shrink=.8)
            fig.suptitle('Test: confidence >=0.25, IoU >=0.50; class-aware matching',fontsize=12)
            savefig(fig,OUT/('confusion_matrices/'+slug+'_confusion_matrix'+('_normalized.png' if norm else '.png')))
    thresholds=np.unique(np.r_[np.geomspace(.001,.1,70),np.linspace(.1,1,181),.25]); curve_rows=[]
    curves={}
    for slug in MODELS:
        for c in [None,0,1,2,3]:
            p,r,f,t,n=curve_points(rows_by_model[slug],flags_by_model[slug],support,thresholds,c); curves[(slug,c)]=(p,r,f)
            for i,threshold in enumerate(thresholds): curve_rows.append(dict(model=MODELS[slug],class_name='overall_micro' if c is None else NAMES[c],confidence=float(threshold),precision=float(p[i]),recall=float(r[i]),f1=float(f[i]),tp=int(t[i]),predictions=int(n[i])))
    csvwrite(OUT/'tables/evaluation_curve_points.csv',curve_rows)
    for kind in ['pr','f1','precision','recall']:
        fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
        for ax,c in zip(axes.flat,[None,0,1,2,3]):
            for slug,color in [('yolo11n','#2878b5'),('skn1','#e17c05')]:
                p,r,f=curves[(slug,c)]; x=r[::-1] if kind=='pr' else thresholds; y=p[::-1] if kind=='pr' else {'f1':f,'precision':p,'recall':r}[kind]
                ax.plot(x,y,label=MODELS[slug],color=color,lw=2)
            ax.set(title='Overall (micro)' if c is None else NAMES[c],xlabel='Recall' if kind=='pr' else 'Confidence',ylabel='Precision' if kind=='pr' else kind.title(),ylim=(0,1.02)); ax.grid(alpha=.2)
            if kind!='pr': ax.axvline(.25,color='#666',ls='--',lw=1)
            ax.legend(fontsize=9)
        axes[1,2].axis('off'); axes[1,2].text(0,.8,'TEST-SET evaluation only\nIoU >=0.50, class-aware one-to-one\nSaved predictions: confidence >=0.001\nNo extrapolation below the floor\nNot training Learning Curves\nNo test-set threshold tuning',fontsize=12,va='top')
        fig.suptitle({'pr':'Precision-Recall','f1':'F1 vs Confidence','precision':'Precision vs Confidence','recall':'Recall vs Confidence'}[kind]+' — common test set',fontsize=18)
        savefig(fig,OUT/f'evaluation_curves/{kind}_curve_comparison.png')
    images=sorted(gt['images'],key=lambda im:im['file_name']); ann_by_image=defaultdict(list)
    for a in gt['annotations']: ann_by_image[a['image_id']].append(a)
    selected=[]; selected_ids=set()
    def choose(candidates,reason):
        for im in candidates:
            if im['id'] not in selected_ids: selected.append((im,reason)); selected_ids.add(im['id']); return
    for c in range(4): choose([im for im in images if any(a['category_id']==c for a in ann_by_image[im['id']])],f'Lexicographic first unused image containing {NAMES[c]}')
    choose([im for im in images if perimage['yolo11n'][im['id']]['tp']>0 and perimage['skn1'][im['id']]['tp']>0],'First unused image with at least one TP for both models')
    choose([im for im in images if perimage['yolo11n'][im['id']]['tp']>0 and perimage['skn1'][im['id']]['tp']==0],'First unused YOLO-only TP image')
    choose([im for im in images if perimage['skn1'][im['id']]['tp']>0 and perimage['yolo11n'][im['id']]['tp']==0],'First unused SKN-only TP image')
    choose([im for im in images if ann_by_image[im['id']] and perimage['skn1'][im['id']]['tp']==0 and perimage['yolo11n'][im['id']]['tp']==0],'First unused positive image missed by both models')
    if len(selected)<8: choose([im for im in images if not ann_by_image[im['id']]],'First unused negative image (no annotated target objects)')
    examples=[]
    for i,(im,reason) in enumerate(selected[:8],1):
        fig,axes=plt.subplots(1,3,figsize=(18,7),layout='constrained'); image=Image.open(DATA/'images/test'/im['file_name']).convert('RGB')
        for ax,title,slug in zip(axes,['Ground Truth','YOLO11n Clean V2','SKN-1'],[None,'yolo11n','skn1']):
            ax.imshow(image); ax.axis('off')
            if slug is None: detections=[dict(class_id=a['category_id'],x1=a['bbox'][0],y1=a['bbox'][1],x2=a['bbox'][0]+a['bbox'][2],y2=a['bbox'][1]+a['bbox'][3]) for a in ann_by_image[im['id']]]
            else: detections=[p for p in rows_by_model[slug] if p['image_id']==im['id'] and p['confidence']>=.25]
            for p in detections:
                ax.add_patch(Rectangle((p['x1'],p['y1']),p['x2']-p['x1'],p['y2']-p['y1'],fill=False,edgecolor=COLORS[p['class_id']],linewidth=1))
                if slug: ax.text(p['x1'],p['y1'],f"{p['confidence']:.2f}",fontsize=5,color='white',bbox=dict(facecolor=COLORS[p['class_id']],alpha=.8,pad=.2))
            extra='' if slug is None else f"; TP {perimage[slug][im['id']]['tp']} / FP {perimage[slug][im['id']]['fp']} / FN {perimage[slug][im['id']]['fn']}"
            ax.set_title(f'{title}\n{len(detections)} boxes'+extra,fontsize=12)
        fig.suptitle(f"Example {i:02d}: {im['file_name']}\n{reason}",fontsize=12)
        fig.legend(handles=[Rectangle((0,0),1,1,color=COLORS[c],label=NAMES[c]) for c in range(4)],loc='lower center',ncols=4)
        path=f'prediction_examples/example_{i:02d}.png'; savefig(fig,OUT/path)
        examples.append(dict(image_id=im['id'],filename=im['file_name'],selection_reason=reason,output=path,yolo11n=perimage['yolo11n'][im['id']],skn1=perimage['skn1'][im['id']]))
    write(OUT/'prediction_examples/selection_manifest.json',examples)
    fig=plt.figure(figsize=(16,11),layout='constrained'); gs=fig.add_gridspec(3,2,height_ratios=[1.2,2,0.6])
    ax=fig.add_subplot(gs[0,0]); ax.axis('off')
    cell=[[r['model'],f"{float(r['mAP50']):.4f}",f"{float(r['mAP50_95']):.4f}",f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1']:.4f}"] for r in combined]
    table=ax.table(cellText=cell,colLabels=['Model','mAP50','mAP50-95','P','R','F1'],loc='center',colWidths=[.32,.13,.17,.12,.12,.12]); table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,2)
    ax.set_title('Shared metrics | P/R/F1 at conf=0.25, IoU=0.50')
    ax=fig.add_subplot(gs[0,1]); x=np.arange(4)
    for j,(slug,label) in enumerate(MODELS.items()): ax.bar(x+(j-.5)*.35,[float(next(r['AP50'] for r in perclass if r['model']==label and r['class_name']==c)) for c in NAMES],.35,label=label)
    ax.set(xticks=x,xticklabels=NAMES,ylabel='AP50',title='Shared per-class AP50',ylim=(0,.65)); ax.legend()
    for j,(slug,m) in enumerate(matrices.items()): draw_matrix(fig.add_subplot(gs[1,j]),m,MODELS[slug]+' — normalized by GT column',True,vmax)
    ax=fig.add_subplot(gs[2,:]); ax.axis('off'); ax.text(0,.9,'SKN-1 training Learning Curve unavailable: original training history was not provided.\nExternal training/test overlap cannot be ruled out. CPU/GPU timing is not directly comparable.\n248 test images / 1,636 GT objects. These evaluation results are not medical diagnoses.',fontsize=13,va='top')
    fig.suptitle('Skin Detection: Step 4 Performance Comparison',fontsize=20); savefig(fig,OUT/'step4_summary.png')
    write(OUT/'tables/model_information.json',{'yolo11n':{'parameters':2582932,'GFLOPs':6.4,'source':'Project recorded model information','backend':'CUDA GTX 1650 Ti'},'skn1':{'architecture':'YOLO-NAS external pretrained object detector','parameters':'N/A — not independently verified','backend':'ONNX CPUExecutionProvider'},'inference':{slug:read(BENCH/slug/'inference_summary.json') for slug in MODELS}})
    write(OUT/'tables/evaluation_method.json',{'confidence':.25,'iou':.5,'class_order':LABELS,'matrix_rows':'predicted','matrix_columns':'ground_truth','normalization':'each ground-truth column sums to 1 if nonempty','matching':'Phase 5B class-aware, descending confidence, greedy maximum IoU to unmatched GT. Wrong-class predictions are FP plus FN, not off-diagonal class assignments. Background/background=0; TN undefined.','curves':'Micro and per-class raw operating-point curves on fixed post-NMS predictions; not COCO-interpolated AP curves. No predictions below confidence 0.001.','thresholds':thresholds.tolist(),'source_evaluator':read(BENCH/'shared_evaluation/evaluation_config.json')})
    summary=['# Step 4: Evaluate and display Performance Metrics','', '## 1. Load / identify best model',f'YOLO11n: `{(TRAIN/"weights/best.pt").relative_to(ROOT)}`. SKN-1: `skn-1/2`, external YOLO-NAS. No inference, training or fine-tuning was rerun; Phase 5B exports are reused.','## 2. Performance Metrics','![Shared comparison](step4_summary.png)','Primary mAP: shared COCOeval from Phase 5B. Secondary P/R/F1: class-aware one-to-one matching at confidence >=0.25 and IoU >=0.50. Tables: tables/performance_metrics.csv and tables/per_class_metrics.csv.','## 3. Learning Curve','![True training history](learning_curves/yolo11n_learning_curve.png)','YOLO11n: genuine 28-epoch training/validation history, Best epoch = 13 (highest recorded validation mAP50-95). Original results.png was copied byte-identically. SKN-1 Training Learning Curve: NOT AVAILABLE; original epoch logs absent. No estimated lines.','## 4. Confusion Matrix','![YOLO](confusion_matrices/yolo11n_confusion_matrix.png)','![SKN](confusion_matrices/skn1_confusion_matrix.png)','Both share identical test set, mapping, thresholds and Phase 5B matching rule. Rows=predicted, columns=ground truth; normalized versions normalize each GT column. Background row=FN; background column=FP. There are no detection TN counts. Class-aware matching means wrong-class detections count as FP+FN, not class-to-class off-diagonal confusion. This is not the historical Ultralytics confusion-matrix algorithm.','## 5. F1 / PR curves','![F1](evaluation_curves/f1_curve_comparison.png)','![PR](evaluation_curves/pr_curve_comparison.png)','Additional precision/recall vs confidence figures and numerical curve points are included. These are TEST-SET evaluation curves, not training Learning Curves or COCO AP interpolation. Confidence range starts at the saved 0.001 floor. Do not use test curves to tune thresholds.','## 6. Prediction examples',f'{len(examples)} deterministic comparison examples: GT / YOLO / SKN at confidence >=0.25. selection_manifest.json records selection criteria, including both-miss and negative cases. This illustrative set is not an unbiased sample.','## 7. Model comparison','SKN-1 has higher shared AP in all four classes on this test set. YOLO11n: 2,582,932 parameters, recorded 6.4 GFLOPs. SKN parameter count: N/A. Inference timing and backend metadata are in tables/model_information.json; CPU vs GPU is not a fair speed comparison.','## 8. Limitations','External SKN-1 training-data overlap with our test set cannot be completely ruled out. Minority-class support is small (dark_spot 15, wrinkle 89 objects). SKN original training history is unavailable. Prediction preprocessing/NMS differ by model; historical Ultralytics val uses multi-label NMS whereas saved YOLO predict exports use single-label NMS. Historical results were preserved, not substituted into shared tables. No medical diagnostic claim.','Reference mapping: results.png -> genuine learning curve + preserved original; confusion_matrix.png -> common-test matrices; BoxF1_curve.png -> test F1 comparison; val_batch0_pred.jpg -> three-panel prediction examples.']
    summary=[s.replace('including both-miss and negative cases','including class-coverage, both-model TP, YOLO-only TP, SKN-only TP and both-model miss cases') for s in summary]
    summary.append('Curve convention: Precision and F1=0 when no predictions are retained (zero-division convention), not evidence of false positives. PR curves are raw operating-point samples, not COCO-interpolated AP curves.')
    (OUT/'step4_summary.md').write_text('\n\n'.join(summary),encoding='utf-8')
    assert before=={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    assert digest(TRAIN/'results.png')==digest(OUT/'learning_curves/yolo11n_results_original.png')
    write(OUT/'verification.json',{'input_hashes_unchanged':True,'input_sha256':before,'original_results_png_byte_identical':True,'matrix_reconciles_phase5b':True,'training_epochs':28,'best_epoch':13,'prediction_examples':len(examples),'images':248,'objects':1636})
    print(json.dumps({'status':'PASS','best_epoch':best,'epochs':len(history),'matrices':{k:v.tolist() for k,v in matrices.items()},'examples':len(examples)},indent=2))
if __name__=='__main__': main()
