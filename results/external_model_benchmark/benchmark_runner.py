"""Phase 5B: local predictions and one shared COCO evaluator; no training."""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'datasets/processed/skin_detection_v2'
GT_PATH=DATA/'annotations/instances_test.json'
ENV=Path('D:/Projects/ML-Assignment-roboflow-env')
NAMES=['acne','wrinkle','dark_spot','enlarged_pore']
MAPPING={'Acne':'acne','Wrinkles':'wrinkle','Dark Spots':'dark_spot','Open Pores':'enlarged_pore'}

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2)

def load(path): return json.loads(path.read_text(encoding='utf-8'))

def preflight():
    gt=load(GT_PATH)
    cats={c['id']:c['name'] for c in gt['categories']}
    assert set(cats.values())==set(NAMES) and len(cats)==4
    counts=Counter(cats[a['category_id']] for a in gt['annotations'])
    assert counts=={'acne':1415,'wrinkle':89,'dark_spot':15,'enlarged_pore':117}
    assert len(gt['images'])==248 and len(gt['annotations'])==1636
    assert all(not a.get('iscrowd',0) and not a.get('ignore',0) for a in gt['annotations'])
    assert {p.name for p in (DATA/'images/test').iterdir() if p.is_file()}=={im['file_name'] for im in gt['images']}
    annotations=defaultdict(list)
    for a in gt['annotations']: annotations[a['image_id']].append(a)
    import numpy as np
    for im in gt['images']:
        lines=[l.split() for l in (DATA/'labels/test'/(Path(im['file_name']).stem+'.txt')).read_text().splitlines() if l.strip()]
        aa=annotations[im['id']]
        assert len(lines)==len(aa)
        for line,a in zip(lines,aa):
            x,y,w,h=a['bbox']
            assert int(line[0])==a['category_id']
            assert np.allclose(list(map(float,line[1:])),[(x+w/2)/im['width'],(y+h/2)/im['height'],w/im['width'],h/im['height']],atol=2e-6)
    return gt,cats,sorted(gt['images'],key=lambda im:im['file_name'])

def prediction(im,cls,cid,conf,box):
    return dict(filename=im['file_name'],image_id=im['id'],class_name=cls,class_id=int(cid),confidence=float(conf),
                x1=float(box[0]),y1=float(box[1]),x2=float(box[2]),y2=float(box[3]))

def skn():
    gt,cats,images=preflight()
    key=os.environ.get('ROBOFLOW_API_KEY')
    assert key
    os.environ['INFERENCE_HOME']=str(ENV/'model_cache')
    os.environ['DEFAULT_DEVICE']='cpu'
    os.environ['ONNXRUNTIME_EXECUTION_PROVIDERS']='CPUExecutionProvider'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    logging.disable(logging.CRITICAL)
    captured=io.StringIO()
    paths=[]
    with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
        import cv2
        from inference_models import AutoModel
        model=AutoModel.from_pretrained('skn-1/2',api_key=key,backend='onnx',device='cpu',
            onnx_execution_providers=['CPUExecutionProvider'],verbose=False,
            allow_untrusted_packages=False,allow_local_code_packages=False,point_model_directory=paths.append)
    assert model._session.get_providers()==['CPUExecutionProvider']
    assert set(MAPPING)<=set(model.class_names)
    raw=[]; canonical=[]; excluded=[]; times=[]; distribution=Counter()
    category_ids={name:cid for cid,name in cats.items()}
    for index,im in enumerate(images,1):
        image=cv2.imread(str(DATA/'images/test'/im['file_name']))
        assert image is not None
        with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
            start=time.perf_counter()
            pred=model(image,confidence=0.001,iou_threshold=0.3,max_detections=300,class_agnostic_nms=False)[0]
            times.append(time.perf_counter()-start)
        for cid,conf,box in zip(pred.class_id.tolist(),pred.confidence.tolist(),pred.xyxy.tolist()):
            label=model.class_names[int(cid)]
            row=prediction(im,label,cid,conf,box)
            raw.append(row); distribution[label]+=1
            if label in MAPPING:
                mapped=dict(row,class_name=MAPPING[label],class_id=category_ids[MAPPING[label]],raw_class_name=label,raw_class_id=int(cid))
                canonical.append(mapped)
            else: excluded.append(row)
        if index%20==0 or index==248: print('SKN-1 processed',index,'/248',flush=True)
    folder=OUT/'skn1'
    for name,rows in [('raw_predictions',raw),('canonical_predictions',canonical),('out_of_scope_predictions',excluded)]:
        save(folder/(name+'.json'),dict(processed_images=[im['file_name'] for im in images],predictions=rows))
    save(folder/'inference_summary.json',dict(model_id='skn-1/2',model_type=type(model).__name__,backend='ONNX CPU',execution_providers=model._session.get_providers(),cache_paths=paths,
        confidence_floor=0.001,nms_iou=0.3,max_detections=300,class_agnostic_nms=False,
        preprocessing=model._inference_config.model_dump(mode='json'),input_shape=model._session.get_inputs()[0].shape,
        processed_images=248,total_inference_seconds=sum(times),mean_inference_seconds=sum(times)/248,
        per_image_seconds=dict(zip([im['file_name'] for im in images],times)),raw_class_counts={name:distribution[name] for name in model.class_names},
        canonical_prediction_count=len(canonical),out_of_scope_count=len(excluded),class_mapping=MAPPING,
        packages={name:importlib.metadata.version(name) for name in ['inference-models','onnxruntime','torch','numpy']}))

def yolo():
    gt,cats,images=preflight()
    (ENV/'yolo_settings').mkdir(exist_ok=True)
    (ENV/'matplotlib_cache').mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(ENV/'yolo_settings')
    os.environ['MPLCONFIGDIR']=str(ENV/'matplotlib_cache')
    import torch
    from ultralytics import YOLO
    checkpoint=ROOT/'runs/detect/results/training_v2/yolo11n_clean/weights/best.pt'
    model=YOLO(str(checkpoint))
    assert set(model.names.values())==set(NAMES)
    device=0 if torch.cuda.is_available() else 'cpu'
    ids={name:cid for cid,name in cats.items()}
    rows=[]; times=[]
    for index,im in enumerate(images,1):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start=time.perf_counter()
        result=model.predict(str(DATA/'images/test'/im['file_name']),imgsz=640,conf=0.001,max_det=300,
            iou=0.7,agnostic_nms=False,device=device,save=False,save_txt=False,save_conf=False,verbose=False)[0]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        for cid,conf,box in zip(result.boxes.cls.tolist(),result.boxes.conf.tolist(),result.boxes.xyxy.tolist()):
            label=model.names[int(cid)]
            rows.append(prediction(im,label,ids[label],conf,box))
        if index%40==0 or index==248: print('YOLO11n processed',index,'/248',flush=True)
    save(OUT/'yolo11n/canonical_predictions.json',dict(processed_images=[im['file_name'] for im in images],predictions=rows))
    save(OUT/'yolo11n/inference_summary.json',dict(model_path=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        classes=model.names,device=device,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        conf=0.001,imgsz=640,nms_iou=model.predictor.args.iou,max_det=model.predictor.args.max_det,agnostic_nms=model.predictor.args.agnostic_nms,
        rect=model.predictor.args.rect,multi_label=False,processed_images=248,total_inference_seconds=sum(times),mean_inference_seconds=sum(times)/248,
        preprocessing='Normal Ultralytics predict letterbox and normalization; no custom transforms',
        packages={n:importlib.metadata.version(n) for n in ['ultralytics','torch','torchvision']}))

def pr(predictions,gt,cats):
    import numpy as np
    truth=defaultdict(list)
    for a in gt['annotations']:
        x,y,w,h=a['bbox']; truth[(a['image_id'],a['category_id'])].append([x,y,x+w,y+h])
    predictions_by_key=defaultdict(list)
    for p in predictions:
        if p['confidence']>=0.25: predictions_by_key[(p['image_id'],p['class_id'])].append(p)
    tp=Counter(); fp=Counter(); supports=Counter(a['category_id'] for a in gt['annotations'])
    for key,rows in predictions_by_key.items():
        boxes=np.array(truth[key],dtype=float).reshape(-1,4); used=set()
        for p in sorted(rows,key=lambda p:-p['confidence']):
            b=np.array([p['x1'],p['y1'],p['x2'],p['y2']]); best=-1; best_iou=-1
            for j,g in enumerate(boxes):
                if j in used: continue
                inter=np.maximum(np.minimum(b[2:],g[2:])-np.maximum(b[:2],g[:2]),0).prod()
                union=np.maximum(b[2:]-b[:2],0).prod()+np.maximum(g[2:]-g[:2],0).prod()-inter
                iou=inter/union if union>0 else 0
                if iou>=0.5 and iou>best_iou: best=j; best_iou=iou
            if best>=0: used.add(best); tp[key[1]]+=1
            else: fp[key[1]]+=1
    def metrics(t,f,s):
        precision=t/(t+f) if t+f else 0; recall=t/s if s else 0
        return dict(tp=t,fp=f,fn=s-t,precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0)
    return dict(micro=metrics(sum(tp.values()),sum(fp.values()),sum(supports.values())),per_class={cats[c]:metrics(tp[c],fp[c],supports[c]) for c in cats})

def evaluate():
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt,cats,images=preflight()
    raw_export=load(OUT/'skn1/raw_predictions.json')
    mapped_export=load(OUT/'skn1/canonical_predictions.json')
    excluded_export=load(OUT/'skn1/out_of_scope_predictions.json')
    assert len(raw_export['predictions'])==len(mapped_export['predictions'])+len(excluded_export['predictions'])
    assert Counter(p['class_name'] for p in raw_export['predictions'])==Counter(load(OUT/'skn1/inference_summary.json')['raw_class_counts'])
    assert all(p['class_name'] not in MAPPING for p in excluded_export['predictions'])
    assert all(MAPPING[p['raw_class_name']]==p['class_name'] for p in mapped_export['predictions'])
    for export in [raw_export,mapped_export,excluded_export,load(OUT/'yolo11n/canonical_predictions.json')]:
        assert len(set(export['processed_images']))==248
        assert all(np.isfinite([p['confidence'],p['x1'],p['y1'],p['x2'],p['y2']]).all() and 0.001<=p['confidence']<=1 for p in export['predictions'])
        assert max(Counter(p['image_id'] for p in export['predictions']).values(),default=0)<=300
    key=os.getenv('ROBOFLOW_API_KEY')
    if key:
        assert all(key not in p.read_text(encoding='utf-8') for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.py','.csv','.md'])
    sink=io.StringIO(); overall=[]; per_class=[]; secondary={}
    with contextlib.redirect_stdout(sink): coco=COCO(str(GT_PATH))
    for label,folder in [('YOLO11n Clean V2','yolo11n'),('SKN-1','skn1')]:
        export=load(OUT/folder/'canonical_predictions.json')
        assert set(export['processed_images'])=={im['file_name'] for im in images} and len(export['processed_images'])==248
        rows=export['predictions']
        assert all(cats[p['class_id']]==p['class_name'] and p['x2']>=p['x1'] and p['y2']>=p['y1'] for p in rows)
        dt=[dict(image_id=p['image_id'],category_id=p['class_id'],score=p['confidence'],bbox=[p['x1'],p['y1'],p['x2']-p['x1'],p['y2']-p['y1']]) for p in rows]
        with contextlib.redirect_stdout(sink):
            detections=coco.loadRes(dt)
            evaluator=COCOeval(coco,detections,'bbox')
            evaluator.params.imgIds=sorted(im['id'] for im in images)
            evaluator.params.catIds=sorted(cats)
            evaluator.params.maxDets=[1,10,300]
            evaluator.evaluate(); evaluator.accumulate()
        precision=evaluator.eval['precision'][:,:, :,0,2]
        def mean_valid(values):
            valid=values[values>=0]
            return float(valid.mean()) if len(valid) else None
        i50=int(np.flatnonzero(np.isclose(evaluator.params.iouThrs,0.5))[0])
        overall.append(dict(model=label,mAP50=mean_valid(precision[i50]),mAP50_95=mean_valid(precision)))
        for k,cid in enumerate(evaluator.params.catIds):
            per_class.append(dict(model=label,class_name=cats[cid],AP50=mean_valid(precision[i50,:,k]),AP50_95=mean_valid(precision[:,:,k])))
        secondary[label]=pr(rows,gt,cats)
    folder=OUT/'shared_evaluation'; folder.mkdir(exist_ok=True)
    for name,rows in [('overall_metrics',overall),('per_class_metrics',per_class)]:
        with (folder/(name+'.csv')).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(folder/'secondary_precision_recall.json',secondary)
    save(folder/'evaluation_config.json',dict(evaluator='pycocotools.COCOeval',version=importlib.metadata.version('pycocotools'),ground_truth_path=str(GT_PATH.relative_to(ROOT)),
        ground_truth_sha256=hashlib.sha256(GT_PATH.read_bytes()).hexdigest(),categories=cats,images=248,objects=1636,
        iou_type='bbox',iou_thresholds=evaluator.params.iouThrs.tolist(),recall_thresholds=evaluator.params.recThrs.tolist(),
        max_detections=[1,10,300],area_ranges=evaluator.params.areaRng,primary_area='all',use_categories=True,
        ap_extraction='Mean valid COCO precision tensor entries at area=all, maxDet=300; avoids summarize() default maxDet=100',
        secondary_method='Confidence >=0.25; class-specific per-image descending-score greedy unmatched-GT maximum IoU >=0.50; continuous xyxy; no crowd/ignore GT present',
        historical_metrics='Context only; not substituted for shared evaluation'))
    text=['# Same-test-set external pretrained comparison','',
        'Test: 248 images, 1,636 objects (acne 1,415; wrinkle 89; dark_spot 15; enlarged_pore 117). COCO JSON and YOLO labels agree.',
        '', '## Shared COCO evaluation', '',
        'COCOeval bbox; IoU 0.50:0.05:0.95; 101 recall points; area=all; maxDets=[1,10,300]. Values are fractions.',
        '', '| Model | mAP50 | mAP50-95 |', '|---|---:|---:|']
    text.extend(f"| {r['model']} | {r['mAP50']:.6f} | {r['mAP50_95']:.6f} |" for r in overall)
    text += ['', '| Model | Class | AP50 | AP50-95 |', '|---|---|---:|---:|']
    text.extend(f"| {r['model']} | {r['class_name']} | {r['AP50']:.6f} | {r['AP50_95']:.6f} |" for r in per_class)
    text += ['', '## Secondary operating point', '',
        'Same class-aware greedy matching: confidence >=0.25, IoU >=0.50. Not the historical Ultralytics P/R definition.', '',
        '| Model | Micro Precision | Micro Recall | Micro F1 |', '|---|---:|---:|---:|']
    text.extend(f"| {label} | {s['micro']['precision']:.6f} | {s['micro']['recall']:.6f} | {s['micro']['f1']:.6f} |" for label,s in secondary.items())
    text += ['', 'Per-class P/R and TP/FP/FN: shared_evaluation/secondary_precision_recall.json.', '', '## Inference protocol', '']
    for folder in ['skn1','yolo11n']:
        s=load(OUT/folder/'inference_summary.json')
        text += [f'### {folder}', '', '```json', json.dumps({k:v for k,v in s.items() if k!='per_image_seconds'},indent=2), '```', '']
    text += ['## Limitations and historical context', '',
        'SKN-1 external training provenance is not fully known relative to this project. Original training images may overlap this test set. This comparison is NOT guaranteed leakage-free.',
        'Published Roboflow metrics are not used. Historical Clean-V2 Ultralytics P=0.321 R=0.0878 mAP50=0.0568 mAP50-95=0.0158 are context only. COCO interpolation, maxDet, inference/NMS and aggregation differ.',
        'SKN-1 ONNX CPU timing must not be used to claim speed superiority/inferiority relative to YOLO CUDA timing.',
        'Timing covers model calls, excludes image reads/model loading and includes first-call initialization. Processes initially overlapped; this is not a controlled hardware-speed benchmark.',
        'Historical per-class Ultralytics mAP50: acne=0.138, wrinkle=0.000198, dark_spot=0.0514, enlarged_pore=0.0376. Existing outputs were not overwritten.',
        'Model-specific preprocessing/NMS preserved: SKN stretch 640 and NMS 0.3; YOLO letterbox imgsz=640 and NMS 0.7. Out-of-scope SKN labels can consume its 300-detection cap; they are not remapped.',
        'Ultralytics predict uses default single-label-per-candidate NMS, whereas detect val enables multi_label=True. This prediction protocol can discard secondary class scores (including dark_spot) and need not reproduce historical val AP. SKN NMS also selects each candidate\'s highest-scoring class before suppression.',
        'Minority support is limited, especially dark_spot (15 objects). No universal superiority or medical diagnostic claim is supported.',
        'SKN-1 dataset/project license shown: CC BY 4.0. Separate model-weight licensing not independently established.',
        'No training or fine-tuning. Test data and existing outputs remain unchanged.']
    (OUT/'benchmark_summary.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(dict(overall=overall,per_class=per_class,secondary=secondary),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['preflight','skn','yolo','evaluate']); args=parser.parse_args()
    try:
        if args.mode=='preflight': preflight(); print('GROUND_TRUTH_PREFLIGHT_PASS: 248 images / 1636 objects')
        else: {'skn':skn,'yolo':yolo,'evaluate':evaluate}[args.mode]()
    except Exception as exc:
        message=str(exc); key=os.getenv('ROBOFLOW_API_KEY')
        if key: message=message.replace(key,'[REDACTED]')
        message=re.sub(r'https?://[^\s\"\x27]+','[URL_REDACTED]',message)
        print(type(exc).__name__,message,flush=True)
        raise SystemExit(1)
