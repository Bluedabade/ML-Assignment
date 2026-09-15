import csv,json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
def save_history(h,out,fine=None):
 keys=["accuracy","val_accuracy","loss","val_loss"]
 with (out/"mobilenetv2_history.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f); w.writerow(["epoch",*keys]); [w.writerow([i+1,*[h[k][i] for k in keys]]) for i in range(len(h["loss"]))]
 (out/"mobilenetv2_history.json").write_text(json.dumps(h,indent=2),encoding="utf-8"); fig,ax=plt.subplots(2,1,figsize=(8,9)); ep=range(1,len(h["loss"])+1)
 for a,k,v,title in [(ax[0],"accuracy","val_accuracy","model accuracy"),(ax[1],"loss","val_loss","model loss")]:
  a.plot(ep,h[k],label="train"); a.plot(ep,h[v],label="validation"); a.set(title=title,xlabel="epoch",ylabel=k); a.legend(); a.grid(alpha=.25)
  if fine is not None:a.axvline(fine+.5,color="gray",ls="--")
 fig.tight_layout(); fig.savefig(out/"training_curves.png",dpi=160); plt.close(fig)
def evaluate_model(model,test,names,out):
 test.reset(); pred=np.argmax(model.predict(test),axis=1); actual=test.classes; text=classification_report(actual,pred,target_names=names,digits=4,zero_division=0); d=classification_report(actual,pred,target_names=names,output_dict=True,zero_division=0)
 m={"accuracy":float(np.mean(actual==pred)),"macro_precision":d["macro avg"]["precision"],"macro_recall":d["macro avg"]["recall"],"macro_f1":d["macro avg"]["f1-score"],"weighted_precision":d["weighted avg"]["precision"],"weighted_recall":d["weighted avg"]["recall"],"weighted_f1":d["weighted avg"]["f1-score"]}; (out/"metrics.json").write_text(json.dumps(m,indent=2),encoding="utf-8"); (out/"classification_report.txt").write_text(text,encoding="utf-8")
 fig,ax=plt.subplots(figsize=(8,7)); ConfusionMatrixDisplay(confusion_matrix(actual,pred),display_labels=names).plot(ax=ax,cmap="Blues",xticks_rotation=25); fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=160); plt.close(fig); return m
