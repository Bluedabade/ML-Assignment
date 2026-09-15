"""Train MobileNetV2 head, optionally fine-tune, then evaluate."""
import json
from datetime import datetime
import tensorflow as tf
from src.config import *
from src.data_loader import create_generators
from src.model_builder import build_model,enable_fine_tuning
from src.evaluation import evaluate_model,save_history
def main():
 validate_config(); print("TensorFlow:",tf.__version__); print("GPU:",tf.config.list_physical_devices("GPU") or "ไม่พบ GPU")
 from scripts.check_dataset import inspect_dataset
 if not inspect_dataset(quiet=True)["valid"]: raise SystemExit("Dataset ไม่พร้อม: รัน python scripts/check_dataset.py")
 train,val,test=create_generators(); print("class_indices:",train.class_indices); MODELS_DIR.mkdir(parents=True,exist_ok=True); (MODELS_DIR/"class_indices.json").write_text(json.dumps(train.class_indices,indent=2),encoding="utf-8")
 run_dir=RESULTS_DIR/"runs"/datetime.now().strftime("%Y%m%d_%H%M%S"); run_dir.mkdir(parents=True); (run_dir/"config.json").write_text(json.dumps({k:repr(v) for k,v in globals().items() if k.isupper()},indent=2),encoding="utf-8")
 model,base=build_model(); model.summary(); checkpoint=MODELS_DIR/"mobilenetv2_5class_best.keras"; cb=[tf.keras.callbacks.ModelCheckpoint(checkpoint,monitor="val_accuracy",mode="max",save_best_only=True)]
 first=model.fit(train,validation_data=val,epochs=HEAD_EPOCHS,callbacks=cb); history={k:list(v) for k,v in first.history.items()}; fine=None
 if ENABLE_FINE_TUNING and FINE_TUNE_EPOCHS>0:
  fine=len(history["loss"]); enable_fine_tuning(model,base); second=model.fit(train,validation_data=val,initial_epoch=fine,epochs=fine+FINE_TUNE_EPOCHS,callbacks=cb)
  for k,v in second.history.items(): history.setdefault(k,[]).extend(v)
 save_history(history,run_dir,fine); best=tf.keras.models.load_model(checkpoint); metrics=evaluate_model(best,test,CLASS_NAMES,run_dir)
 for folder,name in [(PLOTS_DIR,"training_curves.png"),(PLOTS_DIR,"confusion_matrix.png"),(REPORTS_DIR,"metrics.json"),(REPORTS_DIR,"classification_report.txt")]: folder.mkdir(parents=True,exist_ok=True); (folder/name).write_bytes((run_dir/name).read_bytes())
 print("Model:",checkpoint,"\nRun:",run_dir,"\nMetrics:",metrics)
if __name__=="__main__":
 try: main()
 except (FileNotFoundError,ValueError) as e: raise SystemExit(f"ข้อผิดพลาด: {e}")
