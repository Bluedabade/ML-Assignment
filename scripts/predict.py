import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import IMAGE_SIZE,MODELS_DIR
def main(path):
 import numpy as np,tensorflow as tf
 model_path=MODELS_DIR/"mobilenetv2_5class_best.keras"; index_path=MODELS_DIR/"class_indices.json"
 if not path.is_file(): raise SystemExit(f"ไม่พบรูปภาพ: {path}")
 if not model_path.is_file() or not index_path.is_file(): raise SystemExit("ไม่พบ model/class_indices กรุณารัน python train.py ก่อน")
 mapping=json.loads(index_path.read_text(encoding="utf-8")); names=[x[0] for x in sorted(mapping.items(),key=lambda x:x[1])]; image=tf.keras.utils.load_img(path,target_size=IMAGE_SIZE); arr=tf.keras.utils.img_to_array(image)/255.; prob=model.predict(np.expand_dims(arr,0),verbose=0)[0]; pairs=sorted(zip(names,prob),key=lambda x:x[1],reverse=True)
 print("="*48,"\nSkin Analysis Result\n"+"="*48)
 for name,p in pairs: print(f"{name:12}: {p*100:6.2f}%"+("  <-- highest" if name==pairs[0][0] else ""))
 print(f"\nPrediction : {pairs[0][0]}\nConfidence : {pairs[0][1]*100:.2f}%\nProbability sum: {sum(prob)*100:.2f}%\n\nไม่ใช่เปอร์เซ็นต์พื้นที่ผิวหรือการวินิจฉัยทางการแพทย์")
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("image",type=Path); a=p.parse_args(); main(a.image)
