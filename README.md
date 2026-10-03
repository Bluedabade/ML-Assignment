# Skin Detection - ML Assignment

โปรเจกต์นี้ใช้ YOLO ตรวจจับปัญหาผิว 4 คลาส: `0 acne`, `1 wrinkle`, `2 dark_spot` และ `3 enlarged_pore` โดย `normal_skin` ยังไม่อยู่ในการฝึก detection ปัจจุบัน

## 1. Clone Project

```powershell
git clone https://github.com/Bluedabade/ML-Assignment.git
cd ML-Assignment
```

คำสั่งทั้งหมดด้านล่างให้รันจากโฟลเดอร์หลักของ repository (`ML-Assignment`)

## 2. Create Python Virtual Environment

โปรเจกต์ทดสอบด้วย **Python 3.11** ตรวจเวอร์ชันด้วย `python --version` แล้วสร้าง environment แยกสำหรับ dependencies ของโปรเจกต์

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

เมื่อเปิดใช้งานสำเร็จ terminal จะมี `(.venv)` อยู่หน้าบรรทัด ให้ activate ทุกครั้งที่เปิด terminal ใหม่

ตรวจว่าใช้ Python ใน `.venv` จริง:

```powershell
where.exe python
python -c "import sys; print(sys.executable)"
```

ผลแรกของ `where.exe python` และค่า `sys.executable` ควรเป็น path เช่น `D:\Projects\ML-Assignment\.venv\Scripts\python.exe` (ขึ้นกับตำแหน่งที่ clone) หากไม่ได้ชี้เข้า `.venv` ให้ activate ก่อนติดตั้ง dependencies หรือเช็ก CUDA

หาก PowerShell บล็อกการ activate ให้รันคำสั่งนี้ แล้ว activate อีกครั้ง:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

รันใน terminal ที่ activate `.venv` แล้ว:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` ไม่ระบุ `torch`, `torchvision` หรือ `torchaudio` โดยตรง และไม่ได้กำหนด CUDA build จึงต้องตรวจ PyTorch ใน `.venv` ตามข้อถัดไปหลังติดตั้ง

## 4. GPU / CUDA Check

แนะนำให้ใช้ NVIDIA GPU สำหรับ training ตรวจว่าเครื่องเห็น GPU ด้วย:

การฝึก YOLO11n และ YOLO11s ที่เสร็จแล้วใช้ Python 3.11.9, PyTorch 2.14.0+cu130 และ NVIDIA GeForce GTX 1650 Ti โดย `CUDA: True` สมาชิกแต่ละคนต้องตรวจ installation ของเครื่องตัวเอง:

```powershell
nvidia-smi
```

จากนั้นตรวจว่า PyTorch ใน `.venv` ใช้ CUDA ได้:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

ตัวอย่างเมื่อพร้อมใช้ GPU:

```text
CUDA: True
GPU: NVIDIA ...
```

หาก `CUDA: False` ให้ยืนยันก่อนว่า `.venv` active และ `python` ชี้เข้า `.venv` ตามข้อ 2 หากยังเป็น False อาจใช้ PyTorch แบบ CPU-only ให้ติดตั้ง CUDA-enabled PyTorch build ที่เหมาะกับเครื่องตาม [คำแนะนำทางการของ PyTorch](https://pytorch.org/get-started/locally/) หรือถามเจ้าของโปรเจกต์ คำสั่ง training ด้านล่างใช้ `device=0` จึงต้องมี CUDA พร้อม

## 5. Dataset

YOLO dataset ที่เตรียมแล้วอยู่ที่ `datasets/processed/skin_detection/` และใช้ไฟล์ `datasets/processed/skin_detection/data.yaml`

หาก processed dataset มีครบหลัง clone **ไม่ต้องสร้างใหม่** และห้ามแก้ไฟล์ใน `datasets/raw/`

สร้างใหม่เฉพาะเมื่อต้องการ regeneration:

```powershell
python scripts/prepare_detection_dataset.py
```

ตรวจ dataset ก่อน training:

```powershell
python scripts/validate_dataset.py
```

ควรได้ `PASS` หากได้ `FAIL` ให้แก้สาเหตุหรือแจ้งเจ้าของโปรเจกต์ก่อนเริ่ม training

## 6. Optional: Check Bounding Boxes

```powershell
python scripts/visualize_annotations.py
```

เปิด preview images ใน `results/annotation_samples/` เพื่อตรวจตำแหน่ง bounding boxes ก่อน training

## 7. Smoke Test Before Full Training

แนะนำให้ทดสอบ YOLO11n แบบสั้นก่อน:

```powershell
yolo detect train model=yolo11n.pt data=datasets/processed/skin_detection/data.yaml epochs=2 imgsz=640 batch=2 device=0 workers=2 seed=42 project=results/smoke_test name=yolo11n_test
```

`epochs=2` คือทดสอบเพียง 2 รอบ และ `device=0` คือ NVIDIA GPU หมายเลข 0 หากเกิด `CUDA out of memory` ให้ลอง `batch=1`

## 8. Train YOLO11n

ใช้ settings เดียวกับการทดลองของโปรเจกต์:

คำสั่งนี้อ้างอิง `args.yaml` ของ run ที่เสร็จแล้ว ไม่ใช่ค่าปัจจุบันใน `configs/` ซึ่งมี epochs/batch ต่างกัน

```powershell
yolo detect train model=yolo11n.pt data=datasets/processed/skin_detection/data.yaml epochs=30 imgsz=640 batch=4 device=0 workers=2 optimizer=AdamW lr0=0.001 lrf=0.01 momentum=0.937 weight_decay=0.0005 mosaic=1.0 mixup=0.2 degrees=10.0 fliplr=0.5 patience=15 seed=42 project=results/training name=yolo11n
```

Training ใช้เวลาตามความเร็ว GPU หากเกิด `CUDA out of memory` ให้เปลี่ยน `batch=4` เป็น `batch=2` หรือ `batch=1`

## 9. Train YOLO11s

```powershell
yolo detect train model=yolo11s.pt data=datasets/processed/skin_detection/data.yaml epochs=30 imgsz=640 batch=4 device=0 workers=2 optimizer=AdamW lr0=0.001 lrf=0.01 momentum=0.937 weight_decay=0.0005 mosaic=1.0 mixup=0.2 degrees=10.0 fliplr=0.5 patience=15 seed=42 project=results/training name=yolo11s
```

YOLO11s ใหญ่กว่า YOLO11n จึงใช้ GPU memory และเวลามากกว่า หากเกิด CUDA OOM ให้ลด batch size เป็น 2 หรือ 1

## 10. Training Output

เครื่องปัจจุบันใช้ Ultralytics 8.4.166 โดย `runs_dir` ชี้ไปที่โฟลเดอร์ `runs/` ของ repository เมื่อ `project=results/training` เป็น relative path ระบบจะต่อเป็น `runs/detect/results/training/` ผลทดลองที่เสร็จแล้วอยู่ที่:

```text
runs/detect/results/training/yolo11n/
runs/detect/results/training/yolo11s/
```

ไฟล์สำคัญในแต่ละโฟลเดอร์:

- `weights/best.pt`: checkpoint ที่เลือกจาก validation performance
- `weights/last.pt`: checkpoint จาก epoch สุดท้าย
- `results.csv`: ผลแต่ละ epoch
- `results.png`: กราฟ training
- `confusion_matrix.png`: confusion matrix

ให้ดูบรรทัด `Results saved to ...` หลังรัน เพื่อดูตำแหน่ง output จริง เพราะ path อาจต่างกันตาม Ultralytics version/settings และอาจเติมเลขท้ายชื่อเมื่อโฟลเดอร์เดิมมีอยู่แล้ว ให้แทน checkpoint path ในคำสั่งด้านล่างด้วย path ของ run ที่ต้องการ

## 11. Test the Trained Model

YOLO11n:

```powershell
yolo detect val model="runs/detect/results/training/yolo11n/weights/best.pt" data="datasets/processed/skin_detection/data.yaml" split=test imgsz=640 batch=4 device=0 workers=2 project=results/evaluation name=yolo11n_test
```

YOLO11s:

```powershell
yolo detect val model="runs/detect/results/training/yolo11s/weights/best.pt" data="datasets/processed/skin_detection/data.yaml" split=test imgsz=640 batch=4 device=0 workers=2 project=results/evaluation name=yolo11s_test
```

ผลที่แสดงมี Precision, Recall, mAP50 และ mAP50-95 ใช้ Test set หลังเลือกโมเดลจาก Validation แล้ว

## 12. Predict Images

ตัวอย่าง YOLO11n:

```powershell
yolo detect predict model="runs/detect/results/training/yolo11n/weights/best.pt" source="datasets/processed/skin_detection/images/test" conf=0.25 imgsz=640 device=0 project=results/predictions name=yolo11n_test
```

คำสั่งนี้สร้างภาพพร้อม predicted bounding boxes

ผลปัจจุบันยังเป็น Draft: คลาสไม่สมดุลมาก, dark_spot มี objects น้อย, wrinkle ยังทำได้ไม่ดี และมี negative images จำนวนมากจากการกรองคลาสที่ไม่ใช้ รายละเอียดอยู่ใน `docs/DATA_PREPARATION_REPORT.md`

## 13. Quick Start

สำหรับเครื่องที่มี Python 3.11 และ CUDA พร้อมแล้ว ให้ตรวจ PyTorch CUDA ใน `.venv` ตามข้อ 4 ก่อน training:

```powershell
git clone https://github.com/Bluedabade/ML-Assignment.git
cd ML-Assignment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/validate_dataset.py
yolo detect train model=yolo11n.pt data=datasets/processed/skin_detection/data.yaml epochs=30 imgsz=640 batch=4 device=0 workers=2 optimizer=AdamW lr0=0.001 lrf=0.01 momentum=0.937 weight_decay=0.0005 mosaic=1.0 mixup=0.2 degrees=10.0 fliplr=0.5 patience=15 seed=42 project=results/training name=yolo11n
```
