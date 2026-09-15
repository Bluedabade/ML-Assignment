# AI Facial Skin Analysis — MobileNetV2 (5 Classes)

โปรเจกต์ Assignment นี้รับภาพหนึ่งภาพแล้วจำแนกเพียง **หนึ่ง class** จาก `normal`, `wrinkle`, `acne`, `dark_spot`, `large_pore` ด้วย TensorFlow/Keras และ MobileNetV2 Transfer Learning ตาม Chapter 7

ผล Softmax ทั้ง 5 ค่าแข่งขันกันและรวมประมาณ 100% ค่าที่สูงสุดคือ prediction ของภาพ ไม่ได้แปลว่าคนหนึ่งมีสิว 30% และริ้วรอย 40% พร้อมกัน และ **confidence score ไม่ใช่เปอร์เซ็นต์พื้นที่ผิว ไม่ใช่การวินิจฉัยทางการแพทย์**

## โครงสร้างสำคัญ

```text
dataset/raw/                 ข้อมูลต้นฉบับและ metadata/license
dataset/prepared/            training_set, val_set, test_set (70/15/15)
dataset/unmapped/            คำอธิบายข้อมูลที่ไม่ map (ไฟล์จริงคงอยู่ใน raw)
src/config.py                ค่าทดลองทั้งหมด
src/data_loader.py           ImageDataGenerator
src/model_builder.py         MobileNetV2 และ configurable head
scripts/prepare_dataset.py   audit, SHA256 deduplicate และ stratified split
scripts/check_dataset.py     ตรวจความพร้อมข้อมูล
scripts/predict.py           ทำนาย probability 5 classes
models/ และ results/         model, report, history และ plots
```

## ติดตั้งบน Windows + WSL2 + NVIDIA GPU

WSL2 คือ Linux environment ที่ทำงานบน Windows และเป็นวิธีแนะนำสำหรับ TensorFlow NVIDIA GPU เปิด PowerShell แบบ Administrator แล้วรัน:

```powershell
wsl --install -d Ubuntu
```

Restart แล้วเปิด **Ubuntu** จาก Start menu ตั้ง username/password จากนั้นตรวจ GPU ด้วย `nvidia-smi` (ติดตั้ง NVIDIA Windows Driver รุ่นล่าสุดก่อน ไม่ต้องติดตั้ง Linux display driver เอง)

ใน Ubuntu:

```bash
git clone https://github.com/Bluedabade/ML-Assignment.git
cd ML-Assignment
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_gpu.py
```

แนะนำ Python 3.11 หากสำเร็จควรเห็น `CUDA availability: True` และรายการ GPU ออกจาก environment ใช้ `deactivate`

## Dataset และ Training

Dataset ใน repository เตรียมไว้แล้ว ตรวจหรือสร้างใหม่ได้ดังนี้:

```bash
python scripts/check_dataset.py
python scripts/prepare_dataset.py --clean
python train.py
```

prepare จะ map เฉพาะ label ที่ชัดเจน ตรวจภาพเสีย/duplicate ด้วย SHA256 ก่อนสุ่ม seed 42 และแบ่งต่อ class เป็น Training 70%, Validation 15%, Testing 15% ถ้าไม่ครบ 5 classes การ train จะหยุด Model ที่ดีที่สุดอยู่ใน `models/mobilenetv2_5class_best.keras`; metrics และกราฟอยู่ใน `results/`

## ทำนายภาพใหม่

```bash
python scripts/predict.py sample.jpg
```

script อ่าน `models/class_indices.json` ไม่ assume index เอง แสดง probability เรียงจากมากไปน้อย, prediction, confidence และผลรวม

## ทดลองโดยแก้ไฟล์เดียว

เปิด `src/config.py` แล้วเปลี่ยน เช่น `BATCH_SIZE=128` เป็น `32` หรือ `64`; `HEAD_EPOCHS`; `LEARNING_RATE`; และ head:

```python
HEAD_DENSE_UNITS=[]          # GlobalAveragePooling2D -> Dense(5, softmax)
HEAD_DENSE_UNITS=[128]       # เพิ่ม Dense 128
HEAD_DENSE_UNITS=[256,128]   # เพิ่ม Dense 256 และ 128
DROPOUT_RATES=[0.3]          # ใช้ dropout เท่ากันทุก hidden layer
ENABLE_FINE_TUNING=True      # เปลี่ยน False เพื่อปิด
FINE_TUNE_AT=100             # freeze layer ช่วงต้น 100 layers
FINE_TUNE_EPOCHS=5
```

แต่ละ run เก็บ snapshot config/history/metrics/plots แยกใน `results/runs/YYYYMMDD_HHMMSS/`

## Git สำหรับสมาชิกกลุ่ม

```bash
git pull --rebase origin main
git status
python scripts/check_git_files.py
git add .
git commit -m "Describe experiment"
git push origin main
```

ใช้ Git ปกติ ไม่ใช้ LFS และห้าม commit ZIP หรือไฟล์เดี่ยวเกิน 100 MB ก่อน push ตรวจว่าไม่มี token/password และอ่าน `DATA_SOURCES.md`

## Troubleshooting

- `ไม่พบ GPU`: รัน `nvidia-smi` ใน Windows และ Ubuntu, อัปเดต Windows/WSL/NVIDIA driver แล้วสร้าง venv ใหม่
- `No module named ...`: activate `.venv` แล้ว `pip install -r requirements.txt`
- Dataset ไม่ครบ/เสีย/ซ้ำ: รัน checker แล้วอ่าน `results/reports/dataset_report.txt` และ `excluded_images.csv`
- Out of memory: ลด `BATCH_SIZE` เป็น 32 หรือ 16
- ไม่มี model: ต้อง train สำเร็จก่อนจึงใช้ predict ได้
- path รูปผิด: ใช้ path ที่มีอยู่จริงและครอบด้วย quote ถ้ามีช่องว่าง
