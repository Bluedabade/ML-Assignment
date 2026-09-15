# ML Assignment — Facial Skin Classification

คู่มือนี้อธิบายวิธีติดตั้ง Environment และ Train โมเดลด้วย NVIDIA GPU บน Windows + WSL2

Repository:

```text
https://github.com/Bluedabade/ML-Assignment.git
```

---

# 1. สิ่งที่ต้องติดตั้งก่อน

เครื่องที่ใช้ Train แนะนำให้มี:

- Windows 10 หรือ Windows 11
- NVIDIA GPU
- NVIDIA Driver
- WSL2
- Ubuntu
- Git
- Python 3.11

---

# 2. ติดตั้ง WSL2

เปิด **PowerShell แบบ Run as Administrator**

แล้วรัน:

```powershell
wsl --install -d Ubuntu
```

เมื่อติดตั้งเสร็จให้ Restart เครื่อง

จากนั้นเปิด:

```text
Ubuntu
```

จาก Start Menu

ครั้งแรก Ubuntu จะให้ตั้ง:

```text
Username
Password
```

> เวลาพิมพ์ Password ใน Terminal จะไม่แสดงตัวอักษร ถือว่าเป็นปกติ

---

# 3. ตรวจว่า WSL เห็น NVIDIA GPU

เปิด Ubuntu แล้วรัน:

```bash
nvidia-smi
```

ถ้าสำเร็จควรเห็นข้อมูลประมาณ:

```text
NVIDIA-SMI
Driver Version
CUDA Version
GPU Name
GPU Memory
```

ถ้าขึ้นข้อมูล GPU แสดงว่า WSL สามารถมองเห็นการ์ดจอได้แล้ว

---

# 4. Clone Project

ใน Ubuntu:

```bash
git clone https://github.com/Bluedabade/ML-Assignment.git
```

เข้าโฟลเดอร์:

```bash
cd ML-Assignment
```

ตรวจไฟล์:

```bash
ls
```

---

# 5. สร้าง Python Environment

ตรวจ Python:

```bash
python3 --version
```

แนะนำ:

```text
Python 3.11
```

สร้าง Virtual Environment:

```bash
python3 -m venv .venv
```

เปิด Environment:

```bash
source .venv/bin/activate
```

ถ้าสำเร็จจะเห็นประมาณ:

```text
(.venv) user@computer:~/ML-Assignment$
```

---

# 6. ติดตั้ง Library

อัปเดต pip:

```bash
python -m pip install --upgrade pip
```

ติดตั้ง Library ของ Project:

```bash
pip install -r requirements.txt
```

รอจนติดตั้งเสร็จ

---

# 7. ตรวจ TensorFlow และ GPU

รัน:

```bash
python scripts/check_gpu.py
```

ควรเห็นว่า TensorFlow พบ GPU เช่น:

```text
TensorFlow version: 2.16.1

GPU:
PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
```

หรือมีข้อความประมาณ:

```text
CUDA availability: True
```

ถ้าพบ GPU แสดงว่าพร้อม Train แล้ว

สามารถตรวจอีกครั้งด้วย:

```bash
nvidia-smi
```

---

# 8. ตรวจ Dataset

Dataset ที่ใช้ Train ถูกเตรียมไว้ใน Repository แล้ว

ก่อน Train ให้รัน:

```bash
python scripts/check_dataset.py
```

ระบบจะตรวจ:

- Training Set
- Validation Set
- Test Set
- จำนวนภาพแต่ละ Class
- รูปเสีย
- Duplicate
- Class ที่ขาดหาย

Dataset มี 5 Classes:

```text
normal
wrinkle
acne
dark_spot
large_pore
```

หาก Dataset ผ่านการตรวจจึงค่อยเริ่ม Train

---

# 9. ไม่ต้อง Prepare Dataset ใหม่ทุกครั้ง

สำหรับการ Train ปกติ **ไม่ต้องรัน**

```bash
python scripts/prepare_dataset.py --clean
```

เพราะ Dataset ใน:

```text
dataset/prepared/
```

ถูกเตรียมไว้แล้ว

ใช้เพียง:

```bash
python scripts/check_dataset.py
python train.py
```

คำสั่ง:

```bash
python scripts/prepare_dataset.py --clean
```

ใช้เฉพาะเมื่อต้องการสร้าง Prepared Dataset ใหม่จาก Raw Dataset เท่านั้น

---

# 10. เริ่ม Training

เมื่อ:

```text
GPU พร้อม
Dataset พร้อม
Environment พร้อม
```

ให้รัน:

```bash
python train.py
```

จากนั้นรอจน Training เสร็จ

ระหว่าง Train สามารถเปิด Terminal อีกหน้าหนึ่งแล้วใช้:

```bash
nvidia-smi
```

เพื่อตรวจดูการใช้งาน GPU

---

# 11. Model ที่ Train แล้ว

เมื่อ Training สำเร็จ Model ที่ดีที่สุดจะถูกเก็บไว้ที่:

```text
models/mobilenetv2_5class_best.keras
```

Class index จะอยู่ที่:

```text
models/class_indices.json
```

---

# 12. ผลการ Training

ผลการทดลองจะถูกเก็บใน:

```text
results/
```

และแต่ละการทดลองจะถูกแยกประมาณ:

```text
results/runs/YYYYMMDD_HHMMSS/
```

ภายในจะมีผล เช่น:

```text
Accuracy
Validation Accuracy

Loss
Validation Loss

Precision
Recall
F1-score

Classification Report
Confusion Matrix
```

รวมถึงกราฟสำหรับเปรียบเทียบ Train และ Validation

---

# 13. ทดลองภาพใหม่

หลังจากมี Model แล้ว สามารถทดลองภาพใหม่ได้ด้วย:

```bash
python scripts/predict.py path/to/image.jpg
```

ตัวอย่าง:

```bash
python scripts/predict.py sample.jpg
```

ถ้า path มีช่องว่างให้ใส่ `" "` เช่น:

```bash
python scripts/predict.py "/home/user/My Images/face.jpg"
```

ระบบจะแสดงคะแนนทั้ง 5 Classes เช่น:

```text
wrinkle       82.41%
dark_spot      7.91%
acne           4.10%
normal         3.20%
large_pore     2.38%

Prediction: wrinkle
Confidence: 82.41%
```

---

# 14. ปรับค่าการ Training

ค่าหลักสามารถแก้ได้ที่:

```text
src/config.py
```

ตัวอย่างค่าที่สามารถทดลองได้:

```python
BATCH_SIZE = 32
```

หรือ:

```python
BATCH_SIZE = 64
```

จำนวน Epoch:

```python
HEAD_EPOCHS = 10
```

Dense Layer:

```python
HEAD_DENSE_UNITS = []
```

หมายถึง:

```text
MobileNetV2
↓
GlobalAveragePooling2D
↓
Dense(5)
```

เพิ่ม Dense 128:

```python
HEAD_DENSE_UNITS = [128]
```

จะเป็น:

```text
MobileNetV2
↓
GlobalAveragePooling2D
↓
Dense(128)
↓
Dense(5)
```

เพิ่ม 2 Hidden Layers:

```python
HEAD_DENSE_UNITS = [256, 128]
```

จะเป็น:

```text
MobileNetV2
↓
GlobalAveragePooling2D
↓
Dense(256)
↓
Dense(128)
↓
Dense(5)
```

Dropout:

```python
DROPOUT_RATES = [0.3]
```

เปิด Fine-tuning:

```python
ENABLE_FINE_TUNING = True
```

ปิด Fine-tuning:

```python
ENABLE_FINE_TUNING = False
```

กำหนดตำแหน่ง Fine-tuning:

```python
FINE_TUNE_AT = 100
```

หมายถึง Freeze ช่วงต้นประมาณ 100 Layers และเปิดให้ Layer หลังจากนั้นสามารถ Train เพิ่มได้

---

# 15. การทดลองที่แนะนำ

แนะนำให้เปลี่ยนค่าทีละอย่าง เพื่อให้สามารถเปรียบเทียบผลได้ง่าย

ตัวอย่าง:

### Experiment 1

```python
BATCH_SIZE = 32
HEAD_DENSE_UNITS = [128]
ENABLE_FINE_TUNING = False
```

Train:

```bash
python train.py
```

### Experiment 2

เปลี่ยน:

```python
BATCH_SIZE = 64
```

แล้ว Train ใหม่:

```bash
python train.py
```

### Experiment 3

เปลี่ยน Head:

```python
HEAD_DENSE_UNITS = [256, 128]
```

แล้ว Train ใหม่

### Experiment 4

เปิด Fine-tuning:

```python
ENABLE_FINE_TUNING = True
FINE_TUNE_AT = 100
```

แล้ว Train ใหม่

ผลแต่ละ Run จะถูกเก็บแยกใน:

```text
results/runs/
```

ทำให้สามารถนำ Accuracy, Loss, Precision, Recall และ F1-score มาเปรียบเทียบกันได้

---

# 16. หลังจากแก้ Project ให้ Push ขึ้น GitHub

ก่อนแก้ไฟล์แนะนำให้ดึง Version ล่าสุดก่อน:

```bash
git pull --rebase origin main
```

ดูไฟล์ที่เปลี่ยน:

```bash
git status
```

ตรวจไฟล์ก่อน Push:

```bash
python scripts/check_git_files.py
```

เพิ่มไฟล์:

```bash
git add .
```

Commit:

```bash
git commit -m "Update experiment"
```

Push:

```bash
git push origin main
```

หลัง Push สามารถตรวจได้ด้วย:

```bash
git status
```

ควรขึ้นประมาณ:

```text
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

---

# 17. เมื่อเพื่อนมีการ Push งานใหม่

เครื่องอีกคนให้ดึง Version ล่าสุดด้วย:

```bash
git pull --rebase origin main
```

ก่อนเริ่มแก้ไขหรือ Train ต่อ

---

# 18. ปิด Python Environment

เมื่อใช้งานเสร็จ:

```bash
deactivate
```

ถ้าจะกลับมาใช้งานครั้งต่อไป:

```bash
cd ML-Assignment
source .venv/bin/activate
```

ไม่ต้องสร้าง `.venv` ใหม่ทุกครั้ง

---

# Troubleshooting

## TensorFlow ไม่พบ GPU

ลอง:

```bash
nvidia-smi
```

ถ้า Ubuntu ไม่เห็น GPU ให้ตรวจ:

- NVIDIA Driver ใน Windows
- WSL2
- Windows Update
- Restart เครื่อง

จากนั้นลอง:

```bash
python scripts/check_gpu.py
```

อีกครั้ง

---

## No module named ...

ตรวจว่าเปิด Virtual Environment แล้ว:

```bash
source .venv/bin/activate
```

จากนั้น:

```bash
pip install -r requirements.txt
```

---

## GPU Memory ไม่พอ / Out of Memory

เปิด:

```text
src/config.py
```

ลด:

```python
BATCH_SIZE = 32
```

ถ้ายังไม่พอ:

```python
BATCH_SIZE = 16
```

แล้ว Train ใหม่

---

## Predict ไม่ได้เพราะไม่มี Model

ต้อง Train ให้สำเร็จก่อน:

```bash
python train.py
```

แล้วตรวจว่ามี:

```text
models/mobilenetv2_5class_best.keras
```

จากนั้นจึงใช้:

```bash
python scripts/predict.py sample.jpg
```

---

# คำสั่งสรุปสำหรับคนที่ Setup สำเร็จแล้ว

ครั้งแรก:

```bash
git clone https://github.com/Bluedabade/ML-Assignment.git
cd ML-Assignment

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/check_gpu.py
python scripts/check_dataset.py

python train.py
```

ครั้งต่อไป:

```bash
cd ML-Assignment
git pull --rebase origin main
source .venv/bin/activate
python scripts/check_gpu.py
python train.py
```
