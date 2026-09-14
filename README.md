# วิธี Setup และ Train ด้วย NVIDIA GPU ผ่าน WSL2

คู่มือนี้สำหรับผู้ใช้ Windows ที่ต้องการ Train ด้วย NVIDIA GPU ผ่าน WSL2 และ Ubuntu

สิ่งสำคัญก่อนเริ่ม:

- Dataset สำหรับ Train อยู่ใน Repository นี้แล้ว
- ไม่ต้องดาวน์โหลด Dataset จาก Kaggle หรือ Roboflow
- ไม่ต้องรัน `scripts/prepare_dataset.py` ก่อน Train ตามปกติ
- การติดตั้ง WSL2 และ Environment ทำเพียงครั้งแรก
- ครั้งต่อไปใช้ขั้นตอนสั้น ๆ ในหัวข้อที่ 20 ได้เลย
- TensorFlow 2.16.1 ไม่รองรับ NVIDIA GPU ผ่าน Python บน Windows โดยตรง จึงต้องใช้ `Windows → WSL2 → Ubuntu → TensorFlow → NVIDIA GPU`
- ไม่ต้องติดตั้ง CUDA Toolkit หรือ cuDNN แบบ system-wide และไม่ต้องติดตั้ง Linux NVIDIA display driver ใน Ubuntu

> ถ้าต้องการวิธี GPU ที่ง่ายกว่า WSL2 ให้ใช้ [Google Colab Notebook](https://colab.research.google.com/github/Bluedabade/ML-Assignment/blob/main/notebooks/train_on_colab.ipynb)

## 1. ตรวจสอบ NVIDIA GPU บน Windows

เปิด **Windows PowerShell**: กดปุ่ม Start พิมพ์ `PowerShell` แล้วเปิดโปรแกรม

รันคำสั่ง:

```powershell
nvidia-smi
```

คำสั่งนี้ใช้ตรวจสอบ NVIDIA GPU และ Driver ใน Windows ถ้าพร้อมใช้งาน จะเห็นตารางที่มีชื่อ GPU เช่น:

```text
NVIDIA GeForce GTX 1650 Ti
```

ไม่จำเป็นต้องเป็นรุ่นนี้ NVIDIA GPU รุ่นอื่นที่รองรับ WSL2 ก็ใช้ได้

ถ้า PowerShell แจ้งว่าไม่รู้จัก `nvidia-smi` หรือไม่แสดง NVIDIA GPU ให้ติดตั้งหรืออัปเดต **NVIDIA Windows Driver** จากเว็บไซต์ NVIDIA ก่อน แล้ว Restart Windows อย่าติดตั้ง CUDA Toolkit ในขั้นตอนนี้

## 2. ตรวจสอบว่า WSL มีอยู่แล้วหรือยัง

ยังคงรันใน **Windows PowerShell**:

```powershell
wsl --status
wsl -l -v
```

คำสั่งแรกแสดงสถานะ WSL ส่วนคำสั่งที่สองแสดง Linux distribution และ VERSION ตัวอย่าง:

```text
  NAME      STATE     VERSION
* Ubuntu   Stopped   2
```

ถ้ามี `Ubuntu` และคอลัมน์ `VERSION` เป็น `2` อยู่แล้ว ไม่ต้องติดตั้งใหม่ ให้ข้ามไปหัวข้อที่ 5

ถ้าไม่มี Ubuntu ให้ทำหัวข้อที่ 3 ถ้ามี Ubuntu แต่ VERSION เป็น `1` ให้ทำหัวข้อที่ 4

## 3. ติดตั้ง WSL2 สำหรับเครื่องที่ยังไม่มี

ปิด PowerShell เดิม จากนั้นกด Start พิมพ์ `PowerShell` คลิกขวา แล้วเลือก **Run as Administrator**

รัน:

```powershell
wsl --install
```

คำสั่งนี้จะเปิดส่วนประกอบ WSL, ติดตั้ง WSL2 และโดยปกติจะติดตั้ง Ubuntu ให้ด้วย Windows อาจขอให้ Restart เครื่อง

ถ้าคำสั่งแสดงเพียงหน้าช่วยเหลือหรือไม่ได้ติดตั้ง Ubuntu ให้รัน:

```powershell
wsl --install -d Ubuntu
```

หลังติดตั้งและ Restart แล้ว ให้เปิด **Ubuntu** จาก Start Menu ครั้งแรก Ubuntu จะให้สร้าง:

- Linux username: ชื่อผู้ใช้ภายใน Ubuntu ตั้งเองได้
- Linux password: รหัสผ่านสำหรับคำสั่ง `sudo`

ขณะพิมพ์ Linux password หน้าจอจะ **ไม่แสดงตัวอักษร จุด หรือดอกจัน** เป็นพฤติกรรมปกติ ให้พิมพ์รหัสผ่านแล้วกด Enter

## 4. ตรวจสอบว่า Ubuntu ใช้ WSL2

กลับไปเปิด **Windows PowerShell** แล้วรัน:

```powershell
wsl -l -v
```

ถ้า Ubuntu แสดง VERSION `1` ให้เปลี่ยนเป็น WSL2:

```powershell
wsl --set-version Ubuntu 2
```

รอจนเสร็จ แล้วตรวจอีกครั้ง:

```powershell
wsl -l -v
```

ต้องเห็น `Ubuntu` เป็น VERSION `2` ก่อนทำต่อ

## 5. อัปเดต WSL

รันใน **Windows PowerShell**:

```powershell
wsl --update
wsl --shutdown
```

`wsl --update` อัปเดต WSL และ Kernel ส่วน `wsl --shutdown` ปิด WSL ทั้งหมดเพื่อให้เริ่มใหม่ หลังจากนั้นเปิด Ubuntu จาก Start Menu อีกครั้ง

## 6. ตรวจสอบ GPU ภายใน Ubuntu

**ตั้งแต่หัวข้อนี้เป็นต้นไป ให้รันคำสั่งใน Ubuntu / WSL ไม่ใช่ PowerShell เว้นแต่จะระบุเป็นอย่างอื่น**

เปิด Ubuntu แล้วรัน:

```bash
nvidia-smi
```

ต้องเห็นข้อมูล NVIDIA GPU ภายใน Ubuntu เช่นเดียวกับ Windows แสดงว่า WSL2 เข้าถึง GPU ได้

ห้ามติดตั้ง Linux NVIDIA display driver ภายใน WSL เพราะ Driver ฝั่ง Windows เป็นตัวส่ง GPU ให้ WSL2 อยู่แล้ว

ถ้า `nvidia-smi` ใช้ได้ใน Windows แต่ใช้ไม่ได้ใน Ubuntu:

1. ปิด Ubuntu
2. เปิด Windows PowerShell
3. รัน:

```powershell
wsl --update
wsl --shutdown
```

4. เปิด Ubuntu ใหม่
5. รัน `nvidia-smi` ใน Ubuntu อีกครั้ง

ถ้ายังไม่ได้ ให้อัปเดต NVIDIA Windows Driver, Windows และ WSL แล้ว Restart เครื่อง อย่าทดลองติดตั้ง CUDA/cuDNN หลายเวอร์ชันแบบสุ่ม

## 7. ติดตั้ง Git และ Python ใน Ubuntu

รันใน **Ubuntu**:

```bash
sudo apt update
sudo apt install -y git python3 python3-pip python3-venv
```

- `git` ใช้ดาวน์โหลดและอัปเดต Repository
- `python3`, `python3-pip` และ `python3-venv` ใช้สร้าง Environment และติดตั้ง Library

ตรวจสอบว่าติดตั้งสำเร็จ:

```bash
git --version
python3 --version
```

ควรเห็นเลขเวอร์ชันของ Git และ Python โดยแนะนำ Python 3.11 สำหรับ dependency ชุดนี้

## 8. Clone Repository

รันใน **Ubuntu**:

```bash
cd ~
git clone https://github.com/Bluedabade/ML-Assignment.git
cd ML-Assignment
```

- `cd ~` คือไปที่ Home folder ของ Ubuntu
- `git clone` คือดาวน์โหลด Repository จาก GitHub
- `cd ML-Assignment` คือเข้าโฟลเดอร์โปรเจกต์

แนะนำให้เก็บโปรเจกต์ที่ `~/ML-Assignment` ไม่ใช่ `/mnt/c/...` เพื่อให้การอ่าน Dataset และ Train เร็วกว่าใน WSL2

Dataset พร้อมใช้งานอยู่ที่ `data/processed/` ไม่ต้องดาวน์โหลดเพิ่ม ไม่ต้องใช้ Kaggle/Roboflow และไม่ต้องเตรียม Dataset ใหม่

## 9. สร้าง Python Virtual Environment

ตรวจสอบก่อนว่าอยู่ใน `~/ML-Assignment` แล้วรันใน **Ubuntu**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

เมื่อสำเร็จ ด้านหน้าบรรทัดคำสั่งมักแสดง `(.venv)` ตัวอย่าง:

```text
(.venv) username@computer:~/ML-Assignment$
```

จากนั้นอัปเดต pip:

```bash
python -m pip install --upgrade pip
```

ทุกครั้งก่อนติดตั้ง Library หรือ Train ต้องเห็นว่า `.venv` เปิดใช้งานอยู่

## 10. ติดตั้ง Library สำหรับ GPU

รันใน **Ubuntu ขณะที่ `.venv` เปิดใช้งานอยู่**:

```bash
pip install -r requirements-gpu.txt
```

ไฟล์นี้ติดตั้ง `tensorflow[and-cuda]==2.16.1` พร้อม NVIDIA CUDA Python packages ที่เข้ากันได้ จึงไม่ต้องติดตั้ง CUDA Toolkit หรือ cuDNN แบบ system-wide

การติดตั้งครั้งแรกอาจใช้เวลาหลายนาทีและใช้พื้นที่ค่อนข้างมาก เพราะ TensorFlow และ CUDA libraries มีขนาดใหญ่ รอจนคำสั่งเสร็จและกลับมาที่ prompt โดยไม่มี Error

อย่ารัน `pip install tensorflow` หรือเพิ่ม TensorFlow รุ่นอื่นซ้ำใน `.venv` เดียวกัน

## 11. ตรวจสอบว่า TensorFlow เห็น GPU

รันใน **Ubuntu ขณะที่ `.venv` เปิดใช้งานอยู่**:

```bash
python check_environment.py
```

ผลสำเร็จควรมีข้อความใกล้เคียง:

```text
TensorFlow: 2.16.1
GPU detected: YES
GPU device 0: NVIDIA ...
Training device: GPU
```

ตรวจสอบกับ TensorFlow โดยตรงอีกครั้ง:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

ถ้าพร้อม ผลควรคล้าย:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

ถ้าได้ `[]` แปลว่า TensorFlow ยังไม่เห็น GPU **อย่าเริ่ม Train แบบยาว** ให้ไปหัวข้อที่ 19

## 12. Smoke Test ก่อน Train จริง

รันใน **Ubuntu ขณะที่ `.venv` เปิดใช้งานอยู่**:

```bash
python train.py \
  --model mobilenetv2 \
  --smoke-test \
  --device gpu
```

หรือแบบบรรทัดเดียว:

```bash
python train.py --model mobilenetv2 --smoke-test --device gpu
```

Smoke Test จะโหลด Dataset, สร้าง Model และ Train เพียงสั้น ๆ เพื่อตรวจสอบ pipeline ผลจาก Smoke Test ไม่ใช่ผลลัพธ์สุดท้าย และไม่ควรนำไปเปรียบเทียบ Model

เมื่อสำเร็จจะเห็นข้อความ `Smoke test passed` จึงค่อยเริ่ม Train จริง

## 13. Train MobileNetV2

สำหรับ NVIDIA GPU ที่มี VRAM น้อย เช่น 4 GB แนะนำให้เริ่มด้วย batch size 16

คำสั่งหลายบรรทัดสำหรับ **Ubuntu**:

```bash
python train.py \
  --model mobilenetv2 \
  --epochs 10 \
  --batch-size 16 \
  --device gpu
```

คำสั่งบรรทัดเดียวสำหรับ copy/paste:

```bash
python train.py --model mobilenetv2 --epochs 10 --batch-size 16 --device gpu
```

- `--model mobilenetv2` เลือก Model
- `--epochs 10` Train ขั้นแรก 10 รอบ
- `--batch-size 16` ประมวลผลรูปครั้งละ 16 รูป
- `--device gpu` บังคับให้ใช้ GPU และหยุดพร้อม Error ถ้า TensorFlow ไม่เห็น GPU

ถ้าไม่ใส่ `--device` โปรแกรมใช้ค่า `auto`: เลือก GPU เมื่อพบ GPU มิฉะนั้นใช้ CPU

## 14. Train EfficientNetB0

รันใน **Ubuntu ขณะที่ `.venv` เปิดใช้งานอยู่**:

```bash
python train.py --model efficientnetb0 --epochs 10 --batch-size 16 --device gpu
```

ถ้า GPU memory ไม่พอ ให้ลดเป็น `--batch-size 8` หรือ `--batch-size 4`

## 15. Train ResNet50

ResNet50 ใช้ memory มากกว่า แนะนำให้ GPU VRAM 4 GB เริ่มที่ batch size 8:

```bash
python train.py --model resnet50 --epochs 10 --batch-size 8 --device gpu
```

ถ้า memory ไม่พอ ให้ลอง:

```bash
python train.py --model resnet50 --epochs 10 --batch-size 4 --device gpu
```

## 16. ไม่ต้อง Train ทั้งสามตัวพร้อมกัน

อย่าเปิดสาม Terminal แล้ว Train ทั้งสาม Model พร้อมกันบน GPU เดียว เพราะ GPU memory อาจไม่พอและแต่ละงานจะช้าลง

ให้ Train ทีละ Model ตามลำดับ:

```text
MobileNetV2
↓
EfficientNetB0
↓
ResNet50
```

รอให้คำสั่งหนึ่งเสร็จก่อนเริ่มคำสั่งถัดไป โปรแกรมจะบันทึก Model, training history, accuracy/loss plots และ evaluation outputs ให้อัตโนมัติ อย่านำ Metrics ที่ยัง Train ไม่เสร็จหรือ Smoke Test ไปอ้างเป็นผลสุดท้าย

## 17. ดูว่า GPU ถูกใช้งานจริงหรือไม่

ขณะกำลัง Train ให้เปิด **Windows PowerShell อีกหนึ่งหน้าต่าง** แล้วรัน:

```powershell
nvidia-smi -l 1
```

คำสั่งนี้ refresh สถานะ GPU ประมาณทุก 1 วินาที ระหว่าง Train โดยปกติจะเห็น GPU Memory Usage และ GPU Utilization เพิ่มขึ้น และอาจเห็นงานจาก WSL/Python ไม่จำเป็นต้องขึ้น 100% ตลอดเวลา

กด `Ctrl+C` เพื่อหยุดการ refresh

หรือเปิด Ubuntu อีกหนึ่งหน้าต่างแล้วรัน:

```bash
watch -n 1 nvidia-smi
```

กด `Ctrl+C` เพื่อหยุด

## 18. แก้ปัญหา CUDA Out of Memory

ข้อความต่อไปนี้มักหมายถึง GPU memory ไม่พอ:

- `ResourceExhaustedError`
- `OOM`
- `CUDA out of memory`

ไม่ใช่ว่าโปรเจกต์เสีย ให้ลด batch size ตามลำดับ:

```text
16 → 8 → 4
```

ตัวอย่าง:

```bash
python train.py --model mobilenetv2 --epochs 10 --batch-size 8 --device gpu
```

ถ้ายังไม่พอ:

```bash
python train.py --model mobilenetv2 --epochs 10 --batch-size 4 --device gpu
```

GPU ที่มี VRAM 4 GB เช่น GTX 1650 Ti อาจพบปัญหานี้ได้ โดยเฉพาะ ResNet50 โปรแกรมใช้ float32 เป็นค่าเริ่มต้นและไม่ได้เปิด mixed precision อัตโนมัติ

## 19. ถ้า TensorFlow ไม่เจอ GPU

ตรวจตามลำดับ อย่าข้ามขั้น:

1. เปิด **Windows PowerShell** แล้วรัน `nvidia-smi` — ต้องเห็น GPU
2. เปิด **Ubuntu** แล้วรัน `nvidia-smi` — ต้องเห็น GPU เช่นกัน
3. ใน Ubuntu เข้า `~/ML-Assignment` และตรวจว่า `.venv` เปิดอยู่ โดย prompt ควรมี `(.venv)`
4. รัน:

```bash
python check_environment.py
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

ถ้ายังเป็น `[]` ให้ปิด Ubuntu แล้วรันใน **Windows PowerShell**:

```powershell
wsl --update
wsl --shutdown
```

เปิด Ubuntu ใหม่แล้วตรวจอีกครั้ง

ถ้าจำเป็นต้องสร้าง Environment ใหม่ ให้รันคำสั่งต่อไปนี้ใน **Ubuntu** จากโฟลเดอร์ `~/ML-Assignment` คำสั่ง `rm -rf .venv` จะลบเฉพาะ Virtual Environment เดิม ไม่ลบ Dataset หรือ source code:

```bash
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-gpu.txt
python check_environment.py
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

ถ้า `deactivate` แจ้งว่าไม่รู้จักคำสั่ง หมายถึง Environment ไม่ได้เปิดอยู่ ให้เริ่มต่อจาก `rm -rf .venv`

ถ้ายังไม่พบ GPU ให้อัปเดต NVIDIA Windows Driver, Windows และ WSL อย่าติดตั้ง CUDA/cuDNN หรือ Linux NVIDIA Driver แบบสุ่ม

## 20. วิธีเปิดโปรเจกต์ครั้งต่อไป

WSL2, Ubuntu, Library และ `.venv` ติดตั้งเพียงครั้งแรก วันถัดไปไม่ต้องทำทุกหัวข้อใหม่

1. เปิด Ubuntu จาก Start Menu
2. copy/paste คำสั่งนี้:

```bash
cd ~/ML-Assignment
source .venv/bin/activate
git pull
python check_environment.py
python train.py --model mobilenetv2 --epochs 10 --batch-size 16 --device gpu
```

`git pull` ใช้อัปเดต Repository และข้ามได้ถ้าไม่ต้องการอัปเดต ส่วน `python check_environment.py` ใช้ตรวจ GPU ก่อน Train และข้ามได้เมื่อระบบเคยตรวจผ่านแล้ว

## 21. หยุด Virtual Environment

เมื่อทำงานเสร็จ ให้รันใน Ubuntu:

```bash
deactivate
```

ข้อความ `(.venv)` จะหายไป การปิด Ubuntu โดยไม่รันคำสั่งนี้ก็ไม่ทำให้ Dataset หรือผล Train หาย

## คำสั่ง CPU และ Google Colab ที่ยังใช้ได้

ถ้าไม่มี NVIDIA GPU สามารถใช้ CPU บน Windows ตามเดิม เปิด **Windows PowerShell** แล้วรัน:

```powershell
git clone https://github.com/Bluedabade/ML-Assignment.git
cd ML-Assignment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_environment.py
python train.py --model mobilenetv2 --epochs 10 --device cpu
```

ถ้า PowerShell บล็อกการเปิด `.venv` ให้รัน `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` แล้วลอง `.\.venv\Scripts\Activate.ps1` อีกครั้ง CPU Train ได้แต่จะช้ากว่า GPU

หรือใช้ Google Colab ซึ่งง่ายกว่าสำหรับผู้เริ่มต้น:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bluedabade/ML-Assignment/blob/main/notebooks/train_on_colab.ipynb)

เอกสารอ้างอิงการติดตั้งอย่างเป็นทางการ: [TensorFlow pip/WSL2](https://www.tensorflow.org/install/pip), [Microsoft WSL](https://learn.microsoft.com/windows/wsl/install), [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
