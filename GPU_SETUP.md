# การใช้ GPU บนเครื่องด้วย Windows และ WSL2

สำหรับผู้เริ่มต้น แนะนำให้ใช้ [Google Colab Notebook](notebooks/train_on_colab.ipynb) เพราะไม่ต้องตั้งค่า CUDA เอง เอกสารนี้เป็นทางเลือกสำหรับผู้ที่ต้องการใช้ NVIDIA GPU ของเครื่องตนเอง

## ตัวเลือกที่รองรับ

- Windows โดยตรง: Train ด้วย CPU ได้ตามคำสั่งใน `README.md`
- Windows + WSL2: ใช้ NVIDIA GPU ผ่าน Linux
- Google Colab: วิธีที่ง่ายที่สุดสำหรับผู้เริ่มต้น

## ตรวจสอบก่อนเริ่ม

1. เครื่องต้องมี NVIDIA GPU ที่รองรับ
2. ติดตั้ง NVIDIA Driver รุ่นที่รองรับ WSL2 จากเว็บไซต์ NVIDIA
3. เปิดใช้งาน WSL2 และติดตั้ง Ubuntu ตามเอกสารทางการของ Microsoft

ไม่ควรติดตั้ง CUDA หรือ cuDNN แบบสุ่มหลายเวอร์ชัน เพราะอาจทำให้ TensorFlow ใช้ GPU ไม่ได้ ให้ตรวจสอบเวอร์ชันที่ TensorFlow รองรับจากเอกสารทางการก่อนเสมอ

## ติดตั้งโปรเจกต์ภายใน WSL2

เปิด Ubuntu แล้วรัน:

```bash
git clone https://github.com/Bluedabade/ML-Assignment.git
cd ML-Assignment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_environment.py
```

ถ้า `check_environment.py` แสดง GPU แสดงว่าพร้อม Train:

```bash
python train.py --model mobilenetv2 --smoke-test
python train.py --model mobilenetv2 --epochs 10
```

ถ้าไม่พบ GPU ยังสามารถ Train ด้วย CPU ได้ หรือใช้ Google Colab แทน ไม่ต้องแก้ source code และไม่ต้องกำหนดอุปกรณ์ด้วยตนเอง เพราะ TensorFlow จะเลือก GPU ที่ตรวจพบให้อัตโนมัติ
