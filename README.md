# วิธีติดตั้งและเทรนโมเดล

Dataset ที่เตรียมไว้แล้วอยู่ใน Repository นี้ หลังจาก Clone และติดตั้ง Environment สามารถเริ่ม Train ได้ทันที **ไม่ต้องดาวน์โหลดหรือเตรียม Dataset เพิ่ม**

## 1. Clone Repository

```bash
git clone https://github.com/Bluedabade/ML-Assignment.git
cd ML-Assignment
```

- `git clone` คือ ดาวน์โหลดโปรเจกต์จาก GitHub
- `cd ML-Assignment` คือ เข้าไปในโฟลเดอร์โปรเจกต์

## 2. สร้าง Python Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

ถ้า PowerShell ไม่อนุญาตให้เปิดใช้งาน ให้พิมพ์:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

เมื่อเปิดใช้งานสำเร็จ ปกติจะเห็น `(.venv)` อยู่ด้านหน้าบรรทัดคำสั่ง

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. ติดตั้ง Library

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

คำสั่งนี้จะติดตั้ง Library ที่จำเป็นสำหรับการ Train

## 4. ตรวจสอบ Environment

```bash
python check_environment.py
```

คำสั่งนี้ใช้ตรวจสอบ Python, TensorFlow และ GPU

ถ้าไม่พบ GPU ยังสามารถ Train ด้วย CPU ได้ แต่จะใช้เวลานานกว่า

## 5. ทดลองระบบก่อน Train จริง

แนะนำให้ทดลองด้วยคำสั่งนี้ก่อน:

```bash
python train.py --model mobilenetv2 --smoke-test
```

เป็นการทดสอบสั้น ๆ เพื่อดูว่า:

- Dataset โหลดได้ถูกต้อง
- Model เริ่ม Train ได้
- Environment ทำงานได้

ผลจากคำสั่งนี้ **ไม่ใช่ผลการ Train จริงขั้นสุดท้าย**

## 6. เริ่ม Train โมเดล

### MobileNetV2

```bash
python train.py --model mobilenetv2 --epochs 10
```

### EfficientNetB0

```bash
python train.py --model efficientnetb0 --epochs 10
```

### ResNet50

```bash
python train.py --model resnet50 --epochs 10
```

- `--model` ใช้เลือก Model ที่ต้องการ Train
- `--epochs 10` หมายถึงให้ Model เรียนรู้ข้อมูลทั้งหมด 10 รอบ

เวลาในการ Train ขึ้นอยู่กับความเร็วของคอมพิวเตอร์และ GPU

## 7. ถ้าต้องการ Train ทีละโมเดล

สำหรับผู้เริ่มต้น แนะนำให้ Train ทีละ Model โดยเริ่มจาก MobileNetV2:

```bash
python train.py --model mobilenetv2 --epochs 10
```

เมื่อเสร็จแล้วจึง Train EfficientNetB0:

```bash
python train.py --model efficientnetb0 --epochs 10
```

จากนั้น Train ResNet50:

```bash
python train.py --model resnet50 --epochs 10
```

ไม่จำเป็นต้องรันทั้งสามคำสั่งพร้อมกัน

## 8. หลังจาก Train เสร็จ

โปรแกรมจะบันทึกผลการ Train และกราฟให้อัตโนมัติตามระบบที่เตรียมไว้
