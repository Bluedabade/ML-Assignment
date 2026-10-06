# Streamlit Skin Problem Detection

แอปตรวจจับ bounding boxes ของ Acne, Wrinkle, Dark Spot และ Enlarged Pore ไม่ใช่การวินิจฉัยทางการแพทย์ และไม่มีคลาส normal_skin

## 1. เตรียม UI environment

ใช้ Python 3.11 และ environment ที่สาม **นอก repository** ไม่ติดตั้งเพิ่มใน `.venv` หรือ Roboflow environment เดิม

```powershell
cd D:\Projects\ML-Assignment
py -3.11 -m venv D:\Projects\ML-Assignment-streamlit-env
& D:\Projects\ML-Assignment-streamlit-env\Scripts\python.exe -m pip install streamlit Pillow python-dotenv
```

ถ้ามี environment นี้แล้ว ไม่ต้องสร้างใหม่ วิธีติดตั้งอ้างอิง [Streamlit official instructions](https://docs.streamlit.io/get-started/installation/command-line)

## 2. ตั้งค่า SKN-1 authentication

สร้างไฟล์ `.env` ที่ root ของโปรเจกต์ด้วยตนเอง โดยใช้รูปแบบใน `.env.example` แล้วแทน placeholder ด้วย private API key ของตนเอง ไฟล์ `.env` และ `.env.*` ถูก Git ignore ยกเว้น `.env.example` ห้ามส่ง key ผ่านแชตหรือ commit ไฟล์ secret

แอปอ่าน `.env` ด้วย python-dotenv และ `override=False` ค่าที่ตั้งใน process environment อยู่แล้วจะมี priority ก่อน SKN-1 subprocess รับค่าผ่าน environment ไม่ใช่ command-line argument ไม่บันทึก key ในผลลัพธ์ หลังแก้ `.env` ให้เปิดแอปใหม่

อีกทางเลือกคือตั้งใน PowerShell เดียวกับที่เปิดแอป:

```powershell
$env:ROBOFLOW_API_KEY = "<your-key>"
```

YOLO11n ไม่ต้องใช้ key ส่วน SKN-1 ใช้ local ONNX CPU cache เดิมที่ `D:\Projects\ML-Assignment-roboflow-env\model_cache` ไม่มี cloud inference fallback อาจต้องติดต่อ Roboflow เพื่อยืนยันสิทธิ์ตอนโหลด model

## 3. เปิดแอป

```powershell
cd D:\Projects\ML-Assignment
.\run_streamlit.ps1
```

ถ้า PowerShell ปิดกั้น script ให้รัน `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` แล้วลองใหม่ เปิด http://127.0.0.1:8501 แอปผูกกับ localhost เท่านั้น

## 4. ใช้งาน

1. อัปโหลด JPG/JPEG/PNG จะเห็นภาพและขนาด แต่ยังไม่ทำนาย
2. เลือก YOLO11n Clean V2, SKN-1 หรือ Compare Both (ค่าเริ่มต้น)
3. ปรับ Confidence threshold 0.05–0.90 ค่าเริ่มต้น 0.25 ยิ่งสูงยิ่งตัด detection ที่ confidence ต่ำ ไม่ใช่การประเมินความรุนแรง
4. กด **Predict** เพื่อดู bounding boxes ตาราง confidence และจำนวน detection แยกคลาส
5. ถ้าเปลี่ยนภาพ/model/threshold ต้องกด Predict ใหม่

Acne = สิว, Wrinkle = ริ้วรอย, Dark Spot = จุดด่างดำ, Enlarged Pore = รูขุมขนที่เห็นเด่นชัด จำนวนที่แสดงเป็นจำนวนกล่องของโมเดล ไม่ใช่จำนวนโรคหรือการวินิจฉัย

ไม่พบ detection เหนือ threshold **ไม่ได้หมายความว่าผิวปกติ** SKN-1 แสดงเฉพาะ Acne → acne, Wrinkles → wrinkle, Dark Spots → dark_spot, Open Pores → enlarged_pore คลาสอื่นเก็บแยกใน JSON ไม่แสดงเป็นปัญหาผิวเป้าหมาย

## 5. Environment และไฟล์ผลลัพธ์

UI เรียก backend ผ่าน subprocess argument list โดยไม่ใช้ `shell=True`:

- YOLO11n: `.venv\Scripts\python.exe` ใช้ `best.pt` เดิม, imgsz=640, CUDA ถ้าพร้อม
- SKN-1: `D:\Projects\ML-Assignment-roboflow-env\Scripts\python.exe`, model `skn-1/2`, ONNX CPU
- โหลดโมเดลใหม่ต่อการกด Predict เพื่อแยก process/dependencies โดยไม่ import framework ใน UI จึงมีเวลารอโหลดโมเดล

ผลเก็บใน `results/streamlit_predictions/<timestamp_unique-id>/`: original image (bytes เดิม), `input.png` (RGB/EXIF-oriented เหมือนกันสำหรับทั้งสองโมเดล), annotated JPG และ `predictions.json` รวม backend errors แบบไม่เผย credentials อาจมี runtime settings/cache ภายในโฟลเดอร์ session

ภาพใบหน้าอาจเป็นข้อมูลส่วนบุคคล ใช้ภาพที่ได้รับอนุญาต ผลอัปโหลดเก็บ local ไม่ commit โดยไม่ตรวจความยินยอมก่อน แอปไม่แก้ dataset, checkpoints หรือผลทดลองเดิม

เวลา CPU ของ SKN-1 ไม่ควรเทียบโดยตรงกับ CUDA ของ YOLO11n; provenance/training-data overlap ของ SKN-1 ยังยืนยันไม่ได้

## 6. หยุดแอป

กด `Ctrl+C` ใน PowerShell ที่รัน Streamlit ไม่ต้องถอน package หรือแก้ environment เดิม
