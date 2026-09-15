# Unmapped data

ข้อมูลต้นฉบับที่ไม่ใช่ 5 target classes จะไม่ถูกเดา label และยังเก็บครบใน `dataset/raw/`:

- Oily-Dry-Skin-Types: `dry` และ `oily`
- Roboflow: class นอกขอบเขต ภาพ multi-label และภาพไม่มี annotation

รายการระดับไฟล์อยู่ที่ `results/reports/excluded_images.csv` การอ้างกลับไปยัง raw
หลีกเลี่ยงสำเนาหลายพันไฟล์โดยข้อมูลไม่สูญหาย
