# แหล่งข้อมูล Dataset

## kaggle_acne_wrinkle_spots
- Original classes: acne, spots, wrinkles (อย่างละ 200 ภาพ)
- Mapping: acne→acne, spots→dark_spot, wrinkles→wrinkle
- URL/License: ไม่พบ metadata ในไฟล์ที่ได้รับ ต้องยืนยันก่อนเผยแพร่ต่อ

## Oily-Dry-Skin-Types
- Original classes: dry, normal, oily
- Mapping: normal→normal เท่านั้น; dry/oily ไม่ map
- URL/License: ไม่พบ metadata ในไฟล์ที่ได้รับ ต้องยืนยันก่อนเผยแพร่ต่อ

## Skin-Problem-Detection-Relabel-Clean3 v2
- Source: https://universe.roboflow.com/parin-kittipongdaja-vwmn3/skin-problem-detection-relabel-clean3
- License: CC BY 4.0; README ต้นฉบับเก็บไว้ใน raw
- ใช้ COCO image เฉพาะเมื่อ annotation ทั้งหมด map ไป target class เดียว

รายละเอียด: `results/reports/dataset_audit.json`, `results/reports/dataset_report.txt`, `dataset_mapping.json`
