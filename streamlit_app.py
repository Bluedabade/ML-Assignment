"""UI-only Streamlit process; models run in their own existing environments."""
import hashlib
import io
import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=False)
ROOT = BASE_DIR
OUTPUT = ROOT / 'results/streamlit_predictions'
DISPLAY = {'acne':'Acne', 'wrinkle':'Wrinkle', 'dark_spot':'Dark Spot', 'enlarged_pore':'Enlarged Pore'}
MODELS = {
    'YOLO11n Clean V2': (ROOT / '.venv/Scripts/python.exe', 'yolo11n_predict.py'),
    'SKN-1': (ROOT.parent / 'ML-Assignment-roboflow-env/Scripts/python.exe', 'skn1_predict.py'),
}
ERRORS = {
    'MISSING_ROBOFLOW_API_KEY': 'SKN-1 ต้องใช้ ROBOFLOW_API_KEY: สร้างไฟล์ .env ที่ root ของโปรเจกต์ด้วยตนเองตาม .env.example หรือตั้งค่าใน PowerShell แล้วเปิดแอปใหม่ ห้ามส่ง key ผ่านแชต',
    'MISSING_LOCAL_MODEL_CACHE': 'ไม่พบ SKN-1 local model cache กรุณาตรวจ environment เดิม',
    'MISSING_MODEL_OR_CACHE': 'ไม่พบ checkpoint หรือ model cache',
    'ACCESS_REQUIRED': 'ไม่มีสิทธิ์เข้าถึงไฟล์โมเดลหรือ cache',
    'LOCAL_INFERENCE_FAILED': 'Local inference ไม่สำเร็จ กรุณาตรวจ environment/model cache โดยไม่เผยแพร่ credentials',
}

def invoke_backend(name, image_path, confidence, folder):
    interpreter, script = MODELS[name]
    if not interpreter.is_file():
        return {'error': f'ไม่พบ Python environment สำหรับ {name}'}
    if name == 'SKN-1' and not os.getenv('ROBOFLOW_API_KEY'):
        return {'error': ERRORS['MISSING_ROBOFLOW_API_KEY']}
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    try:
        process = subprocess.run([str(interpreter), '-B', str(ROOT / 'scripts/inference' / script),
            '--image', str(image_path), '--confidence', str(confidence), '--output', str(folder)],
            cwd=ROOT, env=environment, capture_output=True, text=True, encoding='utf-8',
            timeout=180, shell=False)
        result = json.loads(process.stdout)
        if process.returncode or 'error' in result:
            return {'error': ERRORS.get(result.get('error'), ERRORS['LOCAL_INFERENCE_FAILED'])}
        if not Path(result['annotated_image']).is_file():
            return {'error': 'Backend ไม่ได้สร้าง annotated image'}
        return result
    except subprocess.TimeoutExpired:
        return {'error': f'{name} ใช้เวลานานเกิน 180 วินาที ลองใหม่ภายหลัง'}
    except (OSError, ValueError, KeyError):
        return {'error': 'Backend ตอบกลับไม่สมบูรณ์ กรุณาตรวจ environment'}

def predict_session(image_bytes, filename, selected, confidence):
    folder = OUTPUT / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True, exist_ok=False)
    # Preserve uploaded bytes without trusting the uploaded filename as a path.
    suffix = Path(filename).suffix.lower()
    suffix = suffix if suffix in {'.jpg', '.jpeg', '.png'} else '.png'
    original = folder / ('original' + suffix)
    original.write_bytes(image_bytes)
    # Both backends receive the same EXIF-oriented RGB input, not the source file.
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert('RGB')
    input_path = folder / 'input.png'
    image.save(input_path)
    names = list(MODELS) if selected == 'Compare Both' else [selected]
    results = {name: invoke_backend(name, input_path, confidence, folder) for name in names}
    payload = dict(filename=Path(filename).name, width=image.width, height=image.height,
                   confidence=confidence, results=results, output_directory=str(folder))
    (folder / 'predictions.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload

def show_result(name, result):
    st.subheader(name)
    if 'error' in result:
        st.error(result['error'])
        return
    st.image(result['annotated_image'], width='stretch')
    predictions = result['predictions']
    st.caption(f"Backend: {result['backend']} | Inference: {result['inference_seconds']:.3f} s (ไม่ใช้เปรียบเทียบความเร็วข้าม backend)")
    if not predictions:
        st.info('No target skin problem was detected above the selected confidence threshold.')
    else:
        st.dataframe([{'Class': DISPLAY[p['canonical_class']], 'Confidence': f"{p['confidence']:.2%}",
            'Bounding Box': ', '.join(f'{p[k]:.1f}' for k in ('x1','y1','x2','y2'))} for p in predictions],
            hide_index=True, width='stretch')
    st.write(f'Total detections: {len(predictions)}')
    for cls, label in DISPLAY.items():
        st.write(f"{label}: {sum(p['canonical_class'] == cls for p in predictions)}")

def main():
    st.set_page_config(page_title='Skin Problem Detection', page_icon='🔬', layout='wide')
    st.title('Skin Problem Detection System')
    st.caption('ตรวจจับปัญหาผิวด้วย YOLO11n และ SKN-1')
    with st.sidebar:
        st.markdown('### Supported Classes')
        for name in DISPLAY.values():
            st.write(f'- {name}')
        st.markdown('### Model')
        selected = st.radio('เลือก Model', [*MODELS, 'Compare Both'], index=2)
        confidence = st.slider('Confidence threshold', 0.05, 0.90, 0.25, 0.05)
        st.caption('YOLO11n Clean V2 = model trained by this project')
        st.caption('SKN-1 = external pretrained skin-specific detector (ONNX CPU)')
        st.caption('โมเดลทำงานใน subprocess แยก environment; ไม่โหลด framework ใน UI')
        if selected != 'YOLO11n Clean V2' and not os.getenv('ROBOFLOW_API_KEY'):
            st.warning(ERRORS['MISSING_ROBOFLOW_API_KEY'])
    uploaded = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert('RGB')
        except (OSError, Image.DecompressionBombError):
            st.error('ไฟล์ภาพไม่ถูกต้องหรือมีขนาดใหญ่เกินไป')
            return
        st.subheader('Original Image')
        st.image(image, width='stretch')
        st.write(f'Filename: {uploaded.name} | Width: {image.width} | Height: {image.height}')
        signature = (hashlib.sha256(image_bytes).hexdigest(), selected, confidence)
        if st.button('Predict', type='primary'):
            with st.spinner('กำลังตรวจจับปัญหาผิว…'):
                try:
                    st.session_state['prediction'] = predict_session(image_bytes, uploaded.name, selected, confidence)
                    st.session_state['prediction_signature'] = signature
                except (OSError, ValueError):
                    st.error('ไม่สามารถบันทึกภาพหรือผลลัพธ์ได้ กรุณาตรวจพื้นที่ดิสก์และสิทธิ์การเข้าถึง')
        if st.session_state.get('prediction_signature') == signature:
            payload = st.session_state['prediction']
            if selected == 'Compare Both':
                for column, (name, result) in zip(st.columns(2), payload['results'].items()):
                    with column:
                        show_result(name, result)
            else:
                show_result(selected, payload['results'][selected])
            st.caption(f"Output: {payload['output_directory']}")
    else:
        st.info('อัปโหลดภาพแล้วกด Predict เพื่อเริ่มตรวจจับ')
    st.divider()
    st.caption('หมายเหตุ: ระบบนี้เป็นต้นแบบสำหรับงานด้าน Machine Learning และไม่ใช่เครื่องมือสำหรับการวินิจฉัยทางการแพทย์')
    st.caption('ไม่พบ detection ไม่ได้แปลว่าผิวปกติ; โมเดลไม่มี normal_skin ที่ผ่านการตรวจสอบ และ provenance ของ SKN-1 อาจซ้อนทับข้อมูลสาธารณะ')

if __name__ == '__main__':
    main()
