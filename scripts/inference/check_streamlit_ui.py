"""Read-only UI smoke checks; never calls model inference."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def main():
    from streamlit.testing.v1 import AppTest
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    args = parser.parse_args()
    app = AppTest.from_file(str(ROOT / 'streamlit_app.py')).run()
    assert not app.exception
    assert app.radio[0].value == 'Compare Both'
    assert app.slider[0].value == 0.25
    for model in ['YOLO11n Clean V2', 'SKN-1', 'Compare Both']:
        app.radio[0].set_value(model).run()
        assert not app.exception
    source = '''
import json
from pathlib import Path
import streamlit as st
from streamlit_app import show_result
payload = json.loads(Path(PATH).read_text(encoding='utf-8'))
for col, (name, result) in zip(st.columns(2), payload['results'].items()):
    with col:
        show_result(name, result)
'''.replace('PATH', repr(str(args.predictions.resolve())))
    comparison = AppTest.from_string(source).run()
    assert not comparison.exception
    print(json.dumps({'ui': 'PASS', 'compare_rendering': 'PASS',
                      'tables': len(comparison.dataframe), 'backend_errors_shown': len(comparison.error)}))

if __name__ == '__main__':
    main()
