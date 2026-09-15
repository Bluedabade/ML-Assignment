"""ค่ากลางสำหรับการทดลอง MobileNetV2 แบบ 5 classes."""
from pathlib import Path
ROOT_DIR=Path(__file__).resolve().parents[1]; ROOT=ROOT_DIR
DATASET_DIR=ROOT_DIR/"dataset"; RAW_DATA_DIR=DATASET_DIR/"raw"; PREPARED_DATA_DIR=DATASET_DIR/"prepared"
TRAIN_DIR=PREPARED_DATA_DIR/"training_set"; VAL_DIR=PREPARED_DATA_DIR/"val_set"; TEST_DIR=PREPARED_DATA_DIR/"test_set"
UNMAPPED_DIR=DATASET_DIR/"unmapped"; MODELS_DIR=ROOT_DIR/"models"; RESULTS_DIR=ROOT_DIR/"results"
REPORTS_DIR=RESULTS_DIR/"reports"; PLOTS_DIR=RESULTS_DIR/"plots"; HISTORIES_DIR=RESULTS_DIR/"histories"
ARTIFACTS_DIR=MODELS_DIR; PROCESSED_DATA_DIR=PREPARED_DATA_DIR
IMAGE_SIZE=(128,128); IMG_SIZE=128; BATCH_SIZE=128; NUM_CLASSES=5
HEAD_EPOCHS=10; INITIAL_EPOCHS=HEAD_EPOCHS; LEARNING_RATE=.0001
HEAD_DENSE_UNITS=[]; DROPOUT_RATES=[]
ENABLE_FINE_TUNING=True; FINE_TUNE_AT=100; FINE_TUNE_EPOCHS=5; FINE_TUNE_LEARNING_RATE=.00001
RANDOM_SEED=42; SEED=RANDOM_SEED
CLASS_NAMES=["normal","wrinkle","acne","dark_spot","large_pore"]
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
FOLDER_CLASS_MAP={"normal":"normal","wrinkle":"wrinkle","wrinkles":"wrinkle","acne":"acne","spots":"dark_spot","dark spot":"dark_spot","dark spots":"dark_spot","dark_spot":"dark_spot","large pore":"large_pore","large pores":"large_pore","large_pore":"large_pore","large_pores":"large_pore"}
COCO_CLASS_MAP={"Acne":"acne","Wrinkles":"wrinkle","Dark-Spots":"dark_spot","Englarged-Pores":"large_pore"}
SOURCE_NAMES=("kaggle_acne_wrinkle_spots","kaggle_oily_dry_normal","roboflow_skin_problem_clean3")
def locate_source(name):
    for p in (RAW_DATA_DIR/name,ROOT_DIR/name):
        if p.exists(): return p
    return None
def validate_config():
    if len(CLASS_NAMES)!=NUM_CLASSES or len(set(CLASS_NAMES))!=NUM_CLASSES: raise ValueError("CLASS_NAMES ต้องครบ 5 classes และไม่ซ้ำ")
    if len(DROPOUT_RATES) not in (0,1,len(HEAD_DENSE_UNITS)): raise ValueError("DROPOUT_RATES ต้องว่าง, 1 ค่า หรือเท่ากับจำนวน Dense layers")
