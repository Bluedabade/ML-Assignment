"""โหลดภาพด้วย ImageDataGenerator ตาม Chapter 7."""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import BATCH_SIZE,CLASS_NAMES,IMAGE_SIZE,RANDOM_SEED,TEST_DIR,TRAIN_DIR,VAL_DIR
def create_generators():
    train_gen=ImageDataGenerator(rescale=1/255.,rotation_range=20,brightness_range=None,shear_range=.2,zoom_range=.1,channel_shift_range=.1,fill_mode="nearest")
    eval_gen=ImageDataGenerator(rescale=1/255.)
    common=dict(target_size=IMAGE_SIZE,batch_size=BATCH_SIZE,class_mode="categorical",classes=CLASS_NAMES)
    return (train_gen.flow_from_directory(TRAIN_DIR,shuffle=True,seed=RANDOM_SEED,**common),eval_gen.flow_from_directory(VAL_DIR,shuffle=False,**common),eval_gen.flow_from_directory(TEST_DIR,shuffle=False,**common))
