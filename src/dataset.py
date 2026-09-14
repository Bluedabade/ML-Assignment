from pathlib import Path
import tensorflow as tf
from config import BATCH_SIZE, CLASS_NAMES, IMG_SIZE, PROCESSED_DATA_DIR, SEED

def load_datasets(data_dir: Path = PROCESSED_DATA_DIR, image_size: int = IMG_SIZE,
                  batch_size: int = BATCH_SIZE, seed: int = SEED):
    options = dict(image_size=(image_size, image_size), batch_size=batch_size,
                   class_names=CLASS_NAMES, label_mode="int")
    train = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", shuffle=True, seed=seed, **options)
    val = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val", shuffle=False, **options)
    test = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test", shuffle=False, **options)
    autotune = tf.data.AUTOTUNE
    return (train.prefetch(autotune), val.prefetch(autotune), test.prefetch(autotune))
