"""CORE PROCESSING: ImageNet transfer-learning model constructors."""
import tensorflow as tf
from config import IMG_SIZE, LEARNING_RATE, NUM_CLASSES, SEED
from src.preprocessing import ModelPreprocessing, make_augmentation

BACKBONES = {
    "mobilenetv2": tf.keras.applications.MobileNetV2,
    "efficientnetb0": tf.keras.applications.EfficientNetB0,
    "resnet50": tf.keras.applications.ResNet50,
}

def build_model(model_name: str, image_size: int = IMG_SIZE, weights: str | None = "imagenet",
                learning_rate: float = LEARNING_RATE):
    name = model_name.lower()
    if name not in BACKBONES:
        raise ValueError(f"Choose one of: {', '.join(BACKBONES)}")
    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    x = make_augmentation(SEED)(inputs)
    x = ModelPreprocessing(name, name="normalize")(x)
    backbone = BACKBONES[name](include_top=False, weights=weights,
                               input_shape=(image_size, image_size, 3))
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="skin_class")(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model, backbone

def enable_fine_tuning(model, backbone, unfreeze_last: int = 30, learning_rate: float = 1e-5):
    backbone.trainable = True
    for layer in backbone.layers[:-unfreeze_last]:
        layer.trainable = False
    # Frozen BatchNorm statistics are safer for small student datasets.
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
