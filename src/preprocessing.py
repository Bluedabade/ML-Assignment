"""PRE-PROCESSING: resize, model-appropriate normalization, augmentation."""
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="skin_classification")
class ModelPreprocessing(tf.keras.layers.Layer):
    """Serializable ImageNet preprocessing for each supported backbone."""
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name.lower()

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        if self.model_name == "mobilenetv2":
            return inputs / 127.5 - 1.0
        if self.model_name == "efficientnetb0":
            # EfficientNetB0 includes its own Rescaling layer in TensorFlow 2.16.
            return inputs
        if self.model_name == "resnet50":
            # Keras ResNet50 uses Caffe-style RGB->BGR and channel mean subtraction.
            return tf.reverse(inputs, axis=[-1]) - tf.constant([103.939, 116.779, 123.68])
        raise ValueError(f"Unknown model: {self.model_name}")

    def get_config(self):
        return {**super().get_config(), "model_name": self.model_name}

def make_augmentation(seed: int) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=seed + 1),
        tf.keras.layers.RandomZoom(0.08, fill_mode="reflect", seed=seed + 2),
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed + 3),
        tf.keras.layers.RandomContrast(0.08, seed=seed + 4),
    ], name="skin_safe_augmentation")
