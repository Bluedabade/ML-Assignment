import platform
import sys

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Detected GPUs: {gpus if gpus else 'None (CPU mode)'}")
    print(f"TensorFlow built with CUDA support: {tf.test.is_built_with_cuda()}")
except ImportError:
    print("TensorFlow: NOT INSTALLED")
    print("Run: pip install -r requirements.txt")
