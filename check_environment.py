"""Print beginner-friendly TensorFlow CPU/GPU environment information."""
from __future__ import annotations

import platform
import sys


def main() -> int:
    print("=" * 48)
    print("Environment Check")
    print("=" * 48)
    print(f"Operating system: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")

    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow: NOT INSTALLED")
        print("GPU detected: NO")
        print("Training device: unavailable")
        print("Install requirements.txt (CPU) or requirements-gpu.txt (WSL2 GPU).")
        print("=" * 48)
        return 1

    from src.device import gpu_device_name

    physical_gpus = tuple(tf.config.list_physical_devices("GPU"))
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Device initialization may already have happened; reporting can continue.
            pass
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))

    print(f"TensorFlow: {tf.__version__}")
    print(f"TensorFlow built with CUDA: {'YES' if tf.test.is_built_with_cuda() else 'NO'}")
    print(f"GPU detected: {'YES' if physical_gpus else 'NO'}")
    print(f"Physical GPUs: {list(physical_gpus) if physical_gpus else 'None'}")
    print(f"Logical GPUs: {list(logical_gpus) if logical_gpus else 'None'}")
    for index, gpu in enumerate(physical_gpus):
        print(f"GPU device {index}: {gpu_device_name(gpu)}")
    print(f"Training device: {'GPU' if physical_gpus else 'CPU'}")

    windows_native = platform.system() == "Windows"
    tf_version = tuple(int(part) for part in tf.__version__.split(".")[:2])
    if windows_native and tf_version >= (2, 11):
        print()
        print("NOTE: TensorFlow 2.11+ does not support NVIDIA CUDA GPU training")
        print("in native Windows Python. Use WSL2 + Ubuntu for NVIDIA GPU training.")

    print("=" * 48)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
