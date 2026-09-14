"""TensorFlow device selection shared by training and environment checks."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    selected: str
    physical_gpus: tuple[object, ...]
    logical_gpus: tuple[object, ...]


def configure_device(requested: str = "auto", *, verbose: bool = True) -> DeviceInfo:
    """Select CPU/GPU before models are created and avoid GPU memory preallocation."""
    import tensorflow as tf

    requested = requested.lower()
    if requested not in {"auto", "gpu", "cpu"}:
        raise ValueError("device must be one of: auto, gpu, cpu")

    physical_gpus = tuple(tf.config.list_physical_devices("GPU"))
    if requested == "gpu" and not physical_gpus:
        raise RuntimeError(
            "--device gpu was requested, but TensorFlow cannot see an NVIDIA GPU. "
            "On Windows with TensorFlow 2.11+, run inside WSL2, activate the GPU "
            "virtual environment, and run python check_environment.py."
        )

    selected = "gpu" if physical_gpus and requested != "cpu" else "cpu"
    if selected == "cpu":
        tf.config.set_visible_devices([], "GPU")
    else:
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as error:
                raise RuntimeError(
                    "GPU memory growth must be configured before model initialization."
                ) from error

    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    info = DeviceInfo(requested, selected, physical_gpus, logical_gpus)
    if verbose:
        print(f"Requested device: {requested.upper()}")
        print(f"Selected training device: {selected.upper()}")
        if selected == "gpu":
            print("GPU memory growth: ENABLED")
    return info


def gpu_device_name(gpu: object) -> str:
    """Return TensorFlow's GPU name when available without extra dependencies."""
    import tensorflow as tf

    try:
        details = tf.config.experimental.get_device_details(gpu)
        return str(details.get("device_name") or getattr(gpu, "name", gpu))
    except Exception:
        return str(getattr(gpu, "name", gpu))
