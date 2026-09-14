import json
import os
import random
from pathlib import Path

import numpy as np

from config import CLASS_NAMES, SEED

def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass

def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

def load_class_names(path: Path) -> list[str]:
    names = json.loads(path.read_text(encoding="utf-8"))
    if names != CLASS_NAMES:
        raise ValueError(f"Unexpected class order in {path}: {names}")
    return names
