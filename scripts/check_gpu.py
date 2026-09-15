import platform,sys
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
try:
 import tensorflow as tf
 print("Python Version:",platform.python_version()); print("TensorFlow Version:",tf.__version__); g=tf.config.list_physical_devices("GPU"); print("CUDA availability:",bool(g)); print("Detected GPU:",g or "ไม่พบ GPU — ตรวจ WSL2 และ NVIDIA driver")
 for x in g:
  try: tf.config.experimental.set_memory_growth(x,True)
  except RuntimeError: pass
except ImportError: print("ไม่พบ TensorFlow: รัน pip install -r requirements.txt")
