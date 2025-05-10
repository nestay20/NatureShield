import tensorflow as tf
import os

# path to the SavedModel directory
SAVED_MODEL_DIR = os.getcwd()  # or './' if you're already in that folder

# 1) Load the SavedModel (omit tags → will auto‑detect the only MetaGraph)
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

# 2) Ensure float32 I/O (no quantization)
converter.optimizations = []

# 3) (Optional) you can set supported ops if you need only the built‑in float kernels
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# 4) Convert and save
tflite_model = converter.convert()
with open("model_float32.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Wrote float32 TFLite model to 'model_float32.tflite'")
