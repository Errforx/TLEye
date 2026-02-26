import tensorflow as tf
import os, h5py, json

# Paths
MODEL_H5 = "sounddetection/best_model_fixed.h5"
MODEL_TFLITE = "sounddetection/best_model_fixed.tflite"

assert os.path.exists(MODEL_H5), f"❌ File not found: {MODEL_H5}"
print(f"🔍 Repairing and converting: {MODEL_H5}")

# --- Step 1: Read and patch config from HDF5 ---
with h5py.File(MODEL_H5, "r") as f:
    raw_cfg = f.attrs["model_config"]
    raw_cfg = raw_cfg.decode() if isinstance(raw_cfg, bytes) else raw_cfg
    cfg = json.loads(raw_cfg)

removed = 0
for layer in cfg["config"]["layers"]:
    if layer["class_name"] == "LSTM" and "time_major" in layer["config"]:
        layer["config"].pop("time_major")
        removed += 1

print(f"🩹 Removed {removed} invalid keys from LSTM layers.")

# --- Step 2: Rebuild the Keras model safely ---
print("🧱 Reconstructing model...")
model = tf.keras.Model.from_config(cfg["config"])
model.load_weights(MODEL_H5)
print("✅ Model loaded successfully.")

# --- Step 3: Convert to TFLite with SELECT_TF_OPS ---
print("⚙️ Converting to float32 TFLite (SELECT_TF_OPS)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()

os.makedirs(os.path.dirname(MODEL_TFLITE), exist_ok=True)
with open(MODEL_TFLITE, "wb") as f:
    f.write(tflite_model)

print(f"✅ Saved converted model at: {MODEL_TFLITE}")
print("💡 You can now use ai_edge_litert.Interpreter for faster inference.")
