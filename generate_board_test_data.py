"""
Generate a compact test data package for on-board FPGA validation.
Outputs:
  - pynq_test_data/test_inputs.npy      (N x 10 scaled+selected features)
  - pynq_test_data/test_labels.npy      (N,) ground truth class indices
  - pynq_test_data/keras_preds.npy      (N x 3) Keras softmax reference
  - pynq_test_data/class_names.txt      Class name mapping

Upload the whole pynq_test_data/ folder to your PYNQ board.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load
try:
    model = tf.keras.models.load_model("hls4ml_prj/keras_model.keras")
except Exception:
    model = tf.keras.models.load_model("fatigue_model_fpga.keras")

scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

df = pd.read_csv("advitam_exp4_eda_hr_fatigue.csv")
X = df.drop(columns=["subject_id", "fatigue_binary", "fatigue_class"])
y_raw = df["fatigue_class"]

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

# Use the same test split as original notebook
_, X_test_raw, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess
X_test_scaled = scaler.transform(X_test_raw)
X_test_sel = selector.transform(X_test_scaled)

# Keras reference predictions
keras_preds = model.predict(X_test_sel, verbose=0)

# Save
out_dir = "pynq_test_data"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/test_inputs.npy", X_test_sel.astype(np.float32))
np.save(f"{out_dir}/test_labels.npy", y_test.astype(np.int32))
np.save(f"{out_dir}/keras_preds.npy", keras_preds.astype(np.float32))

with open(f"{out_dir}/class_names.txt", "w") as f:
    for i, name in enumerate(encoder.classes_):
        f.write(f"{i} {name}\n")

print(f"Test data saved to {out_dir}/")
print(f"  test_inputs.npy  : {X_test_sel.shape}  (preprocessed features)")
print(f"  test_labels.npy  : {y_test.shape}       (ground truth)")
print(f"  keras_preds.npy  : {keras_preds.shape}  (Keras reference)")
print(f"  class_names.txt  : {list(encoder.classes_)}")
print(f"\nUpload this folder to your PYNQ board for on-board testing.")
