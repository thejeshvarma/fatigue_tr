"""
Generate testbench data for HLS C-simulation.
Run this BEFORE running Vitis HLS C-sim.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Try loading model - fall back to rebuilding if version mismatch
try:
    model = tf.keras.models.load_model("hls4ml_prj/keras_model.keras")
except Exception:
    try:
        model = tf.keras.models.load_model("fatigue_model_fpga.keras")
    except Exception:
        print("Cannot load saved model, rebuilding from notebook architecture...")
        from tensorflow.keras.layers import Input, Dense, ReLU
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2
        # Rebuild won't have trained weights - this path shouldn't be needed
        raise RuntimeError(
            "Could not load model. Please re-run the training notebook "
            "and save with: model.save('fatigue_model_fpga.keras')"
        )
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

# Load data
df = pd.read_csv("advitam_exp4_eda_hr_fatigue.csv")
X = df.drop(columns=["subject_id", "fatigue_binary", "fatigue_class"])
y = df["fatigue_class"]

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Preprocess (same pipeline as training)
X_scaled = scaler.transform(X)
X_selected = selector.transform(X_scaled)

# Get Keras predictions (float32 reference)
preds = model.predict(X_selected, verbose=0)

# Save to hls4ml testbench format (space-separated)
os.makedirs("hls4ml_prj/tb_data", exist_ok=True)
np.savetxt("hls4ml_prj/tb_data/tb_input_features.dat", X_selected, fmt="%.6f")
np.savetxt("hls4ml_prj/tb_data/tb_output_predictions.dat", preds, fmt="%.6f")

print(f"Saved {len(X_selected)} test vectors to hls4ml_prj/tb_data/")
print(f"  tb_input_features.dat    : shape {X_selected.shape}")
print(f"  tb_output_predictions.dat: shape {preds.shape}")
print(f"  Classes: {list(encoder.classes_)}")
