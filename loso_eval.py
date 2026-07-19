"""
LOSO (Leave-One-Subject-Out) Cross-Validation for Fatigue Classification
=========================================================================
Fixes the data leakage issue by ensuring no subject appears in both
train and test sets simultaneously.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # suppress TF info/warnings

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut

from tensorflow.keras.layers import Input, Dense, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ── reproducibility ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── load data ────────────────────────────────────────────────────────────
df = pd.read_csv("advitam_exp4_eda_hr_fatigue.csv")
print(f"Dataset shape : {df.shape}")
print(f"Subjects      : {sorted(df['subject_id'].unique())}")
print(f"Class balance :\n{df['fatigue_class'].value_counts()}\n")

X = df.drop(columns=["subject_id", "fatigue_binary", "fatigue_class"])
y_raw = df["fatigue_class"]
groups = df["subject_id"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
class_names = encoder.classes_
n_classes = len(class_names)

print(f"Classes       : {list(class_names)}  ->  encoded {list(range(n_classes))}")
print(f"Num subjects  : {len(np.unique(groups))}")
print("=" * 70)

# ── helper: build fresh model ────────────────────────────────────────────
def build_model(n_features=10, n_classes=3):
    inputs = Input(shape=(n_features,))
    x = Dense(16, use_bias=False, kernel_regularizer=l2(1e-4))(inputs)
    x = ReLU()(x)
    x = Dense(8, use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = ReLU()(x)
    outputs = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ── LOSO cross-validation ───────────────────────────────────────────────
logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(X, y, groups)

all_y_true = []
all_y_pred = []
fold_results = []

print(f"\nRunning {n_splits}-fold LOSO cross-validation ...\n")

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
    subject_id = groups[test_idx[0]]
    n_train, n_test = len(train_idx), len(test_idx)

    X_train_raw = X.iloc[train_idx].values
    X_test_raw  = X.iloc[test_idx].values
    y_train     = y[train_idx]
    y_test      = y[test_idx]

    # ── per-fold scaling (fit on train ONLY) ─────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    # ── per-fold feature selection (fit on train ONLY) ───────────────────
    selector = SelectKBest(score_func=mutual_info_classif, k=10)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel  = selector.transform(X_test_scaled)

    # ── one-hot labels ───────────────────────────────────────────────────
    y_train_cat = to_categorical(y_train, n_classes)
    y_test_cat  = to_categorical(y_test,  n_classes)

    # ── build & train a fresh model each fold ────────────────────────────
    tf.random.set_seed(SEED)
    model = build_model(n_features=10, n_classes=n_classes)

    model.fit(
        X_train_sel,
        y_train_cat,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )

    # ── predict ──────────────────────────────────────────────────────────
    preds_proba = model.predict(X_test_sel, verbose=0)
    preds = np.argmax(preds_proba, axis=1)

    fold_acc = accuracy_score(y_test, preds)
    fold_f1  = f1_score(y_test, preds, average="macro", zero_division=0)

    fold_results.append({
        "fold": fold_idx,
        "subject_id": subject_id,
        "n_train": n_train,
        "n_test": n_test,
        "accuracy": fold_acc,
        "f1_macro": fold_f1,
    })

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(preds.tolist())

    print(f"  Fold {fold_idx:2d}  |  Subject {subject_id:3d}  |  "
          f"samples {n_test:3d}  |  acc {fold_acc:.4f}  |  F1 {fold_f1:.4f}")

    # free memory
    del model
    tf.keras.backend.clear_session()

# ── aggregate results ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("LOSO CROSS-VALIDATION RESULTS")
print("=" * 70)

results_df = pd.DataFrame(fold_results)
print("\nPer-subject breakdown:")
print(results_df.to_string(index=False))

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

overall_acc = accuracy_score(all_y_true, all_y_pred)
overall_f1  = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
mean_acc    = results_df["accuracy"].mean()
std_acc     = results_df["accuracy"].std()
mean_f1     = results_df["f1_macro"].mean()
std_f1      = results_df["f1_macro"].std()

print(f"\n{'-' * 50}")
print(f"  Overall accuracy (pooled)   : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
print(f"  Overall F1 macro (pooled)   : {overall_f1:.4f}")
print(f"  Mean accuracy  ± std        : {mean_acc:.4f} ± {std_acc:.4f}")
print(f"  Mean F1 macro  ± std        : {mean_f1:.4f} ± {std_f1:.4f}")
print(f"{'-' * 50}")

print(f"\nClassification Report (pooled across all folds):\n")
print(classification_report(
    all_y_true, all_y_pred,
    target_names=[str(c) for c in class_names],
    zero_division=0,
))

print("Confusion Matrix (rows=true, cols=pred):")
cm = confusion_matrix(all_y_true, all_y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df)
print()

# ── comparison summary ───────────────────────────────────────────────────
print("=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"  Original (random split, data leakage)  : 99.48% accuracy")
print(f"  LOSO     (no data leakage)             : {overall_acc*100:.2f}% accuracy")
print(f"  Accuracy drop                          : {(99.48 - overall_acc*100):.2f} pp")
print("=" * 70)
