import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from model_gru import AttentionLayer
from data_loader import load_dataset

MODEL_DIR = "models"
THRESHOLD_DIR = "thresholds"
THRESHOLD_FILE = os.path.join(THRESHOLD_DIR, "best_threshold.json")

EXPECTED_FEATURE_DIM = 47

os.makedirs(THRESHOLD_DIR, exist_ok=True)


def load_model(model_id):
    model_path = f"{MODEL_DIR}/alz_{model_id}_final.keras"

    if model_id == "gru_attention":
        return tf.keras.models.load_model(
            model_path,
            custom_objects={"AttentionLayer": AttentionLayer},
        )

    return tf.keras.models.load_model(model_path)


def validate_shapes(model, X):
    if X.shape[2] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"❌ Data feature mismatch: {X.shape[2]} != {EXPECTED_FEATURE_DIM}"
        )

    if model.input_shape[2] != X.shape[2]:
        raise ValueError(
            f"❌ Model expects {model.input_shape[2]} features, "
            f"but data has {X.shape[2]}"
        )


def find_best_threshold():
    print("📂 Loading validation dataset...")
    _, _, X_val, y_val = load_dataset()

    models = {
        "cnn_lstm": load_model("cnn_lstm"),
        "gru_attention": load_model("gru_attention"),
    }

    for m in models.values():
        validate_shapes(m, X_val)

    print("🔮 Running ensemble predictions once...")
    preds = []
    for model in models.values():
        preds.append(model.predict(X_val).ravel())

    ensemble_prob = np.mean(preds, axis=0)

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0

    print("🔍 Searching for optimal threshold...")

    for t in thresholds:
        y_pred = (ensemble_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    print(f"✅ Best Threshold Found: {best_threshold:.2f}")
    print(f"🏆 Best F1-score     : {best_f1:.4f}")

    with open(THRESHOLD_FILE, "w") as f:
        json.dump(
            {
                "best_threshold": best_threshold,
                "best_f1": best_f1
            },
            f,
            indent=4
        )

    print(f"💾 Threshold saved to: {THRESHOLD_FILE}")


if __name__ == "__main__":
    find_best_threshold()
