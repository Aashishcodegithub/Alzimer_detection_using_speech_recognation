import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import class_weight
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from model import build_cnn_lstm_model
from model_gru import build_gru_attention_model
from data_loader import load_dataset


MODEL_DIR = "models"
PLOTS_DIR = "plots"
HISTORY_DIR = os.path.join(PLOTS_DIR, "history")

EPOCHS = 50
BATCH_SIZE = 32

LR_CONFIG = {
    "cnn_lstm": 3e-4,
    "gru_attention": 5e-4,
}

REPORT_THRESHOLD = 0.6

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


def train():
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <cnn_lstm | gru_attention>")
        sys.exit(1)

    MODEL_ID = sys.argv[1]

    if MODEL_ID not in ["cnn_lstm", "gru_attention"]:
        raise ValueError("Invalid model id")

    print("\n📂 Loading dataset...")
    X_train, y_train, X_val, y_val = load_dataset()
    time_steps = X_train.shape[1]
    feature_dim = X_train.shape[2]

    print(f"🧠 Input shape: (time_steps={time_steps}, feature_dim={feature_dim})")

    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {0: weights[0], 1: weights[1]}

    print("⚖ Class weights:", class_weights)

    if MODEL_ID == "cnn_lstm":
        model = build_cnn_lstm_model(
            time_steps=time_steps,
            feature_dim=feature_dim,
            lstm_units=96,
            dropout=0.5
        )
    else:
        model = build_gru_attention_model(
            time_steps=time_steps,
            feature_dim=feature_dim,
            gru_units=128,
            dropout=0.3
        )

    lr = LR_CONFIG[MODEL_ID]
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n🚀 Training {MODEL_ID.upper()} | LR = {lr}")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    np.save(
        os.path.join(HISTORY_DIR, f"{MODEL_ID}_history.npy"),
        history.history
    )

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.legend()
    plt.title(f"{MODEL_ID.upper()} Accuracy")
    plt.savefig(os.path.join(PLOTS_DIR, f"{MODEL_ID}_accuracy.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.legend()
    plt.title(f"{MODEL_ID.upper()} Loss")
    plt.savefig(os.path.join(PLOTS_DIR, f"{MODEL_ID}_loss.png"))
    plt.close()

    y_prob = model.predict(X_val).ravel()
    y_pred = (y_prob >= REPORT_THRESHOLD).astype(int)

    print("\n📊 Validation Metrics (reporting only)")
    print(f"Threshold : {REPORT_THRESHOLD}")
    print(f"Precision : {precision_score(y_val, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_val, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_val, y_prob):.4f}")

    model_path = os.path.join(MODEL_DIR, f"alz_{MODEL_ID}_final.keras")
    model.save(model_path)

    print(f"\n💾 Model saved to: {model_path}")
    print("✅ Training complete")


if __name__ == "__main__":
    train()
