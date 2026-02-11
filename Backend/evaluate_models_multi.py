import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

from model_gru import AttentionLayer
from data_loader import load_dataset

MODEL_DIR = "models"
PLOTS_DIR = "plots"
THRESHOLD_FILE = "thresholds/best_threshold.json"


plt.style.use('seaborn-v0_8-muted')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_threshold():
    if not os.path.exists(THRESHOLD_FILE):
        print(f"⚠️ Warning: {THRESHOLD_FILE} not found. Using default 0.5")
        return 0.5

    with open(THRESHOLD_FILE, "r") as f:
        data = json.load(f)

    return float(data["best_threshold"])

DECISION_THRESHOLD = load_threshold()

def load_model(model_id):
    model_path = f"{MODEL_DIR}/alz_{model_id}_final.keras"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file {model_path} not found. Did you run train_model.py?")
        sys.exit(1)

    print(f"🔹 Loading {model_id} from {model_path}...")
    if model_id == "gru_attention":
        return tf.keras.models.load_model(
            model_path,
            custom_objects={"AttentionLayer": AttentionLayer},
        )

    return tf.keras.models.load_model(model_path)



def plot_threshold_analysis(y_true, y_prob, name):
    """
    Generates the requested Threshold vs F1 & Recall Curve
    """
    thresholds = np.arange(0.1, 0.95, 0.02)
    precisions = []
    recalls = []
    f1_scores = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1_scores.append(f1_score(y_true, preds, zero_division=0))

 
    best_idx = np.argmax(f1_scores)
    best_t = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    plt.figure(figsize=(10, 6))
    

    plt.plot(thresholds, f1_scores, label='F1 Score', color='blue', linewidth=2.5)
    plt.plot(thresholds, recalls, label='Recall', color='orange', linestyle='--', linewidth=2)
    plt.plot(thresholds, precisions, label='Precision', color='green', linestyle=':', alpha=0.7)

    plt.axvline(best_t, color='red', linestyle='-.', alpha=0.5, label=f'Best Threshold ({best_t:.2f})')
    plt.scatter([best_t], [best_f1], color='red', zorder=5)
    plt.text(best_t, best_f1 + 0.02, f"Max F1: {best_f1:.3f}", ha='center', fontweight='bold', color='red')

    plt.title(f"{name.upper()} - Threshold Optimization Curve", fontsize=14)
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    save_path = f"{PLOTS_DIR}/{name}_threshold_analysis.png"
    plt.savefig(save_path)
    print(f"📊 Threshold Curve saved to {save_path}")
    plt.close()

def plot_roc(y_true, y_prob, name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color='purple', lw=2)
    plt.plot([0, 1], [0, 1], "--", color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOTS_DIR}/{name}_roc.png")
    plt.close()

def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'Alzheimer'], 
                yticklabels=['Control', 'Alzheimer'])
    plt.title(f"{name} Confusion Matrix (Thresh={DECISION_THRESHOLD})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"{PLOTS_DIR}/{name}_confusion.png")
    plt.close()

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "y_pred": y_pred,
    }
def evaluate_model(model_id, X, y):
    model = load_model(model_id)
    
    if model.input_shape[-1] != X.shape[-1]:
        print(f"❌ Shape Mismatch! Model expects {model.input_shape[-1]} features, but data has {X.shape[-1]}.")
        print("👉 Please retrain your models using 'python train_model.py'")
        sys.exit(1)

    print(f"🧠 Predicting with {model_id}...")
    y_prob = model.predict(X, verbose=1).ravel()

    metrics = compute_metrics(y, y_prob)

    print(f"\n✅ RESULTS — {model_id.upper()}")
    print(f"Threshold : {DECISION_THRESHOLD}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")

    plot_roc(y, y_prob, model_id)
    plot_confusion(y, metrics["y_pred"], model_id)
    plot_threshold_analysis(y, y_prob, model_id)
    return y_prob

def evaluate_ensemble(X, y):
    print("👥 Loading Ensemble Models...")
    cnn = load_model("cnn_lstm")
    gru = load_model("gru_attention")

    if cnn.input_shape[-1] != X.shape[-1]:
        print(f"❌ Shape Mismatch! Data has {X.shape[-1]} features.")
        sys.exit(1)

    print("🧠 Computing Ensemble Predictions...")
    pred_cnn = cnn.predict(X, verbose=1).ravel()
    pred_gru = gru.predict(X, verbose=1).ravel()
    
    prob = (pred_cnn + pred_gru) / 2

    metrics = compute_metrics(y, prob)

    print("\n✅ RESULTS — ENSEMBLE")
    print(f"Threshold : {DECISION_THRESHOLD}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")

    plot_roc(y, prob, "ensemble")
    plot_confusion(y, metrics["y_pred"], "ensemble")
    plot_threshold_analysis(y, prob, "ensemble") 

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_models_multi.py <cnn_lstm|gru_attention|ensemble>")
        sys.exit(1)

    model_id = sys.argv[1]

    print("📂 Loading validation/test dataset...")
    _, _, X_test, y_test = load_dataset() 
    
    print(f"🔍 Data Shape: {X_test.shape}")
    
    if model_id == "ensemble":
        evaluate_ensemble(X_test, y_test)
    else:
        evaluate_model(model_id, X_test, y_test)

if __name__ == "__main__":
    main()