import os
import json
import numpy as np
import tensorflow as tf
import librosa

from audio_preprocessing import preprocess_audio
from model_gru import AttentionLayer


TIME_STEPS = 300
FEATURE_DIM = 47  
MODEL_DIR = "models"
THRESHOLD_FILE = "thresholds/best_threshold.json"

def load_best_threshold():
    """Loads optimal threshold from training pipeline"""
    if os.path.exists(THRESHOLD_FILE):
        try:
            with open(THRESHOLD_FILE, "r") as f:
                data = json.load(f)
                return float(data["best_threshold"])
        except:
            pass
    return 0.4  

DECISION_THRESHOLD = load_best_threshold()

MODELS = {}


try:
    cnn_path = os.path.join(MODEL_DIR, "alz_cnn_lstm_final.keras")
    if os.path.exists(cnn_path):
        MODELS["cnn_lstm"] = tf.keras.models.load_model(cnn_path)
except Exception as e:
    print(f"⚠️ Warning: Could not load CNN-LSTM: {e}")

try:
    gru_path = os.path.join(MODEL_DIR, "alz_gru_attention_final.keras")
    if os.path.exists(gru_path):
        MODELS["gru_attention"] = tf.keras.models.load_model(
            gru_path,
            custom_objects={"AttentionLayer": AttentionLayer},
        )
except Exception as e:
    print(f"⚠️ Warning: Could not load GRU-Attention: {e}")


MODEL_ID_MAP = {
    "cnn": "cnn_lstm",
    "cnn_lstm": "cnn_lstm",
    "gru": "gru_attention",
    "gru_attention": "gru_attention",
}

def extract_fused_features(file_path):
    """
    Returns: X (1, 300, 47), prosody (7,)
    """
  
    y, sr, prosody = preprocess_audio(file_path)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40,
        n_fft=1024,
        hop_length=512
    ).T


    if mfcc.shape[0] > TIME_STEPS:
        mfcc = mfcc[:TIME_STEPS]
    else:
        pad_width = TIME_STEPS - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))

    prosody_rep = np.tile(prosody, (TIME_STEPS, 1))

    features = np.hstack([mfcc, prosody_rep]).astype(np.float32)
    
    return features[np.newaxis, ...], prosody


def predict_file(
    file_path: str,
    model_id: str = "gru_attention", 
    use_ensemble: bool = False,
):
    X, prosody = extract_fused_features(file_path)

    if X.shape[-1] != FEATURE_DIM:
        raise ValueError(f"Feature mismatch! Expected {FEATURE_DIM}, got {X.shape[-1]}")

    prob = 0.0
    
    if use_ensemble and len(MODELS) > 1:
        probs = [float(m.predict(X, verbose=0)[0][0]) for m in MODELS.values()]
        prob = float(np.mean(probs))
        used_model = "Ensemble"

    else:
      
        real_id = MODEL_ID_MAP.get(model_id, "gru_attention")
        
        if real_id not in MODELS:
            return {"error": f"Model {real_id} not loaded. Check 'models/' folder."}
            
        prob = float(MODELS[real_id].predict(X, verbose=0)[0][0])
        used_model = real_id

    label = "Alzheimer" if prob >= DECISION_THRESHOLD else "Control"

    return {
        "prediction": label,
        "probability": round(prob, 4),
        "threshold_used": DECISION_THRESHOLD,
        "model_used": used_model,
        "biomarkers": {
            "avg_pause_duration": float(prosody[0]),
            "intra_sentence_pauses": int(prosody[3]),
            "jitter": float(prosody[5]),
            "shimmer": float(prosody[6]),
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(predict_file(sys.argv[1]))