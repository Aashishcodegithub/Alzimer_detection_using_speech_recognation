import os
import random
import tempfile
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split

from audio_preprocessing import preprocess_audio

RAW_DIR = "data/raw_real"
OUT_DIR = "data/processed"

SAMPLE_RATE = 16000
MAX_TIME_STEPS = 300
AUG_PER_FILE = 13

os.makedirs(OUT_DIR, exist_ok=True)

def augment_audio(y, sr):
    """Safe audio augmentation (raw waveform)"""

    # Gaussian noise
    if random.random() < 0.6:
        y = y + np.random.normal(0, 0.005, len(y))

    # Pitch shift
    if random.random() < 0.5:
        y = librosa.effects.pitch_shift(
            y=y, sr=sr, n_steps=random.uniform(-2, 2)
        )

    # Time stretch
    if random.random() < 0.5:
        y = librosa.effects.time_stretch(
            y=y, rate=random.uniform(0.9, 1.1)
        )

    return np.nan_to_num(y)

def extract_feature_matrix(file_path):
    """
    Returns shape: (MAX_TIME_STEPS, 45)
    """
    y, sr, prosody = preprocess_audio(file_path)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512
    )

    mfcc = mfcc.T
    if mfcc.shape[0] > MAX_TIME_STEPS:
        mfcc = mfcc[:MAX_TIME_STEPS]
    else:
        mfcc = np.pad(
            mfcc,
            ((0, MAX_TIME_STEPS - mfcc.shape[0]), (0, 0))
        )

    prosody_rep = np.repeat(
        prosody[np.newaxis, :],
        MAX_TIME_STEPS,
        axis=0
    )

    return np.hstack([mfcc, prosody_rep]).astype(np.float32)

def process_class(class_name, label):
    X, y = [], []
    class_dir = os.path.join(RAW_DIR, class_name)

    for fname in os.listdir(class_dir):
        if not fname.endswith(".wav"):
            continue

        file_path = os.path.join(class_dir, fname)

    
        X.append(extract_feature_matrix(file_path))
        y.append(label)

        y_raw, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        for _ in range(AUG_PER_FILE):
            aug_audio = augment_audio(y_raw, sr)
            tmp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp_path = tmp.name

                sf.write(tmp_path, aug_audio, sr)
                X.append(extract_feature_matrix(tmp_path))
                y.append(label)

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    return X, y

def main():
    X, y = [], []

    for cls, label in [("control", 0), ("alzheimer", 1)]:
        Xi, yi = process_class(cls, label)
        X.extend(Xi)
        y.extend(yi)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    np.save(f"{OUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUT_DIR}/y_val.npy", y_val)

    print("✅ Dataset ready")
    print("Train:", X_train.shape, "Val:", X_val.shape)

if __name__ == "__main__":
    main()
