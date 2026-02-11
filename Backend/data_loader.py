import os
import numpy as np

PROCESSED_DIR = "data/processed"

X_TRAIN_FILE = os.path.join(PROCESSED_DIR, "X_train.npy")
X_VAL_FILE   = os.path.join(PROCESSED_DIR, "X_val.npy")
y_TRAIN_FILE = os.path.join(PROCESSED_DIR, "y_train.npy")
y_VAL_FILE   = os.path.join(PROCESSED_DIR, "y_val.npy")


def load_dataset(test_only=False):
    """
    Returns:
        If test_only=False:
            X_train, y_train, X_val, y_val

        If test_only=True:
            None, None, X_val, y_val
    """

    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(
            "Processed data not found. Run data_pipeline.py first."
        )

    X_val = np.load(X_VAL_FILE)
    y_val = np.load(y_VAL_FILE)

    if test_only:
        return None, None, X_val, y_val

    X_train = np.load(X_TRAIN_FILE)
    y_train = np.load(y_TRAIN_FILE)

    assert X_train.ndim == 3, "X_train must be 3D (samples, time, features)"
    assert X_val.ndim == 3, "X_val must be 3D (samples, time, features)"

    assert X_train.shape[2] == X_val.shape[2], "Feature dim mismatch"
    assert X_train.shape[1] == X_val.shape[1], "Time steps mismatch"

    print("📊 Dataset Loaded Successfully")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Feature dim: {X_train.shape[2]}")

    return X_train, y_train, X_val, y_val
