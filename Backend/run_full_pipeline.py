import os
import sys
import subprocess

# ==================================================
# Ensure backend directory is in PYTHONPATH
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ==================================================
# Helper to run steps
# ==================================================
def run_step(title, command):
    print("\n══════════════════════════════")
    print(f"▶ {title}")
    print("══════════════════════════════")

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        shell=True
    )

    if result.returncode != 0:
        print(f"❌ Failed at: {title}")
        sys.exit(1)

# ==================================================
# MAIN PIPELINE
# ==================================================
def main():
    print("🚀 Alzheimer Detection – Real-World Pipeline\n")

    # --------------------------------------------------
    # STEP 1: PROCESS REAL-WORLD DATA (ALWAYS RUN)
    # --------------------------------------------------
    run_step(
        "PROCESS REAL-WORLD AUDIO → FEATURES",
        "python data_pipeline.py"
    )

    # --------------------------------------------------
    # STEP 2: Train CNN-LSTM
    # --------------------------------------------------
    run_step(
        "TRAIN cnn_lstm",
        "python train_model.py cnn_lstm"
    )

    # --------------------------------------------------
    # STEP 3: Train GRU-Attention
    # --------------------------------------------------
    run_step(
        "TRAIN gru_attention",
        "python train_model.py gru_attention"
    )

    # --------------------------------------------------
    # STEP 4: Find optimal threshold (ensemble-based)
    # --------------------------------------------------
    run_step(
        "FIND OPTIMAL DECISION THRESHOLD",
        "python find_threshold.py"
    )

    # --------------------------------------------------
    # STEP 5: Evaluate CNN-LSTM
    # --------------------------------------------------
    run_step(
        "EVALUATE cnn_lstm",
        "python evaluate_models_multi.py cnn_lstm"
    )

    # --------------------------------------------------
    # STEP 6: Evaluate GRU-Attention
    # --------------------------------------------------
    run_step(
        "EVALUATE gru_attention",
        "python evaluate_models_multi.py gru_attention"
    )

    # --------------------------------------------------
    # STEP 7: Evaluate Ensemble
    # --------------------------------------------------
    run_step(
        "EVALUATE ensemble",
        "python evaluate_models_multi.py ensemble"
    )

    print("\n✅ REAL-WORLD PIPELINE COMPLETED SUCCESSFULLY")

# ==================================================
# ENTRY POINT
# ==================================================
if __name__ == "__main__":
    main()
