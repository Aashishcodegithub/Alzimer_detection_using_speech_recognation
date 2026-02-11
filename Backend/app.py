import os
import shutil
import tempfile
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Depends, Form
from fastapi.middleware.cors import CORSMiddleware

from auth import router as auth_router, get_current_user, User
from inference import predict_file
from gemini import router as gemini_router

app = FastAPI(title="Alzheimer Voice Lab API")


app.include_router(auth_router)
app.include_router(gemini_router)


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "Alzheimer Voice Lab API is running"}

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    model_id: str = Form("cnn_lstm"),
    use_ensemble: bool = Form(False),
    current_user: User = Depends(get_current_user),
):
    """
    model_id:
        - cnn_lstm
        - gru_attn / gru-attention
    use_ensemble:
        - true / false
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        result = predict_file(
            temp_path,
            model_id=model_id,
            use_ensemble=use_ensemble,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result


@app.post("/generate-report")
def generate_report(
    data: Dict,
    current_user: User = Depends(get_current_user),
):
    """
    Generates a human-readable AI diagnostic report
    from model prediction + extracted voice biomarkers.
    """

    probability = float(data.get("probability", 0.0))
    model_version = data.get("model_version", "v2.1.0")

    if probability < 0.3:
        risk_level = "Low"
        summary = (
            "The analyzed speech demonstrates stable fluency, minimal hesitation, "
            "and consistent vocal control. No significant Alzheimer-associated "
            "speech biomarkers were detected."
        )
    elif probability < 0.6:
        risk_level = "Moderate"
        summary = (
            "The speech sample shows mild hesitation and reduced fluency, which may "
            "indicate early cognitive stress. Further clinical evaluation is advised."
        )
    else:
        risk_level = "High"
        summary = (
            "The speech exhibits noticeable pauses, reduced phonation rate, and "
            "vocal instability. These patterns are commonly associated with "
            "Alzheimer-related cognitive impairment."
        )

    # ---------------- Report ----------------
    report = {
        "risk_level": risk_level,
        "confidence_percentage": round(probability * 100, 2),
        "model_version": model_version,
        "clinical_summary": summary,
        "disclaimer": (
            "This is an AI-assisted screening report generated from voice analysis. "
            "It is not a medical diagnosis and must be verified by a certified "
            "neurologist or healthcare professional."
        ),
    }

    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
