from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import uvicorn
import mlflow
import os
import time
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import SessionLocal, PredictionLog

# ── App Setup ──
app = FastAPI(
    title="JaundiScan API",
    description="Neonatal Jaundice Detection with MLOps monitoring",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        os.getenv("FRONTEND_URL", "http://localhost:3000"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Database Dependency ──
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Model Loading ──
model = None


@app.on_event("startup")
async def load_model():
    global model
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            print(f"MLflow tracking URI set to: {mlflow_uri}")

        model_path = os.getenv("MODEL_PATH", "jaundice_model.h5")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model file '{model_path}' not found. /predict will return 503.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")


# ── Preprocessing (same as training pipeline) ──
def ycbcr_skin_preprocess(img: np.ndarray) -> np.ndarray:
    """YCbCr skin segmentation — same preprocessing used during training."""
    img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    _, Cr, Cb = cv2.split(ycbcr)
    mask_cb = cv2.inRange(Cb, 77, 127)
    mask_cr = cv2.inRange(Cr, 133, 173)
    mask = cv2.bitwise_and(mask_cb, mask_cr)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    img_masked = np.where(mask_3ch > 0, img, 0).astype(np.float32)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_masked)
    return img_preprocessed


def calculate_brightness(img_array: np.ndarray) -> float:
    """Calculate average brightness (V channel of HSV) as a drift proxy."""
    hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 2]))


def check_drift(brightness: float) -> str:
    """Heuristic drift check: flag if brightness is far outside training range."""
    # Training images typically have brightness between 80–180
    if brightness < 50 or brightness > 220:
        return "Drift Detected (Lighting anomaly)"
    return "Normal"


def preprocess_image(image_bytes: bytes) -> tuple:
    """Load image bytes, resize, preprocess, and compute brightness."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    brightness = calculate_brightness(img_array)
    img_preprocessed = ycbcr_skin_preprocess(img_array)
    return np.expand_dims(img_preprocessed, axis=0), brightness


# ── Endpoints ──

@app.get("/")
async def root():
    return {"message": "JaundiScan API is running", "status": "healthy", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "not configured"),
        "database": os.getenv("DATABASE_URL", "sqlite (fallback)"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a neonatal scleral image and get a jaundice prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    # Input validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Must be an image (JPEG, PNG, etc.).",
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        start = time.time()

        img_array, brightness = preprocess_image(contents)
        drift_status = check_drift(brightness)

        prob = float(model.predict(img_array, verbose=0)[0][0])
        prediction = "Jaundice" if prob >= 0.5 else "Normal"
        confidence = round(prob * 100, 2) if prediction == "Jaundice" else round((1 - prob) * 100, 2)
        risk_level = "High" if prob >= 0.7 else "Moderate" if prob >= 0.5 else "Low"

        latency_ms = round((time.time() - start) * 1000, 2)

        # Log to database
        try:
            db_log = PredictionLog(
                prediction=prediction,
                probability=prob,
                confidence=confidence,
                risk_level=risk_level,
                drift_flag=drift_status,
                avg_brightness=brightness,
            )
            db.add(db_log)
            db.commit()
        except Exception as db_err:
            print(f"⚠️ DB logging failed (non-fatal): {db_err}")
            db.rollback()

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probability": round(prob, 4),
            "risk_level": risk_level,
            "drift_warning": drift_status,
            "latency_ms": latency_ms,
            "message": (
                "Jaundice indicators detected. Please consult a doctor immediately."
                if prediction == "Jaundice"
                else "No jaundice indicators detected. Eyes appear normal."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/drift")
async def drift_report(db: Session = Depends(get_db)):
    """Get a drift detection report based on recent predictions."""
    try:
        # Last 100 predictions
        recent = (
            db.query(PredictionLog)
            .order_by(PredictionLog.timestamp.desc())
            .limit(100)
            .all()
        )

        if len(recent) < 5:
            return {
                "status": "Insufficient Data",
                "message": f"Only {len(recent)} predictions logged. Need at least 5.",
            }

        brightness_values = [r.avg_brightness for r in recent if r.avg_brightness is not None]
        drift_flags = [r.drift_flag for r in recent]

        drift_count = sum(1 for f in drift_flags if f and "Drift" in f)
        drift_rate = round(drift_count / len(drift_flags) * 100, 2)

        avg_brightness = round(np.mean(brightness_values), 2) if brightness_values else None
        std_brightness = round(np.std(brightness_values), 2) if brightness_values else None

        # Prediction distribution
        pred_counts = {}
        for r in recent:
            pred_counts[r.prediction] = pred_counts.get(r.prediction, 0) + 1

        return {
            "status": "🚨 Drift Alert" if drift_rate > 20 else "✅ Stable",
            "total_predictions_analyzed": len(recent),
            "drift_rate_percent": drift_rate,
            "avg_brightness": avg_brightness,
            "std_brightness": std_brightness,
            "prediction_distribution": pred_counts,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")


@app.get("/logs")
async def prediction_logs(
    limit: int = Query(default=20, le=100, ge=1),
    prediction: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """Query recent prediction logs for monitoring."""
    try:
        query = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc())

        if prediction:
            query = query.filter(PredictionLog.prediction == prediction)

        logs = query.limit(limit).all()

        return {
            "count": len(logs),
            "logs": [
                {
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "prediction": log.prediction,
                    "probability": log.probability,
                    "confidence": log.confidence,
                    "risk_level": log.risk_level,
                    "drift_flag": log.drift_flag,
                    "avg_brightness": log.avg_brightness,
                }
                for log in logs
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
