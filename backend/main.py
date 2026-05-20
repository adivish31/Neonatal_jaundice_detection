from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import uvicorn

app = FastAPI(title="Jaundice Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-deployed-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model("jaundice_model.h5")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")


def ycbcr_skin_preprocess(img: np.ndarray) -> np.ndarray:
    """
    YCbCr skin segmentation (Patravali et al. 2014)
    Same preprocessing used during training.
    """
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


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_preprocessed = ycbcr_skin_preprocess(img_array)
    return np.expand_dims(img_preprocessed, axis=0)


@app.get("/")
async def root():
    return {"message": "Jaundice Detection API is running", "status": "healthy"}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        prob = float(model.predict(img_array, verbose=0)[0][0])
        prediction = "Jaundice" if prob >= 0.5 else "Normal"
        confidence = round(prob * 100, 2) if prediction == "Jaundice" else round((1 - prob) * 100, 2)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probability": round(prob, 4),
            "risk_level": "High" if prob >= 0.7 else "Moderate" if prob >= 0.5 else "Low",
            "message": (
                "Jaundice indicators detected. Please consult a doctor immediately."
                if prediction == "Jaundice"
                else "No jaundice indicators detected. Eyes appear normal."
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
