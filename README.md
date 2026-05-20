# JaundiScan — AI Jaundice Detection

A full-stack web application for jaundice detection from scleral (eye white) images using ResNet50 transfer learning.

**Stack:** Python · FastAPI · TensorFlow · Next.js · TypeScript

---

## Project Structure

```
jaundice-app/
├── backend/
│   ├── main.py              # FastAPI app with /predict endpoint
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── app/
    │   │   ├── layout.tsx
    │   │   ├── page.tsx
    │   │   └── globals.css
    │   └── components/
    │       ├── JaundiceScan.tsx
    │       └── JaundiceScan.module.css
    ├── package.json
    ├── tsconfig.json
    ├── next.config.js
    └── .env.example
```

---

## Setup

### Step 1 — Export Your Trained Model

At the end of your notebook, add:

```python
model.save("jaundice_model.h5")
```

Then copy `jaundice_model.h5` into the `backend/` folder.

---

### Step 2 — Run the Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env file
cp .env.example .env

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at: http://localhost:8000  
API docs at: http://localhost:8000/docs

---

### Step 3 — Run the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Copy env file
cp .env.example .env.local

# Start dev server
npm run dev
```

Frontend runs at: http://localhost:3000

---

## API Endpoints

### `POST /predict`
Upload a scleral image and get a jaundice prediction.

**Request:** `multipart/form-data` with `file` field (image)

**Response:**
```json
{
  "prediction": "Jaundice",
  "confidence": 87.43,
  "probability": 0.8743,
  "risk_level": "High",
  "message": "Jaundice indicators detected. Please consult a doctor immediately."
}
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

---

## Deployment

### Backend → Render (Free Tier)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo, select `backend/` as root directory
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port 8000`
5. Upload `jaundice_model.h5` as a persistent disk or host it on Google Drive / S3 and download on startup

### Frontend → Vercel (Free)

1. Go to [vercel.com](https://vercel.com) → New Project
2. Connect GitHub repo, select `frontend/` as root directory
3. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = your Render backend URL
4. Deploy

---

## Resume Bullets (SDE-ready)

- Developed and deployed an end-to-end **medical image classification web application** using ResNet50 transfer learning, served via a **FastAPI REST endpoint** integrated with a **Next.js** frontend for real-time scleral image uploads.
- Implemented a two-phase fine-tuning pipeline with YCbCr skin segmentation preprocessing, data augmentation, and confidence-score-based threshold optimization achieving **86% validation accuracy** on a class-imbalanced dataset.
- **Tech Stack:** Python, TensorFlow, Keras, FastAPI, REST API, Next.js, TypeScript, Docker, Render, Vercel

---

## Disclaimer

This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis.
