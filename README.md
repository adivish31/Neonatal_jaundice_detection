# JaundiScan — AI Neonatal Jaundice Detection with MLOps

A production-grade medical image classification system for neonatal jaundice detection from scleral (eye white) images, built with a full MLOps pipeline including experiment tracking, automated retraining, data drift monitoring, and containerized deployment.

**Stack:** Python · TensorFlow · FastAPI · Next.js · MLflow · PostgreSQL · Docker · Nginx · GitHub Actions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION STACK                            │
│                                                                     │
│  ┌──────────┐     ┌───────────────┐     ┌──────────────────────┐   │
│  │  Vercel   │     │    Nginx      │     │   GitHub Actions     │   │
│  │ (Frontend)│────▶│ Reverse Proxy │     │  (Scheduled Retrain) │   │
│  │  Next.js  │     │   Port 80     │     │   Weekly cron job    │   │
│  └──────────┘     └───────┬───────┘     └──────────┬───────────┘   │
│                           │                         │               │
│              ┌────────────┼────────────┐            │               │
│              │            │            │            │               │
│              ▼            ▼            ▼            ▼               │
│  ┌───────────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐        │
│  │   FastAPI      │ │  MLflow   │ │PostgreSQL│ │  train.py │        │
│  │   Backend      │ │  Tracking │ │  Logging │ │  + MLflow │        │
│  │                │ │  Server   │ │          │ │           │        │
│  │ /predict       │ │  Port 5000│ │ Port 5432│ │ Compares  │        │
│  │ /health        │ │           │ │          │ │ & promotes│        │
│  │ /drift         │ │ Params    │ │ Pred logs│ │ best model│        │
│  │ /logs          │ │ Metrics   │ │ Drift    │ │           │        │
│  │                │ │ Models    │ │ flags    │ │           │        │
│  │ Port 8000      │ │           │ │          │ │           │        │
│  └───────┬────────┘ └───────────┘ └──────────┘ └──────────┘        │
│          │                                                          │
│          ▼                                                          │
│  ┌───────────────┐                                                  │
│  │  TensorFlow   │                                                  │
│  │  ResNet50     │                                                  │
│  │  Model (.h5)  │                                                  │
│  └───────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
jaundice-app/
├── backend/
│   ├── main.py                # FastAPI app (/predict, /health, /drift, /logs)
│   ├── models.py              # SQLAlchemy models for prediction logging
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Backend container
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── app/               # Next.js pages
│   │   └── components/        # React components
│   ├── Dockerfile             # Frontend container
│   ├── vercel.json            # Vercel deployment config
│   └── package.json
├── mlflow/
│   └── Dockerfile             # MLflow tracking server container
├── nginx/
│   └── nginx.conf             # Reverse proxy configuration
├── notebook/
│   └── jaundice_resnet50.ipynb # Original training notebook
├── .github/
│   └── workflows/
│       └── mlops.yml          # Scheduled retrain & deploy pipeline
├── train.py                   # MLflow-integrated training script
├── run_experiments.py         # Multi-experiment runner (4 configs)
├── drift_detector.py          # Standalone drift detection module
├── docker-compose.yml         # Full stack orchestration
├── baseline_metrics.json      # Current best model's metrics (auto-generated)
└── README.md
```

---

## Quick Start

### Option 1: Docker Compose (Full Stack)

```bash
# Clone the repository
git clone https://github.com/your-username/jaundice-app.git
cd jaundice-app

# Place your trained model in backend/
cp path/to/jaundice_model.h5 backend/

# Launch everything
docker compose up --build
```

| Service   | URL                       |
|-----------|---------------------------|
| Frontend  | http://localhost:3000      |
| Backend   | http://localhost:8000      |
| API Docs  | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000      |
| Nginx     | http://localhost:80        |

### Option 2: Local Development

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

---

## API Endpoints

### `POST /predict`
Upload a neonatal scleral image and get a jaundice prediction with drift monitoring.

**Request:** `multipart/form-data` with `file` field (image, max 10 MB)

**Response:**
```json
{
  "prediction": "Jaundice",
  "confidence": 87.43,
  "probability": 0.8743,
  "risk_level": "High",
  "drift_warning": "Normal",
  "latency_ms": 142.5,
  "message": "Jaundice indicators detected. Please consult a doctor immediately."
}
```

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "mlflow_uri": "http://mlflow:5000",
  "database": "postgresql://..."
}
```

### `GET /drift`
Returns a drift detection report based on recent predictions.
```json
{
  "status": "✅ Stable",
  "total_predictions_analyzed": 100,
  "drift_rate_percent": 3.0,
  "avg_brightness": 132.45,
  "std_brightness": 28.7,
  "prediction_distribution": { "Normal": 67, "Jaundice": 33 }
}
```

### `GET /logs?limit=20&prediction=Jaundice`
Query recent prediction logs for monitoring and auditing.

---

## MLOps Pipeline

### 1. Experiment Tracking (MLflow)

Run 4 experiments with different hyperparameters:
```bash
python run_experiments.py
```

Or run a single training with custom params:
```bash
python train.py --epochs 30 --lr 0.0005 --batch-size 16 --dropout 0.4
```

The training script automatically:
- Logs all parameters, metrics, and the model to MLflow
- Compares the new model against the current registered best
- Only promotes the new model if `val_accuracy` improves
- Saves `baseline_metrics.json` for drift detection

### 2. Prediction Logging (PostgreSQL)

Every `/predict` call logs:
- Timestamp, prediction result, probability, confidence
- Risk level, drift flag, average image brightness

### 3. Drift Detection

**Real-time:** Each prediction checks image brightness against training-time ranges.

**Batch:** Run the standalone detector:
```bash
python drift_detector.py
```

**API:** Query `/drift` for a live drift report.

### 4. Scheduled Retraining (GitHub Actions)

The `.github/workflows/mlops.yml` workflow runs every Sunday at 02:00 UTC:
1. **Retrain** the model with the latest data
2. **Compare** new accuracy against the current registered model
3. **Promote** only if the new model is better
4. **Deploy** by committing the updated model and triggering Render

Can also be triggered manually via `workflow_dispatch`.

---

## Deployment

### Frontend → Vercel (Free)

1. Go to [vercel.com](https://vercel.com) → New Project
2. Connect GitHub repo, select `frontend/` as root directory
3. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = your backend URL (e.g. Render)
4. Deploy

### Backend → Render (Free Tier)

1. Go to [render.com](https://render.com) → New Web Service
2. Connect GitHub repo, select `backend/` as root directory
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port 8000`
4. Add environment variables:
   - `DATABASE_URL` = your PostgreSQL connection string (Neon.tech / Supabase)
   - `MLFLOW_TRACKING_URI` = your MLflow server URL (DagsHub)
5. Upload `jaundice_model.h5` via persistent disk or download on startup

### Database → Neon.tech or Supabase (Free)

Create a free PostgreSQL database and use its connection string as `DATABASE_URL`.

### MLflow → DagsHub (Free)

Use [DagsHub](https://dagshub.com) for a free hosted MLflow tracking server.

---

## Metrics

| Metric               | Value         |
|----------------------|---------------|
| Dataset Size         | ~1,200 images |
| Class Split          | 75% Normal, 25% Jaundice |
| Validation Accuracy  | 86%           |
| Sensitivity (Recall) | 94%           |
| Validation Loss      | 0.34          |
| API Latency (p50)    | ~150 ms       |
| Model Size           | ~98 MB (.h5)  |
| Training Time        | ~8 min (GPU)  |

> **Baseline Metric:** 86% validation accuracy and **94% sensitivity (recall)** on a 3:1 imbalanced binary classification task using ResNet50 transfer learning with YCbCr skin segmentation preprocessing.

---

## Resume Bullets

- Built an end-to-end **MLOps pipeline** for a medical image classification system: MLflow experiment tracking, PostgreSQL prediction logging, automated drift detection, and scheduled retraining via GitHub Actions.
- Trained and optimized a **TensorFlow/ResNet50** model using YCbCr skin segmentation, achieving **86% overall accuracy and 94% sensitivity (recall)** on a 3:1 class-imbalanced dataset, prioritizing false negative reduction for neonatal screening.
- Containerized a **FastAPI + TensorFlow** backend with **Docker Compose** (5 services: API, Next.js frontend, PostgreSQL, MLflow, Nginx reverse proxy) for single-command deployment.
- Implemented a **CI/CD retraining pipeline** that automatically compares new model accuracy against the registered baseline and promotes only if performance improves, reducing manual intervention to zero.
- **Tech Stack:** Python, TensorFlow, Keras, FastAPI, MLflow, PostgreSQL, SQLAlchemy, Docker, Nginx, Next.js, TypeScript, GitHub Actions, Vercel, Render

---

## Done Criteria Checklist

- [x] Runs from a fresh clone with `docker compose up`
- [x] Every tool can be explained without notes
- [x] At least one number is real and measured (86% accuracy, ~150ms latency)
- [x] MLflow tracking with 4+ experiments
- [x] Docker on Render/EC2 with Nginx
- [x] Scheduled retrain via GitHub Actions
- [x] Prediction logging to PostgreSQL
- [x] Drift detection (brightness-based + prediction distribution)

---

## Disclaimer

This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis or clinical screening of newborns.
