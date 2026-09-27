"""
drift_detector.py
─────────────────
Standalone drift detection module.

Compares the distribution of incoming prediction images (logged in PostgreSQL)
against training-time baseline statistics. Flags drift when distributions shift
significantly.

Can be run as a standalone script (e.g., via cron) or imported into the API.

Usage:
    python drift_detector.py
"""

import os
import json
import numpy as np
from sqlalchemy import create_engine, text

# ── Configuration ──
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mlops.db")
BASELINE_FILE = os.getenv("BASELINE_FILE", "baseline_metrics.json")

# Training-time brightness statistics (measured from training dataset)
# These should be updated after each retraining run
TRAINING_BRIGHTNESS_MEAN = 128.0
TRAINING_BRIGHTNESS_STD = 35.0

# Drift thresholds
Z_SCORE_THRESHOLD = 2.5          # flag if incoming mean is > 2.5 std devs away
DRIFT_SAMPLE_WINDOW = 100        # look at the last N predictions


def load_baseline():
    """Load baseline metrics from file."""
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            return json.load(f)
    return None


def fetch_recent_brightness(engine, window: int = DRIFT_SAMPLE_WINDOW):
    """Fetch the avg_brightness of the last N predictions from the database."""
    query = text(f"""
        SELECT avg_brightness
        FROM prediction_logs
        WHERE avg_brightness IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT :window
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"window": window}).fetchall()
    return [row[0] for row in result]


def compute_drift_score(recent_values: list) -> dict:
    """
    Compare the mean of recent incoming brightness values
    against the training-time distribution using a Z-score.
    """
    if len(recent_values) < 10:
        return {
            "status": "Insufficient Data",
            "message": f"Only {len(recent_values)} samples available (need >= 10)",
            "z_score": None,
            "incoming_mean": None,
            "training_mean": TRAINING_BRIGHTNESS_MEAN,
        }

    incoming_mean = np.mean(recent_values)
    incoming_std = np.std(recent_values)

    # Z-score of the incoming mean relative to training distribution
    z_score = abs(incoming_mean - TRAINING_BRIGHTNESS_MEAN) / TRAINING_BRIGHTNESS_STD

    if z_score > Z_SCORE_THRESHOLD:
        status = "🚨 DRIFT DETECTED"
        message = (
            f"Incoming image brightness (mean={incoming_mean:.1f}) has shifted "
            f"significantly from training distribution (mean={TRAINING_BRIGHTNESS_MEAN:.1f}). "
            f"Z-score: {z_score:.2f} > threshold {Z_SCORE_THRESHOLD}. "
            f"Consider retraining the model."
        )
    else:
        status = "✅ No Drift"
        message = (
            f"Incoming image brightness (mean={incoming_mean:.1f}) is within "
            f"acceptable range of training distribution (mean={TRAINING_BRIGHTNESS_MEAN:.1f}). "
            f"Z-score: {z_score:.2f}"
        )

    return {
        "status": status,
        "message": message,
        "z_score": round(z_score, 4),
        "incoming_mean": round(incoming_mean, 2),
        "incoming_std": round(incoming_std, 2),
        "training_mean": TRAINING_BRIGHTNESS_MEAN,
        "training_std": TRAINING_BRIGHTNESS_STD,
        "sample_count": len(recent_values),
    }


def check_prediction_distribution(engine, window: int = DRIFT_SAMPLE_WINDOW) -> dict:
    """
    Check if the prediction distribution itself has shifted
    (e.g., sudden spike in jaundice predictions).
    """
    query = text(f"""
        SELECT prediction, COUNT(*) as cnt
        FROM prediction_logs
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        GROUP BY prediction
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        
        counts = {row[0]: row[1] for row in result}
        total = sum(counts.values())
        
        if total == 0:
            return {"status": "No predictions in last 24h", "distribution": {}}
        
        distribution = {k: round(v / total, 4) for k, v in counts.items()}
        
        # Flag if jaundice rate is unusually high (>80%) or low (<5%)
        jaundice_rate = distribution.get("Jaundice", 0)
        if jaundice_rate > 0.8:
            alert = "⚠️ Unusually high jaundice detection rate"
        elif jaundice_rate < 0.05 and total > 20:
            alert = "⚠️ Unusually low jaundice detection rate"
        else:
            alert = "Normal"
        
        return {
            "status": alert,
            "total_predictions_24h": total,
            "distribution": distribution,
        }
    except Exception as e:
        return {"status": f"Error: {str(e)}", "distribution": {}}


def main():
    print("=" * 60)
    print("  DRIFT DETECTION REPORT")
    print("=" * 60)

    engine = create_engine(DATABASE_URL)

    # 1. Brightness drift check
    print("\n📊 Feature Drift (Image Brightness):")
    recent_brightness = fetch_recent_brightness(engine)
    brightness_result = compute_drift_score(recent_brightness)
    for k, v in brightness_result.items():
        print(f"   {k}: {v}")

    # 2. Prediction distribution check
    print("\n📊 Prediction Distribution Drift:")
    pred_result = check_prediction_distribution(engine)
    for k, v in pred_result.items():
        print(f"   {k}: {v}")

    # 3. Baseline comparison
    print("\n📊 Baseline Model Performance:")
    baseline = load_baseline()
    if baseline:
        print(f"   Best Validation Accuracy: {baseline.get('best_val_accuracy')}")
        print(f"   Best Run ID: {baseline.get('best_run_id')}")
    else:
        print("   No baseline_metrics.json found. Run run_experiments.py first.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
