# api.py
# ------------------------------------------------------------
# Flask REST API for Anomaly Scoring (Standardized Service)
# Endpoints:
#   - GET /health       -> {status, model_loaded, timestamp}
#   - GET /model/info   -> {model_type, feature_names, params}
#   - GET /metrics      -> {requests_total, 4xx, 5xx, avg_latency_ms, preds}
#   - POST /predict     -> JSON {temperature, humidity, sound_volume}
#                          => {timestamp, is_anomaly, anomaly_score}
# Features:
#   - Loads saved model at startup (Flask 3.x compatible)
#   - Minimal built-in metrics counters + avg latency
#   - CSV prediction log for live demos: app/outputs/predictions_log.csv
# Usage:
#   - Local:  `python -m app.api`
#   - Docker: image ENTRYPOINT runs this module
# Notes:
#   - Uses features required by Task 1: temperature, humidity, sound_volume
# ------------------------------------------------------------

from __future__ import annotations

import os
import csv
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any

import pandas as pd
from flask import Flask, request, jsonify

from app.model import TurbineAnomalyDetector

# --- Config / Paths ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/turbine_iforest.pkl")
OUT_DIR = "app/outputs"
LOG_PATH = os.path.join(OUT_DIR, "predictions_log.csv")

# --- App + Model ---
app = Flask(__name__)
det = TurbineAnomalyDetector()
MODEL_LOADED = False

# --- Minimal in-memory metrics ---
METRICS = defaultdict(int)  # keys: requests, 4xx, 5xx, preds
LAT_SUM = 0.0               # total seconds across requests


def _ensure_model() -> None:
    """Try to load the trained model from disk."""
    global MODEL_LOADED
    try:
        det.load(MODEL_PATH)
        MODEL_LOADED = True
    except Exception:
        MODEL_LOADED = False


def _log_prediction(row: Dict[str, Any], is_anom: bool, score: float) -> None:
    """Append a single inference to CSV for live visibility during the talk."""
    os.makedirs(OUT_DIR, exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "temperature", "humidity",
                "sound_volume", "is_anomaly", "anomaly_score"
            ])
        w.writerow([
            datetime.utcnow().isoformat(),
            row["temperature"], row["humidity"], row["sound_volume"],
            int(is_anom), float(score)
        ])


# Load the model once at import time (Flask 3.x friendly; no before_first_request)
_ensure_model()


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.get("/model/info")
def model_info():
    if not MODEL_LOADED:
        return jsonify({"error": "model not loaded"}), 500
    return jsonify({
        "model_type": "IsolationForest",
        "feature_names": det.feature_names,
        "contamination": det.contamination,
        "n_estimators": det.n_estimators,
        "random_state": det.random_state
    })


@app.get("/metrics")
def metrics():
    avg_ms = (LAT_SUM / max(1, METRICS["requests"])) * 1000.0
    return jsonify({
        "requests_total": METRICS["requests"],
        "requests_4xx": METRICS["4xx"],
        "requests_5xx": METRICS["5xx"],
        "predictions_total": METRICS["preds"],
        "avg_latency_ms": round(avg_ms, 3)
    })


@app.post("/predict")
def predict():
    start = time.perf_counter()
    METRICS["requests"] += 1
    try:
        if not MODEL_LOADED:
            METRICS["5xx"] += 1
            return jsonify({"error": "model not loaded"}), 500

        payload = request.get_json(silent=True)
        if payload is None:
            raw = request.get_data(cache=False, as_text=True)
            try:
                import json as _json
                payload = _json.loads(raw)
            except Exception:
                payload = {}

        # Optional: accept simple form posts too
        if not payload and request.form:
            payload = request.form.to_dict(flat=True)
            for k in payload:
                try: payload[k] = float(payload[k])
                except: pass

        missing = [f for f in det.feature_names if f not in payload]
        if missing:
            METRICS["4xx"] += 1
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Enforce feature order
        df = pd.DataFrame([payload], columns=det.feature_names)
        res = det.predict(df)
        is_anom = bool(res["is_anomaly"][0])
        score = float(res["anomaly_score"][0])
        METRICS["preds"] += 1

        # Optional CSV log for presentation
        _log_prediction(payload, is_anom, score)

        return jsonify({
            "timestamp": datetime.utcnow().isoformat(),
            "is_anomaly": is_anom,
            "anomaly_score": score
        })
    except Exception:
        METRICS["5xx"] += 1
        return jsonify({"error": "internal error"}), 500
    finally:
        global LAT_SUM
        LAT_SUM += (time.perf_counter() - start)


if __name__ == "__main__":
    # If the model file is missing, train once and load to keep the demo smooth.
    if not os.path.exists(MODEL_PATH):
        try:
            from app.train_eval import main as train_then_save
            train_then_save()
        except Exception as e:
            # If training fails, we still start the API so /health reports 'degraded'
            pass
        _ensure_model()

    app.run(host="0.0.0.0", port=5000)
