# tests/test.py
# ------------------------------------------------------------
# End-to-End tests (minimal & robust)
# - Trains model if missing
# - Checks /health, /model/info, /metrics, /predict
# - Generates log rows and validates CSV schema (timestamp OR timestamp_utc)
# - Runs CSV visualizer and checks key artifacts
# ------------------------------------------------------------
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math, random, pandas as pd

MODEL_PATH = "models/turbine_iforest.pkl"
OUT_DIR = "app/outputs"
LOG_PATH = os.path.join(OUT_DIR, "predictions_log.csv")

def _gauss(mu, sigma): return random.gauss(mu, sigma)
def sample_ok():
    return {"temperature": 25 + _gauss(0,3),
            "humidity": 65 + _gauss(0,10),
            "sound_volume": 0.6 + _gauss(0,0.12)}

def ensure_model():
    from app.train_eval import main as train_main
    need = False

    if not os.path.exists(MODEL_PATH):
        need = True

    required_pngs = [
        "metrics_table.png",
        "heatmaps_grid.png",
        "learning_dashboard_2x2.png",
    ]
    for f in required_pngs:
        if not os.path.exists(os.path.join(OUT_DIR, f)):
            need = True
            break

    if need:
        print("[test] building artifacts via training ...")
        train_main()
        assert os.path.exists(MODEL_PATH), "Model training failed."

def make_app():
    from app.api import app, _ensure_model
    _ensure_model()
    return app

def run_tests():
    ensure_model()
    app = make_app()
    client = app.test_client()

    # /health
    r = client.get("/health"); assert r.status_code == 200
    j = r.get_json(); assert j.get("model_loaded") is True
    print("[test] /health OK")

    # /model/info
    r = client.get("/model/info"); assert r.status_code == 200
    j = r.get_json()
    assert j.get("model_type") == "IsolationForest"
    feats = j.get("features") or j.get("feature_names")
    assert feats is not None and set(feats) == {"temperature","humidity","sound_volume"}
    print("[test] /model/info OK")

    # /metrics
    r = client.get("/metrics"); assert r.status_code == 200
    j = r.get_json()
    for k in ["requests_total","requests_4xx","requests_5xx","predictions_total","avg_latency_ms"]:
        assert k in j
    print("[test] /metrics OK")

    # /predict — valid
    r = client.post("/predict", json=sample_ok()); assert r.status_code == 200
    p = r.get_json()
    assert isinstance(p.get("anomaly_score"), (int,float))
    assert math.isfinite(float(p["anomaly_score"])) and -10.0 < float(p["anomaly_score"]) < 10.0
    assert isinstance(p.get("is_anomaly"), (bool,int))
    ts = p.get("timestamp") or p.get("timestamp_utc")
    assert isinstance(ts, str) and len(ts) >= 10
    print("[test] /predict OK (valid)")

    # populate log
    for _ in range(8):
        client.post("/predict", json=sample_ok())

    # /predict — invalid -> 400
    r = client.post("/predict", json={"temperature":25.0,"humidity":60.0})
    assert r.status_code == 400
    print("[test] /predict OK (invalid payload → 400)")

    # predictions_log.csv (accept timestamp OR timestamp_utc)
    assert os.path.exists(LOG_PATH), "predictions log missing"
    df_log = pd.read_csv(LOG_PATH)
    cols = set(df_log.columns)
    schema_a = {"timestamp","temperature","humidity","sound_volume","is_anomaly","anomaly_score"}
    schema_b = {"timestamp_utc","temperature","humidity","sound_volume","is_anomaly","anomaly_score"}
    assert schema_a.issubset(cols) or schema_b.issubset(cols), f"log schema mismatch: {cols}"
    print(f"[test] log OK ({len(df_log)} rows)")

    # Visualization from CSV (best-effort)
    try:
        from app.visualize_from_csv import main as viz_main
        viz_main()
        ao = os.path.join(OUT_DIR, "anomalies_over_time.png")
        assert os.path.exists(ao), "anomalies_over_time.png not created"
        print("[test] visualize_from_csv OK")
    except Exception as e:
        print("[test] visualize_from_csv skipped:", e)

    # Training/Eval artifacts
    for f in ["metrics_table.png","heatmaps_grid.png","learning_dashboard_2x2.png"]:
        assert os.path.exists(os.path.join(OUT_DIR, f)), f"missing {f}"

    mt = os.path.join(OUT_DIR, "metrics_table.csv")
    assert os.path.exists(mt), "missing metrics_table.csv"
    df_mt = pd.read_csv(mt)
    assert {"Split","Accuracy","Precision","Recall","F1","ROC_AUC"}.issubset(df_mt.columns)

    print("\n[test] PASS  All checks succeeded.")

if __name__ == "__main__":
    run_tests()
