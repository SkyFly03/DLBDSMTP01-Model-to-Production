# app/sender.py
# ------------------------------------------------------------
# Stream Simulator (Fictional Sample Data → REST API)
# Sends JSON to /predict periodically; supports SENDER_COUNT to auto-stop.
# Aligned with API contract & ranges.
# ------------------------------------------------------------
from __future__ import annotations
import os, time, random, math, signal, sys
import requests

API_URL       = os.getenv("API_URL", "http://localhost:5000/predict")
SENDER_COUNT  = int(os.getenv("SENDER_COUNT", "0"))   # 0 = infinite
INTERVAL_SEC  = float(os.getenv("SENDER_INTERVAL_SEC", "0.5"))
TIMEOUT_SEC   = float(os.getenv("SENDER_TIMEOUT_SEC", "3"))
ANOM_RATE     = float(os.getenv("SENDER_ANOM_RATE", "0.05"))  # 5% anomalies

# Valid ranges per API contract
RANGES = {
    "temperature":  (-50.0, 120.0),
    "humidity":     (0.0,   100.0),
    "sound_volume": (0.0,   200.0),
}

def _clip(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, float(v)))

def _rnd(mu: float, sigma: float) -> float:
    return random.gauss(mu, sigma)

def sample() -> dict:
    """
    Generate one reading. Mostly normal ops with occasional anomalies.
    Values are clipped to API-accepted ranges to avoid 4xx responses.
    """
    # Normal operating point
    x = {
        "temperature": 25 + _rnd(0, 3),
        "humidity":    65 + _rnd(0, 10),
        "sound_volume": 0.6 + _rnd(0, 0.12),
    }

    # Inject anomaly with probability ANOM_RATE
    if random.random() < ANOM_RATE:
        x["temperature"]  = random.choice([45 + _rnd(0, 4), 10 + _rnd(0, 4)])
        x["humidity"]     = random.choice([20 + _rnd(0, 8), 95 + _rnd(0, 4)])
        x["sound_volume"] = random.choice([1.4 + _rnd(0, 0.25), 0.05 + _rnd(0, 0.02)])

    # Clip to contract ranges and round for pretty logs
    for k, (lo, hi) in RANGES.items():
        x[k] = round(_clip(x[k], lo, hi), 3)

    return x

def _install_sigint_handler():
    def _handler(signum, frame):
        print("\n[sender] received interrupt, exiting.")
        sys.exit(0)
    signal.signal(signal.SIGINT, _handler)

def main():
    _install_sigint_handler()
    sess = requests.Session()

    # Optional quick health probe
    try:
        r = sess.get(API_URL.replace("/predict", "/health"), timeout=TIMEOUT_SEC)
        if r.ok:
            print(f"[sender] health: {r.json().get('status', 'unknown')}")
        else:
            print(f"[sender] health check failed: HTTP {r.status_code}")
    except Exception as e:
        print(f"[sender] health check error: {e}")

    count = 0
    while True:
        payload = sample()
        try:
            r = sess.post(API_URL, json=payload, timeout=TIMEOUT_SEC)
            if r.ok:
                data = r.json()
                print(f"[{r.status_code}] anom={data.get('is_anomaly')} score={data.get('anomaly_score'):.4f} inputs={payload}")
            else:
                # Print server error body for visibility
                try:
                    print(f"[{r.status_code}] {r.json()}")
                except Exception:
                    print(f"[{r.status_code}] {r.text[:200]}")
        except Exception as e:
            print(f"[sender] send error: {e}")

        count += 1
        if SENDER_COUNT and count >= SENDER_COUNT:
            break
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
