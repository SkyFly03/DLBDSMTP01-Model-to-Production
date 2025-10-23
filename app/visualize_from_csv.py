# visualize_from_csv.py
# ------------------------------------------------------------
# Create two presentation visuals from the live API log (PNG only).
# Input : app/outputs/predictions_log.csv
# Outputs:
#   app/outputs/anomalies_over_time_from_log.png   (5-min buckets)
#   app/outputs/bubble_scatter_pred.png            (predicted anomalies)
# ------------------------------------------------------------
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR  = "app/outputs"               # where predictions_log.csv lives
SAVE_DIR = "app/outputs"               # where images are written
LOG_PATH = os.path.join(OUT_DIR, "predictions_log.csv")

def _save(fig: plt.Figure, base: str):
    """Save PNG only (no JPGs)."""
    fig.savefig(os.path.join(SAVE_DIR, f"{base}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

def anomalies_over_time(df: pd.DataFrame):
    """Strict 5-minute buckets, no smoothing."""
    g = df.copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"], errors="coerce")
    g = g.dropna(subset=["timestamp"])
    g["is_anomaly"] = g["is_anomaly"].astype(int)

    per_5 = g.set_index("timestamp").resample("5min")["is_anomaly"].sum().reset_index()

    fig = plt.figure(figsize=(9, 3))
    ax = plt.gca()
    ax.plot(per_5["timestamp"], per_5["is_anomaly"], linewidth=2)
    ax.set_title("Anomalies over time (5-minute buckets)")
    ax.set_xlabel("time")
    ax.set_ylabel("count of anomalies")
    ax.grid(True, alpha=0.3)
    _save(fig, "anomalies_over_time_from_log")

def bubble_scatter_pred(df: pd.DataFrame):
    """Temp vs Humidity; bubble size = sound_volume; color by predicted is_anomaly."""
    d = df.copy()
    d["is_anomaly"] = pd.to_numeric(d["is_anomaly"], errors="coerce").fillna(0).astype(int)
    d["temperature"]   = pd.to_numeric(d["temperature"], errors="coerce")
    d["humidity"]      = pd.to_numeric(d["humidity"], errors="coerce")
    d["sound_volume"]  = pd.to_numeric(d["sound_volume"], errors="coerce")

    # Drop rows with missing key fields
    d = d.dropna(subset=["temperature","humidity","sound_volume","is_anomaly"])

    sizes = np.clip(d["sound_volume"].values, 0.01, None) * 220.0

    fig = plt.figure(figsize=(7.5, 5))
    ax = plt.gca()

    m0 = (d["is_anomaly"] == 0)
    ax.scatter(d.loc[m0, "temperature"], d.loc[m0, "humidity"],
               s=sizes[m0], alpha=0.45, label="pred: normal")

    m1 = ~m0
    ax.scatter(d.loc[m1, "temperature"], d.loc[m1, "humidity"],
               s=sizes[m1], alpha=0.9, label="pred: anomaly")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_title("Predicted anomalies (bubble size = sound_volume)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    _save(fig, "bubble_scatter_pred")

def main():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Missing {LOG_PATH}. Start the API + sender to generate it.")
    os.makedirs(SAVE_DIR, exist_ok=True)
    df = pd.read_csv(LOG_PATH)

    # Only generate the two visuals you want
    anomalies_over_time(df)
    bubble_scatter_pred(df)

    # Quick console summary
    anomalies = int((pd.to_numeric(df["is_anomaly"], errors="coerce") == 1).sum())
    print(f"[viz] rows={len(df)} anomalies={anomalies} rate={(anomalies/max(1,len(df))):.2%}")
    print(f"[viz] saved PNGs → {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    main()
