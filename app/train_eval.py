# app/train_eval.py
# ------------------------------------------------------------
# Training & Evaluation
# - Generate fictional sensor data (temp, humidity, sound_volume)
# - 70/10/20 split (Train/Val/Test), train IsolationForest via model.py
# - Export: metrics_table.png, heatmaps_grid.png
# - 2×2 learning dashboard: ROC(Test) + F1 curves + size/sweep plots
# - Per-feature anomalies over time (5-min buckets) from predictions_log.csv
# ------------------------------------------------------------
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.utils import shuffle
try:
    # When running as a module: python -m app.train_eval
    from app.model import TurbineAnomalyDetector
except ImportError:
    # When running directly: python app/train_eval.py
    from model import TurbineAnomalyDetector

OUT_DIR = os.path.join("app", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# Data generation & split
# -------------------------
def make_fictional_sensor_data(n: int = 3000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    k_anom = max(1, int(0.05 * n))
    k_norm = n - k_anom

    temp_n = rng.normal(25, 3, k_norm)
    hum_n = rng.normal(60, 8, k_norm)
    snd_n = rng.normal(0.5, 0.1, k_norm)

    temp_a = rng.choice([rng.normal(45, 3, k_anom), rng.normal(10, 3, k_anom)]).copy()
    hum_a = rng.choice([rng.normal(90, 5, k_anom), rng.normal(20, 5, k_anom)]).copy()
    snd_a = rng.choice([rng.normal(1.2, 0.1, k_anom), rng.normal(0.1, 0.05, k_anom)]).copy()

    X = pd.DataFrame({
        "temperature": np.concatenate([temp_n, temp_a]),
        "humidity": np.concatenate([hum_n, hum_a]),
        "sound_volume": np.concatenate([snd_n, snd_a]),
    })
    y = np.array([0] * k_norm + [1] * k_anom, dtype=int)
    X, y = shuffle(X, y, random_state=seed)
    return X.reset_index(drop=True), y


def split_70_10_20(X: pd.DataFrame, y: np.ndarray, seed: int = 42):
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    i_tr = idx[:n_train]
    i_va = idx[n_train:n_train + n_val]
    i_te = idx[n_train + n_val:]
    return (X.iloc[i_tr].reset_index(drop=True), X.iloc[i_va].reset_index(drop=True),
            X.iloc[i_te].reset_index(drop=True), y[i_tr], y[i_va], y[i_te])


# -------------------------
# Metrics & small utilities
# -------------------------
def compute_metrics(y_true, scores, yhat, split_name):
    return {
        "Split": split_name,
        "Accuracy": accuracy_score(y_true, yhat),
        "Precision": precision_score(y_true, yhat, zero_division=0),
        "Recall": recall_score(y_true, yhat, zero_division=0),
        "F1": f1_score(y_true, yhat, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, scores),
    }


def _save_table_csv(df: pd.DataFrame, name: str):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    return path


def plot_score_distributions(det: TurbineAnomalyDetector, splits: dict):
    """
    Histogram (density) of anomaly_score for Train/Val/Test.
    Robust to DataFrame/dict/tuple/ndarray returns from det.predict.
    """
    def _extract_scores(pred_out):
        if isinstance(pred_out, pd.DataFrame) and "anomaly_score" in pred_out.columns:
            return pred_out["anomaly_score"].to_numpy()
        if isinstance(pred_out, dict) and "anomaly_score" in pred_out:
            return np.asarray(pred_out["anomaly_score"])
        if isinstance(pred_out, (list, tuple)) and len(pred_out) > 0:
            return np.asarray(pred_out[0])
        arr = np.asarray(pred_out)
        return arr[:, 0] if arr.ndim == 2 and arr.shape[1] >= 1 else arr

    colors = {"Train": "#56B4E9", "Val": "#E69F00", "Test": "#D55E00"}  # color-blind–safe
    # Get all scores once to set a common bin range
    all_scores = []
    per_split = {}
    for name, (Xsplit, _) in splits.items():
        s = _extract_scores(det.predict(Xsplit))
        per_split[name] = s
        if s.size:
            all_scores.append(s)
    if all_scores:
        all_scores = np.concatenate(all_scores)
        smin, smax = np.min(all_scores), np.max(all_scores)
    else:
        smin, smax = -1.0, 1.0  # fallback

    plt.figure(figsize=(10, 5))
    for name, s in per_split.items():
        if s.size:
            plt.hist(s, bins=40, range=(smin, smax), density=True,
                     alpha=0.35, label=name, color=colors.get(name))

    plt.title("Anomaly Score Distributions (density)")
    plt.xlabel("anomaly_score (higher = more anomalous)")
    plt.ylabel("density")
    plt.legend()

    # Format x-axis as European date + time: DD-MM-YY HH:MM
    import matplotlib.dates as mdates
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# -------------------------
# Visuals
# -------------------------
def plot_heatmaps_grid(X, yval, yhat_val, yte, yhat_te, split_label):
    import itertools, numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # ---- Correlation heatmap (robust to bad dtypes / NaNs) ----
    ax = axes[0]
    cols = [c for c in ["temperature", "humidity", "sound_volume"] if c in X.columns]

    # Coerce to numeric, handle NaNs
    df_num = X[cols].apply(pd.to_numeric, errors="coerce")
    # Drop rows that are all-NaN across the three features
    df_num = df_num.dropna(how="all")
    # Fill remaining NaNs with column means (if any)
    if not df_num.empty:
        df_num = df_num.fillna(df_num.mean(numeric_only=True))

    # Compute correlation or fall back to identity
    if df_num.shape[0] >= 2 and len(cols) >= 2:
        corr = df_num.corr().to_numpy()
    else:
        # Not enough data — use identity so the plot is still informative
        n = max(1, len(cols))
        corr = np.eye(n)

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title("Correlation heatmap")

    # Annotate each cell with the numeric value
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="#111")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Confusion matrices (Val/Test) ----
    def _conf(ax, y, yhat, ttl):
        y = np.asarray(y).astype(int)
        yhat = np.asarray(yhat).astype(int)
        cm = confusion_matrix(y, yhat, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for i, j in itertools.product(range(2), range(2)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_yticklabels(["Normal", "Anomaly"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(ttl)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _conf(axes[1], yval, yhat_val, "Confusion (Val)")
    _conf(axes[2], yte,  yhat_te,  "Confusion (Test)")

    plt.suptitle(split_label, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "heatmaps_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()

def plot_feature_histograms_by_label(X: pd.DataFrame, y: np.ndarray, bins: int = 40):
    """
    Three side-by-side histograms showing normal vs anomaly for:
      - temperature (°C)
      - humidity (%)
      - sound_volume (arb)
    Colors: normal=blue (#56B4E9), anomaly=orange (#E69F00)
    Saves: app/outputs/histograms.png
    """
    # Make a clean numeric copy (in case anything came from CSV)
    cols = ["temperature", "humidity", "sound_volume"]
    Xp = X.copy()
    for c in cols:
        Xp[c] = pd.to_numeric(Xp[c], errors="coerce")
    mask = ~Xp[cols].isna().any(axis=1)
    Xp = Xp.loc[mask].reset_index(drop=True)
    yp = np.asarray(y)[mask.values]

    normal = Xp[yp == 0]
    anom   = Xp[yp == 1]

    # Okabe–Ito color-blind friendly
    C_NORMAL = "#56B4E9"  # blue
    C_ANOM   = "#E69F00"  # orange

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    specs = [
        ("temperature",  "Temperature (°C)"),
        ("humidity",     "Humidity (%)"),
        ("sound_volume", "Sound volume (arb)"),
    ]

    for ax, (col, xlab) in zip(axes, specs):
        # normal
        ax.hist(
            normal[col].to_numpy(),
            bins=bins, alpha=0.85, color=C_NORMAL,
            edgecolor="white", linewidth=0.4, label="normal"
        )
        # anomaly
        ax.hist(
            anom[col].to_numpy(),
            bins=bins, alpha=0.85, color=C_ANOM,
            edgecolor="white", linewidth=0.4, label="anomaly"
        )

        ax.set_title(col)
        ax.set_xlabel(xlab)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    axes[0].set_ylabel("count")
    fig.suptitle("", y=1.02)  # no big title to match your reference style
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "histograms.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def learning_dashboard_2x2_v2(Xtr, ytr, Xval, yval, Xte, yte, contamination=0.05):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    base = TurbineAnomalyDetector(contamination=contamination, n_estimators=100, random_state=42)
    base.train(Xtr)

    def _scores(pred_out):
        if isinstance(pred_out, pd.DataFrame) and "anomaly_score" in pred_out.columns:
            return pred_out["anomaly_score"].to_numpy()
        if isinstance(pred_out, dict) and "anomaly_score" in pred_out:
            return np.asarray(pred_out["anomaly_score"])
        if isinstance(pred_out, (list, tuple)) and len(pred_out) > 0:
            return np.asarray(pred_out[0])
        arr = np.asarray(pred_out)
        return arr[:, 0] if arr.ndim == 2 and arr.shape[1] >= 1 else arr

    s_tr = _scores(base.predict(Xtr))
    s_va = _scores(base.predict(Xval))
    s_te = _scores(base.predict(Xte))

    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(yte, s_te)
    auc = roc_auc_score(yte, s_te)
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], "--")
    ax.set_title(f"ROC (Test)  AUC = {auc:.3f}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.grid(True, alpha=0.3)

    def _smooth(y, k=7):
        return np.convolve(y, np.ones(k) / k, mode="same") if len(y) >= k else y

    ax = axes[0, 1]
    tmin, tmax = min(s_tr.min(), s_va.min(), s_te.min()), max(s_tr.max(), s_va.max(), s_te.max())
    ts = np.linspace(tmin, tmax, 300)

    def _f1s(scores, y):
        return np.array([f1_score(y, (scores >= t).astype(int), zero_division=0) for t in ts])

    ax.plot(ts, _smooth(_f1s(s_tr, ytr)), label="Train", linewidth=2)
    ax.plot(ts, _smooth(_f1s(s_va, yval)), label="Val", linewidth=2)
    ax.plot(ts, _smooth(_f1s(s_te, yte)), label="Test", linewidth=2)
    ax.set_title("F1 vs threshold — Train / Val / Test")
    ax.set_xlabel("anomaly_score threshold")
    ax.set_ylabel("F1")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    fracs = np.linspace(0.1, 1.0, 12)
    f1_curve = []
    for f in fracs:
        n = max(50, int(len(Xtr) * f))
        d = TurbineAnomalyDetector(contamination=contamination, n_estimators=100, random_state=42)
        d.train(Xtr.iloc[:n])
        scores_val = _scores(d.predict(Xval))
        f1_curve.append(f1_score(yval, (scores_val >= 0).astype(int), zero_division=0))

    ax.plot(fracs, _smooth(np.array(f1_curve), 3), marker="o")
    ax.set_title("Val F1 vs fraction of train")
    ax.set_xlabel("fraction of train")
    ax.set_ylabel("F1")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    est_list = [25, 50, 100, 150, 200, 300]
    f1_est = []
    for n in est_list:
        d = TurbineAnomalyDetector(contamination=contamination, n_estimators=n, random_state=42)
        d.train(Xtr)
        scores_val = _scores(d.predict(Xval))
        f1_est.append(f1_score(yval, (scores_val >= 0).astype(int), zero_division=0))

    ax.plot(est_list, _smooth(np.array(f1_est), 3), marker="o")
    ax.set_title("Val F1 vs n_estimators")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("F1")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "learning_dashboard_2x2.png"), dpi=150)
    plt.close()


def plot_feature_anomalies_over_time(
    log_path: str,
    ref_means: dict | None = None,
    ref_stds: dict | None = None,
    window_minutes: int = 5,
    smooth_points: int = 3,
    hours_back: int | None = None,     # kept for backward compat
    start_at: str | None = None,       # e.g., "2025-10-13"
    end_at: str | None = None,         # e.g., "2025-10-13 23:59:59"
):
    """
    Build per-feature anomaly counts over time from predictions_log.csv.

    Robustness:
    - Parses timestamps and coerces feature columns to numeric.
    - If hours_back is set, windows relative to the log's latest timestamp (not now),
      so older logs still render.
    - If ref_means/ref_stds are None, computes baselines from the log itself
      (prefer is_anomaly==0 rows if present, else all rows).
    - Buckets by N minutes and draws 3 colored lines (temperature, humidity, sound_volume).
    """
    if not os.path.exists(log_path):
        return

    df = pd.read_csv(log_path)

    # --- Parse timestamp robustly ---
    if "timestamp" not in df.columns:
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    # make tz-naive for plotting
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    # --- Coerce features to numeric ---
    for col in ["temperature", "humidity", "sound_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # if any feature missing, bail gracefully
            return
    df = df.dropna(subset=["temperature", "humidity", "sound_volume"])

    # --- Window selection ---
    if start_at or end_at:
        if start_at:
            start_ts = pd.to_datetime(start_at, utc=True).tz_convert(None)
            df = df[df["timestamp"] >= start_ts]
        if end_at:
            end_ts = pd.to_datetime(end_at, utc=True).tz_convert(None)
            df = df[df["timestamp"] <= end_ts]
    elif hours_back is not None and len(df) > 0:
        latest = df["timestamp"].max()
        cutoff = latest - pd.Timedelta(hours=hours_back)
        df = df[df["timestamp"] >= cutoff]
    if df.empty:
        # Write an empty-but-informative plot instead of failing silently
        plt.figure(figsize=(12, 3))
        plt.title("Anomalies over time — no data in selected window")
        plt.xlabel("time"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "anomalies_over_time.png"), dpi=150, bbox_inches="tight")
        plt.close()
        return

    # --- Choose baseline (means/stds) ---
    if ref_means is None or ref_stds is None:
        # Prefer normal rows if a label exists
        base = df
        if "is_anomaly" in df.columns:
            try:
                base = df[df["is_anomaly"].astype(int) == 0]
                if base.empty:
                    base = df
            except Exception:
                base = df
        ref_means = {c: float(base[c].mean()) for c in ["temperature","humidity","sound_volume"]}
        ref_stds  = {c: float(base[c].std(ddof=0) or 1e-9) for c in ["temperature","humidity","sound_volume"]}

    # --- Per-feature anomaly flags by z-score ---
    g = df[["timestamp","temperature","humidity","sound_volume"]].copy().set_index("timestamp").sort_index()
    out = pd.DataFrame(index=g.index)
    for c in ["temperature","humidity","sound_volume"]:
        mu = float(ref_means[c]); sd = max(float(ref_stds[c]), 1e-9)
        out[c] = ((g[c] - mu).abs() / sd > 3).astype(int)

    # --- Resample to N-minute buckets ---
    per = out.resample(f"{window_minutes}min").sum().reset_index()
    if per.empty:
        # Ensure we still save a file
        plt.figure(figsize=(12, 3))
        plt.title(f"Anomalies over time ({window_minutes}-minute buckets) — no buckets")
        plt.xlabel("time"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "anomalies_over_time.png"), dpi=150, bbox_inches="tight")
        plt.close()
        return

    # --- Plot 3 lines with light smoothing ---
    plt.figure(figsize=(12, 4))
    palette = {
        "temperature": "#E69F00",  # orange
        "humidity":    "#0072B2",  # blue
        "sound_volume":"#009E73",  # green
    }
    for c in ["temperature","humidity","sound_volume"]:
        y = per[c].to_numpy()
        if smooth_points and len(y) >= smooth_points:
            y = np.convolve(y, np.ones(smooth_points)/smooth_points, mode="same")
        plt.plot(per["timestamp"], y, label=c, linewidth=2, color=palette[c])
    
    # Format x-axis as DD-MM-YY HH:MM
    import matplotlib.dates as mdates
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%y %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.title(f"Anomalies over time ({window_minutes}-minute buckets) — per feature")
    plt.xlabel("time"); plt.ylabel("count")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "anomalies_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close()

def render_metrics_table_image(df, title, out_path):
    fmt = df.copy()
    for c in fmt.columns:
        if pd.api.types.is_numeric_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda x: f"{float(x):.3f}")
        else:
            fmt[c] = fmt[c].astype(str)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=12)
    tbl = ax.table(cellText=fmt.values, colLabels=fmt.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    X, y = make_fictional_sensor_data(n=9800, seed=42)
    Xtr, Xval, Xte, ytr, yval, yte = split_70_10_20(X, y)
    SPLIT_LABEL = "70% Train / 10% Val / 20% Test"
    plot_feature_histograms_by_label(X, y)

    det = TurbineAnomalyDetector(contamination=0.05, n_estimators=100, random_state=42)
    det.train(Xtr)

    res_tr, res_va, res_te = det.predict(Xtr), det.predict(Xval), det.predict(Xte)
    dfm = pd.DataFrame([
        compute_metrics(ytr, res_tr["anomaly_score"], res_tr["is_anomaly"], "Train"),
        compute_metrics(yval, res_va["anomaly_score"], res_va["is_anomaly"], "Val"),
        compute_metrics(yte, res_te["anomaly_score"], res_te["is_anomaly"], "Test"),
    ])
    _save_table_csv(dfm, "metrics_table.csv")
    render_metrics_table_image(
        dfm[["Split", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]],
        f"Model metrics — {SPLIT_LABEL}",
        os.path.join(OUT_DIR, "metrics_table.png")
    )

    plot_heatmaps_grid(X, yval, res_va["is_anomaly"], yte, res_te["is_anomaly"], SPLIT_LABEL)

    learning_dashboard_2x2_v2(Xtr, ytr, Xval, yval, Xte, yte)

    ref_means = Xtr.mean().to_dict()
    ref_stds = Xtr.std(ddof=0).to_dict()
    plot_feature_anomalies_over_time(
        os.path.join(OUT_DIR, "predictions_log.csv"), ref_means, ref_stds
    )

    # --- Per-feature anomalies over time: ONLY 13.10.2025 ---
    plot_feature_anomalies_over_time(
        log_path=os.path.join(OUT_DIR, "predictions_log.csv"),
        ref_means=None,
        ref_stds=None,
        window_minutes=15,         # 15-min buckets look clean for a full day
        smooth_points=5,
        hours_back=None,           # ignored because start_at/end_at are set
        start_at="2025-10-13 00:00:00",
        end_at="2025-10-13 23:59:59",
    )

if __name__ == "__main__":
    main()
