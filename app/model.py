# app/model.py
# ------------------------------------------------------------
# Wind Turbine Anomaly Detector (IsolationForest)
# Features: temperature, humidity, sound_volume
# Purpose: minimal wrapper to train, save/load, and predict
# Output: models/turbine_iforest.pkl (model + scaler + meta)
# ------------------------------------------------------------
from __future__ import annotations

import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class TurbineAnomalyDetector:
    """
    Minimal anomaly detector with:
      - StandardScaler for numeric stability
      - IsolationForest for unsupervised anomaly detection
      - score_samples() used for anomaly_score (inverted so higher = more anomalous)
      - Save/load including metadata (features, params, threshold, version)
    """

    feature_names: List[str] = ["temperature", "humidity", "sound_volume"]

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
        threshold: float = 0.0,
        version: str = "iforest-1.0",
    ):
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.threshold = float(threshold)       # can be updated after Val search in train_eval.py
        self.version = str(version)

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None

    # ---------------------------
    # Training / inference
    # ---------------------------
    def train(self, X: pd.DataFrame) -> None:
        """Fit scaler + model on provided DataFrame."""
        self._check_cols(X)
        Xn = X[self.feature_names].to_numpy(dtype=float, copy=False)

        self.scaler = StandardScaler().fit(Xn)
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(self.scaler.transform(Xn))

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Returns:
          - is_anomaly: int array {0,1}
          - anomaly_score: float array (higher = more anomalous)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained/loaded.")
        self._check_cols(X)
        Z = self.scaler.transform(X[self.feature_names].to_numpy(dtype=float, copy=False))

        # scikit-learn: lower score_samples -> more anomalous
        # invert so that higher means more anomalous (intuitive for plots/thresholding)
        anomaly_score = -self.model.score_samples(Z)
        is_anomaly = (self.model.predict(Z) == -1).astype(int)

        return {"is_anomaly": is_anomaly, "anomaly_score": anomaly_score}

    def predict_row(self, temperature: float, humidity: float, sound_volume: float) -> Dict[str, float]:
        """
        Convenience method for single-row predictions (useful for API).
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained/loaded.")
        x = np.array([[float(temperature), float(humidity), float(sound_volume)]], dtype=float)
        z = self.scaler.transform(x)
        score = float(-self.model.score_samples(z)[0])
        label = int(self.model.predict(z)[0] == -1)
        return {"is_anomaly": label, "anomaly_score": score}

    # ---------------------------
    # Persistence / metadata
    # ---------------------------
    def save(self, path: str = "models/turbine_iforest.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "meta": {
                "feature_names": self.feature_names,
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "threshold": self.threshold,
                "version": self.version,
            },
        }
        joblib.dump(payload, path)

    def load(self, path: str = "models/turbine_iforest.pkl") -> None:
        obj = joblib.load(path)
        self.model = obj["model"]
        self.scaler = obj["scaler"]
        meta = obj.get("meta", {})

        # Restore metadata if present, otherwise keep current defaults
        self.feature_names = meta.get("feature_names", self.feature_names)
        self.contamination = float(meta.get("contamination", self.contamination))
        self.n_estimators = int(meta.get("n_estimators", self.n_estimators))
        self.random_state = int(meta.get("random_state", self.random_state))
        self.threshold = float(meta.get("threshold", self.threshold))
        self.version = str(meta.get("version", self.version))

    def get_meta(self) -> Dict[str, object]:
        """Expose minimal metadata for /model/info or logs."""
        return {
            "version": self.version,
            "threshold": self.threshold,
            "features": list(self.feature_names),
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }

    # ---------------------------
    # Utilities
    # ---------------------------
    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)

    def _check_cols(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
