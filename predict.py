"""
predict.py
==========
Loads all trained artifacts ONCE (at import time / API startup) and exposes
`run_prediction(payload)`, which combines:

    Isolation Forest -> anomaly_score, is_anomaly
    LSTM             -> trend_risk
    SHAP             -> feature contributions (on the Isolation Forest)
    -> combined into a final risk_level + forecast_hours

app.py imports this module and calls `run_prediction()` inside the
/predict handler. Models are NOT retrained or reloaded per-request.

RISK THRESHOLDS (documented per task requirement #6)
------------------------------------------------------
`combined_risk = 0.5 * anomaly_score + 0.5 * trend_risk`

    combined_risk <  0.35   -> "LOW"
    0.35 <= combined_risk < 0.70 -> "MEDIUM"
    combined_risk >= 0.70   -> "HIGH"

These thresholds and the 50/50 weighting are PROTOTYPE / DEMO values
chosen for a readable three-tier output, NOT derived from real railway
safety data or any regulatory standard. Anyone deploying this against real
assets must recalibrate these thresholds using real labelled failure data
(e.g. via ROC-curve analysis against confirmed maintenance events).

`forecast_hours` (documented per task requirement #9)
-------------------------------------------------------
A simple heuristic, NOT a physics-based or data-fitted remaining-useful-
life model: assuming the current `trend_risk` continues to grow at the
average rate observed in the synthetic training cycles (~0.017 trend_risk
units per 10-minute reading, i.e. roughly linear degradation), we estimate
hours until `trend_risk` would reach 1.0 (full synthetic degradation):

    remaining_fraction = max(0, 1 - trend_risk)
    forecast_hours = remaining_fraction / synthetic_hourly_rate

This is intentionally simple and clearly a DEMO heuristic — it must be
replaced with a model fitted on real degradation-to-failure data before
any real maintenance-scheduling decision is based on it.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np

import preprocessing as pp
import explainability as ex

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Demo risk thresholds — see module docstring.
RISK_LOW_MAX = 0.35
RISK_MEDIUM_MAX = 0.70

# Synthetic degradation rate used only for the demo forecast_hours heuristic:
# in generate_synthetic_dataset(), trend_risk goes 0 -> 1 over
# `cycle_length` readings taken 10 minutes apart.
_SYNTHETIC_READING_INTERVAL_HOURS = 10 / 60
_SYNTHETIC_CYCLE_LENGTH = 60
_SYNTHETIC_HOURLY_RATE = 1.0 / (_SYNTHETIC_CYCLE_LENGTH * _SYNTHETIC_READING_INTERVAL_HOURS)


class ModelBundle:
    """Loads every artifact once and keeps them in memory for reuse."""

    def __init__(self, models_dir: str = MODELS_DIR):
        config_path = os.path.join(models_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No trained models found in {models_dir}. Run `python train.py` first."
            )

        with open(config_path) as f:
            self.config = json.load(f)

        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        self.isolation_forest = joblib.load(os.path.join(models_dir, "isolation_forest.pkl"))

        import tensorflow as tf  # local import so the module can be inspected without TF installed

        self.lstm_model = tf.keras.models.load_model(os.path.join(models_dir, "lstm_model.keras"))

        background = np.load(os.path.join(models_dir, "shap_background.npy"))
        self.explainer = ex.AnomalyExplainer(self.isolation_forest, background)

        self.score_range = self.config["anomaly_score_range"]
        self.sequence_length = self.config["sequence_length"]
        self.features = self.config["features"]


_bundle: ModelBundle | None = None


def get_bundle() -> ModelBundle:
    """Lazily load the model bundle once, then reuse it for every call."""
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle


def _anomaly_score_from_scaled(bundle: ModelBundle, x_scaled_row: np.ndarray) -> float:
    raw = float(-bundle.isolation_forest.decision_function(x_scaled_row.reshape(1, -1))[0])
    rmin, rmax = bundle.score_range["raw_min"], bundle.score_range["raw_max"]
    score = (raw - rmin) / (rmax - rmin)
    return float(np.clip(score, 0.0, 1.0))


def _risk_level(combined_risk: float) -> str:
    if combined_risk < RISK_LOW_MAX:
        return "LOW"
    if combined_risk < RISK_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def run_prediction(asset_id: str, timestamp: str, sequence: list[dict]) -> dict:
    """
    sequence: list of dicts, each with keys motor_current_A, vibration_g,
    temperature_C, ordered OLDEST -> NEWEST, already validated by app.py to
    have at least `sequence_length` entries. Only the LAST
    `sequence_length` entries are used.

    Returns the exact response schema documented in README.md / app.py.
    """
    bundle = get_bundle()
    seq_len = bundle.sequence_length

    window = sequence[-seq_len:]
    X_scaled_seq = pp.sequence_payload_to_array(window, bundle.scaler)  # (seq_len, n_features)
    latest_scaled = X_scaled_seq[-1]  # current reading, scaled

    # --- Isolation Forest -------------------------------------------------
    anomaly_score = _anomaly_score_from_scaled(bundle, latest_scaled)
    is_anomaly = bool(anomaly_score > 0.6)  # same demo threshold as train.py evaluation

    # --- LSTM ---------------------------------------------------------------
    lstm_input = X_scaled_seq.reshape(1, seq_len, len(bundle.features))
    trend_risk = float(bundle.lstm_model.predict(lstm_input, verbose=0).reshape(-1)[0])
    trend_risk = float(np.clip(trend_risk, 0.0, 1.0))

    # --- SHAP (explains the Isolation Forest's score on the latest reading) -
    shap_values = bundle.explainer.explain(latest_scaled)

    # --- Combine into final risk assessment --------------------------------
    combined_risk = 0.5 * anomaly_score + 0.5 * trend_risk
    risk_level = _risk_level(combined_risk)

    remaining_fraction = max(0.0, 1.0 - trend_risk)
    forecast_hours = round(remaining_fraction / _SYNTHETIC_HOURLY_RATE, 1)

    return {
        "asset_id": asset_id,
        "timestamp": timestamp,
        "anomaly_score": round(anomaly_score, 4),
        "is_anomaly": is_anomaly,
        "trend_risk": round(trend_risk, 4),
        "risk_level": risk_level,
        "forecast_hours": forecast_hours,
        "shap": {k: round(v, 4) for k, v in shap_values.items()},
    }
