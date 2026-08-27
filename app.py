"""
app.py
The Python ML microservice for Component E.

Exposes:  POST /predict
Also exposes: GET /health  (so n8n / Docker can check the service is alive)

Run locally:      python3 app.py
Run in Docker:     see Dockerfile in the same folder

INPUT (JSON body):
{
  "device_id": "PM-001",
  "readings": [
    {"timestamp": "2026-08-27T09:00:00Z", "current": 1.40, "vibration": 0.30, "temperature": 38.1},
    ... (send the most recent 24 hourly readings, oldest first; fewer than 24 is
         accepted and will be auto-padded, but 24 gives the most accurate forecast)
  ]
}

OUTPUT (JSON body):
{
  "device_id": "PM-001",
  "anomaly_score": 0.83,
  "risk_level": "HIGH",
  "forecast": {
    "risk_in_24hr": 0.91,
    "risk_in_72hr": 0.97,
    "trend": "increasing"
  },
  "shap_explanation": {
    "current": 0.52,
    "vibration": 0.33,
    "temperature": 0.15
  },
  "top_contributing_sensor": "current",
  "padded_input": false,
  "timestamp": "2026-08-27T09:00:00Z"
}
"""

import numpy as np
import pandas as pd
import joblib
import shap
from flask import Flask, request, jsonify
from tensorflow import keras

app = Flask(__name__)

FEATURES = ["current", "vibration", "temperature"]
WINDOW_SIZE = 24

# ---- Load all three models once, at startup (not per-request) ----
iso_model = joblib.load("models/isoforest.pkl")
iso_scaler = joblib.load("models/scaler.pkl")
lstm_model = keras.models.load_model("models/lstm_model.keras")
feature_scaler = joblib.load("models/feature_scaler.pkl")
shap_explainer = shap.TreeExplainer(iso_model)


def risk_from_decision_score(raw_score: float) -> float:
    """Same formula used during training: maps decision_function output to a 0-1 risk score."""
    return float(np.clip((0.25 - raw_score) / 0.5, 0.0, 1.0))


def risk_level(score: float) -> str:
    if score < 0.35:
        return "LOW"
    if score < 0.65:
        return "MEDIUM"
    return "HIGH"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body or "readings" not in body or len(body["readings"]) == 0:
        return jsonify({"error": "Request must include a non-empty 'readings' list."}), 400

    device_id = body.get("device_id", "unknown")
    readings = body["readings"]

    # validate each reading has the three required sensor fields
    for r in readings:
        missing = [f for f in FEATURES if f not in r]
        if missing:
            return jsonify({"error": f"Each reading needs {FEATURES}. Missing: {missing}"}), 400

    padded = False
    if len(readings) < WINDOW_SIZE:
        # pad by repeating the earliest reading backwards so the LSTM gets a full window
        pad_count = WINDOW_SIZE - len(readings)
        readings = [readings[0]] * pad_count + readings
        padded = True
    else:
        readings = readings[-WINDOW_SIZE:]  # keep only the most recent WINDOW_SIZE

    latest = readings[-1]
    latest_df = pd.DataFrame([[latest[f] for f in FEATURES]], columns=FEATURES)

    # ---- 1. Isolation Forest: anomaly score for the LATEST reading ----
    X_iso = iso_scaler.transform(latest_df)
    raw_score = float(iso_model.decision_function(X_iso)[0])
    anomaly_score = risk_from_decision_score(raw_score)

    # ---- 2. SHAP: which sensor is driving that anomaly score ----
    shap_values = shap_explainer.shap_values(X_iso)[0]  # one row -> shape (3,)
    # flip sign: a NEGATIVE shap value pushes decision_function down, i.e. MORE anomalous.
    # we report positive "contribution to anomaly" so it's intuitive to read.
    contributions = np.clip(-shap_values, 0, None)
    total = contributions.sum()
    if total > 0:
        contributions = contributions / total
    shap_explanation = {f: round(float(c), 3) for f, c in zip(FEATURES, contributions)}
    top_sensor = FEATURES[int(np.argmax(contributions))]

    # ---- 3. LSTM: 24hr / 72hr forward risk trend, using the full window ----
    window_df = pd.DataFrame([[r[f] for f in FEATURES] for r in readings], columns=FEATURES)
    window_scaled = feature_scaler.transform(window_df)
    window_scaled = window_scaled.reshape(1, WINDOW_SIZE, len(FEATURES))
    forecast = lstm_model.predict(window_scaled, verbose=0)[0]
    risk_24, risk_72 = float(forecast[0]), float(forecast[1])
    trend = "increasing" if risk_72 > anomaly_score + 0.05 else (
        "decreasing" if risk_72 < anomaly_score - 0.05 else "stable"
    )

    response = {
        "device_id": device_id,
        "anomaly_score": round(anomaly_score, 3),
        "risk_level": risk_level(anomaly_score),
        "forecast": {
            "risk_in_24hr": round(risk_24, 3),
            "risk_in_72hr": round(risk_72, 3),
            "trend": trend
        },
        "shap_explanation": shap_explanation,
        "top_contributing_sensor": top_sensor,
        "padded_input": padded,
        "timestamp": latest.get("timestamp")
    }
    return jsonify(response), 200


if __name__ == "__main__":
    # 0.0.0.0 so it's reachable from other Docker containers (like n8n) or other devices on WiFi
    app.run(host="0.0.0.0", port=5000)
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    