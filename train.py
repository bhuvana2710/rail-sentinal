"""
train.py
========
Full training pipeline for Person 3's ML system.

Run with:
    python train.py

This script:
  1. Generates the synthetic dataset (see preprocessing.py for why).
  2. Cleans it and splits it by asset_id into train/val/test (no leakage).
  3. Fits a StandardScaler on the training split only.
  4. Trains an Isolation Forest on (mostly normal) training data.
  5. Computes and stores the anomaly-score normalization range.
  6. Builds LSTM sequences and trains an LSTM regressor to predict
     `trend_risk_label` (synthetic degradation proxy, 0-1) from the last
     SEQUENCE_LENGTH readings.
  7. Evaluates both models on the held-out test split.
  8. Builds and stores a SHAP background sample.
  9. Saves every artifact needed by predict.py / app.py to models/.

It intentionally does NOT start the API — see app.py. Models are trained
once, here, and loaded (not retrained) on every /predict call.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)

import preprocessing as pp

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------
def train_isolation_forest(train_df):
    """
    Train an Isolation Forest on the training split.

    How Isolation Forest works (documented per task requirement #3):
    Isolation Forest builds an ensemble of random binary trees. Each tree
    isolates points by repeatedly picking a random feature and a random
    split value between that feature's min/max. Anomalies are, by
    definition, "few and different" — they tend to require FEWER random
    splits to isolate them into their own leaf than normal points do
    (normal points are surrounded by many similar points and need many
    splits to separate). The model's anomaly indicator is therefore the
    AVERAGE PATH LENGTH to isolate a point across all trees in the forest:
    short average path length -> likely anomaly, long average path length
    -> likely normal.

    We train it mostly on `is_anomaly_label == 0` rows (i.e. rows we know,
    from the synthetic generator, are normal operating readings) so the
    forest learns the shape of "normal" behaviour. We keep a small fraction
    of the injected anomalies out of training (they go to val/test only) so
    we can measure detection performance.
    """
    normal_train = train_df[train_df["is_anomaly_label"] == 0]
    scaler = pp.fit_scaler(normal_train)  # fit scaler on normal training rows
    X_train_scaled = pp.scale_features(normal_train, scaler)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=pp.RANDOM_SEED,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_scaled)

    return iso_forest, scaler, X_train_scaled


def compute_anomaly_score_range(iso_forest, X_scaled: np.ndarray) -> dict:
    """
    IsolationForest.decision_function(X) returns higher values for MORE
    NORMAL points and lower/negative values for MORE ANOMALOUS points
    (scikit-learn convention). To turn this into an intuitive
    "anomaly_score" where 0 = normal and 1 = highly anomalous, we:

        raw = -decision_function(x)              # flip sign: higher = worse
        anomaly_score = clip((raw - raw_min) / (raw_max - raw_min), 0, 1)

    where raw_min/raw_max are taken from the TRAINING distribution of raw
    scores (5th/95th percentile, to avoid a single extreme training point
    collapsing the whole scale). This range is saved to disk and reused,
    UNCHANGED, at inference time — it is never recomputed per request,
    so scores are comparable across requests.
    """
    raw = -iso_forest.decision_function(X_scaled)
    raw_min = float(np.percentile(raw, 1))
    raw_max = float(np.percentile(raw, 99.9))
    #raw_min = float(np.percentile(raw, 5))
    #raw_max = float(np.percentile(raw, 95))
    if raw_max <= raw_min:
        raw_max = raw_min + 1e-6
    return {"raw_min": raw_min, "raw_max": raw_max}


def evaluate_isolation_forest(iso_forest, scaler, score_range, test_df):
    """
    Evaluate against the SYNTHETIC ground-truth anomaly labels
    (is_anomaly_label) that only exist because we generated the data
    ourselves. On real data these labels will not exist ahead of time;
    this evaluation is a sanity check for the demo, not a production KPI.
    """
    X_test_scaled = pp.scale_features(test_df, scaler)
    raw = -iso_forest.decision_function(X_test_scaled)
    anomaly_score = np.clip(
        (raw - score_range["raw_min"]) / (score_range["raw_max"] - score_range["raw_min"]),
        0,
        1,
    )
    # Demo threshold for "is_anomaly" boolean — see README for full
    # threshold documentation. This is a prototype threshold, not a
    # validated railway safety threshold.
    pred_is_anomaly = (anomaly_score > 0.6).astype(int)
    true_is_anomaly = test_df["is_anomaly_label"].values

    metrics = {
        "precision": float(precision_score(true_is_anomaly, pred_is_anomaly, zero_division=0)),
        "recall": float(recall_score(true_is_anomaly, pred_is_anomaly, zero_division=0)),
        "f1": float(f1_score(true_is_anomaly, pred_is_anomaly, zero_division=0)),
        "n_test_rows": int(len(test_df)),
        "n_true_anomalies": int(true_is_anomaly.sum()),
        "n_flagged_anomalies": int(pred_is_anomaly.sum()),
        "note": (
            "Computed against SYNTHETIC injected-anomaly labels. Demonstration "
            "metric only — not evidence of real-world detection performance."
        ),
    }
    return metrics


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
def build_lstm_model(sequence_length: int, n_features: int):
    """
    Architecture: a single LSTM layer followed by a small dense head that
    regresses `trend_risk_label` (a continuous 0-1 degradation proxy).

    Design choices:
      * Input: (sequence_length=12, n_features=3) — the last 12 scaled
        readings of motor_current_A, vibration_g, temperature_C.
      * Target: trend_risk_label of the LAST timestep — "how degraded does
        this asset look right now, given its recent trajectory".
      * Output activation: sigmoid, because the target is bounded in [0, 1].
      * Loss: mean absolute error (MAE) — robust, directly interpretable in
        the same 0-1 units as the target.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length, n_features)),
            layers.LSTM(32, return_sequences=False),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mae", metrics=["mae"])
    return model


def train_lstm(train_df, val_df, scaler):
    X_train, y_train, _ = pp.build_sequences(train_df, scaler)
    X_val, y_val, _ = pp.build_sequences(val_df, scaler)

    model = build_lstm_model(pp.SEQUENCE_LENGTH, len(pp.FEATURES))

    from tensorflow import keras

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )
    return model, history


def evaluate_lstm(model, scaler, test_df):
    X_test, y_test, _ = pp.build_sequences(test_df, scaler)
    y_pred = model.predict(X_test, verbose=0).reshape(-1)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "n_test_sequences": int(len(y_test)),
        "note": (
            "Computed against a SYNTHETIC degradation-progress label. "
            "Demonstration metric only — not a validated failure probability."
        ),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[1/9] Generating synthetic dataset...")
    df = pp.generate_synthetic_dataset()
    df = pp.clean_and_impute(df)
    df.to_csv(os.path.join(DATA_DIR, "synthetic_sensor_data.csv"), index=False)
    print(f"    -> {len(df)} rows across {df['asset_id'].nunique()} assets")

    print("[2/9] Splitting by asset_id (train/val/test, no leakage)...")
    train_df, val_df, test_df = pp.split_by_asset(df)
    print(f"    -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print("[3/9] Training Isolation Forest...")
    iso_forest, scaler, X_train_normal_scaled = train_isolation_forest(train_df)

    print("[4/9] Computing anomaly score normalization range...")
    score_range = compute_anomaly_score_range(iso_forest, X_train_normal_scaled)
    print(f"    -> {score_range}")

    print("[5/9] Evaluating Isolation Forest on test split...")
    if_metrics = evaluate_isolation_forest(iso_forest, scaler, score_range, test_df)
    print(f"    -> {if_metrics}")

    print("[6/9] Training LSTM (this may take a few minutes)...")
    lstm_model, history = train_lstm(train_df, val_df, scaler)

    print("[7/9] Evaluating LSTM on test split...")
    lstm_metrics = evaluate_lstm(lstm_model, scaler, test_df)
    print(f"    -> {lstm_metrics}")

    print("[8/9] Building SHAP background sample...")
    import explainability as ex

    background = ex.build_background_sample(X_train_normal_scaled)

    print("[9/9] Saving artifacts to models/ ...")
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    np.save(os.path.join(MODELS_DIR, "shap_background.npy"), background)
    lstm_model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))

    with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
        json.dump(
            {
                "features": pp.FEATURES,
                "sequence_length": pp.SEQUENCE_LENGTH,
                "anomaly_score_range": score_range,
                "isolation_forest_metrics": if_metrics,
                "lstm_metrics": lstm_metrics,
            },
            f,
            indent=2,
        )

    print("\nDone. Artifacts saved in models/:")
    for fname in sorted(os.listdir(MODELS_DIR)):
        print(f"  - {fname}")
    print(
        "\nReminder: all evaluation metrics above are computed on SYNTHETIC "
        "data and are demonstration results only."
    )


if __name__ == "__main__":
    main()
