"""
preprocessing.py
=================
Person 3 - ML pipeline for railway point-machine predictive maintenance.

This module is responsible for:
    1. Generating a realistic SYNTHETIC railway point-machine sensor dataset
       (used because no real labelled dataset was supplied).
    2. Cleaning / handling missing values.
    3. Splitting into train / validation / test sets WITHOUT leakage
       (split is done by asset_id, so no asset's timeseries appears in
       more than one split).
    4. Fitting and applying a StandardScaler to the numeric sensor features.
    5. Building fixed-length sequences for the LSTM model.

IMPORTANT — SYNTHETIC DATA DISCLAIMER
--------------------------------------
No real railway point-machine dataset was provided for this task. Every
number produced by this module (anomaly labels, degradation labels,
timestamps, sensor values) is SYNTHETICALLY GENERATED for development and
demonstration purposes only. It does NOT represent real railway operational
data and must not be used to make real safety claims. Replace
`generate_synthetic_dataset()` with a loader for real sensor logs
(e.g. `pd.read_csv(...)`) when real data becomes available — every
downstream function only needs a DataFrame with the same columns, so the
rest of the pipeline does not need to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Fixed configuration shared by training and inference. Keeping this in one
# place means app.py / train.py / explainability.py all agree on the
# contract (feature order, sequence length) with no risk of drift.
# ---------------------------------------------------------------------------
FEATURES = ["motor_current_A", "vibration_g", "temperature_C"]
SEQUENCE_LENGTH = 12  # number of historical readings the LSTM consumes
RANDOM_SEED = 42


def generate_synthetic_dataset(
    n_assets: int = 25,
    cycles_per_asset: int = 4,
    cycle_length: int = 60,
    anomaly_probability: float = 0.03,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic railway point-machine sensor dataset.

    Design of the synthetic data
    -----------------------------
    Each asset goes through several "life cycles". A cycle starts in a
    healthy/normal operating state and gradually degrades (rising motor
    current, vibration and temperature, plus increasing noise) until a
    simulated maintenance event resets it back to a healthy state. This
    mimics real point-machine wear patterns (mechanical wear increases
    friction -> current draw and vibration rise; heat builds up).

    On top of the smooth degradation trend we inject:
      * Gaussian sensor noise (every reading).
      * Sparse anomaly spikes (`anomaly_probability` chance per reading) that
        are short, large deviations unrelated to the slow degradation trend
        (e.g. a jammed point, a sudden electrical fault). These get
        `is_anomaly_label = 1` and are the ground truth used to evaluate the
        Isolation Forest.

    Two label columns are produced, BOTH SYNTHETIC:
      * `is_anomaly_label` (0/1): ground truth for the injected point
        anomalies, used only to evaluate the Isolation Forest in train.py.
      * `trend_risk_label` (0.0-1.0): how far the asset is through its
        current degradation cycle (0 = just serviced/healthy, 1 = at the
        end of the cycle, i.e. maximally degraded). This is the regression
        target for the LSTM. It is a synthetic proxy for "closeness to
        failure/service", not a measured failure probability.

    Returns
    -------
    pd.DataFrame with columns:
        asset_id, timestamp, motor_current_A, vibration_g, temperature_C,
        is_anomaly_label, trend_risk_label
    """
    rng = np.random.default_rng(seed)
    rows = []

    base_time = pd.Timestamp("2026-01-01T00:00:00")

    for asset_idx in range(n_assets):
        asset_id = f"PM-{asset_idx + 1:03d}"
        t_cursor = base_time
        step_index = 0

        # Small per-asset baseline offsets so assets aren't all identical.
        current_base = rng.normal(1.7, 0.1)
        vibration_base = rng.normal(0.30, 0.03)
        temperature_base = rng.normal(30.0, 1.5)

        for _cycle in range(cycles_per_asset):
            for i in range(cycle_length):
                progress = i / (cycle_length - 1)  # 0 -> healthy, 1 -> end of cycle

                # Degradation grows super-linearly near the end of the cycle,
                # mimicking accelerating mechanical wear.
                degradation_factor = progress ** 1.8

                motor_current = (
                    current_base
                    + degradation_factor * 1.6
                    + rng.normal(0, 0.05 + 0.05 * degradation_factor)
                )
                vibration = (
                    vibration_base
                    + degradation_factor * 1.1
                    + rng.normal(0, 0.03 + 0.06 * degradation_factor)
                )
                temperature = (
                    temperature_base
                    + degradation_factor * 20.0
                    + rng.normal(0, 0.5 + 1.0 * degradation_factor)
                )

                is_anomaly = 0
                # Sparse, sharp anomaly spikes independent of the slow trend.
                if rng.random() < anomaly_probability:
                    is_anomaly = 1
                    spike_choice = rng.integers(0, 3)
                    if spike_choice == 0:
                        motor_current += rng.uniform(1.5, 3.0)
                    elif spike_choice == 1:
                        vibration += rng.uniform(1.2, 2.5)
                    else:
                        temperature += rng.uniform(15, 30)

                rows.append(
                    {
                        "asset_id": asset_id,
                        "timestamp": t_cursor.isoformat(),
                        "motor_current_A": round(float(max(motor_current, 0)), 3),
                        "vibration_g": round(float(max(vibration, 0)), 3),
                        "temperature_C": round(float(temperature), 2),
                        "is_anomaly_label": is_anomaly,
                        "trend_risk_label": round(float(np.clip(progress, 0, 1)), 4),
                    }
                )

                t_cursor = t_cursor + pd.Timedelta(minutes=10)
                step_index += 1

    df = pd.DataFrame(rows)
    return df


def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning:
      * Drop exact duplicate rows.
      * Coerce sensor columns to numeric (invalid parses -> NaN).
      * Forward/backward-fill missing sensor values PER ASSET (so we never
        fill one asset's gaps using another asset's data), then fall back
        to the column median for any still-missing values.
    """
    df = df.drop_duplicates().copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["asset_id", "timestamp"])
    df[FEATURES] = df.groupby("asset_id")[FEATURES].transform(
        lambda s: s.ffill().bfill()
    )
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    return df.reset_index(drop=True)


def split_by_asset(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = RANDOM_SEED,
):
    """
    Split the dataset by asset_id (not by row) so that no timeseries is
    split across train/val/test. This is the leakage-prevention strategy:
    a sequence built from an asset's data will only ever appear in exactly
    one of train/val/test.
    """
    rng = np.random.default_rng(seed)
    assets = np.array(sorted(df["asset_id"].unique()), dtype=object)
    rng.shuffle(assets)

    n_train = int(len(assets) * train_frac)
    n_val = int(len(assets) * val_frac)

    train_assets = set(assets[:n_train])
    val_assets = set(assets[n_train : n_train + n_val])
    test_assets = set(assets[n_train + n_val :])

    train_df = df[df["asset_id"].isin(train_assets)].reset_index(drop=True)
    val_df = df[df["asset_id"].isin(val_assets)].reset_index(drop=True)
    test_df = df[df["asset_id"].isin(test_assets)].reset_index(drop=True)

    return train_df, val_df, test_df


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on TRAINING data only (never on val/test)."""
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES].values)
    return scaler


def scale_features(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Apply an already-fitted scaler to a dataframe's feature columns."""
    return scaler.transform(df[FEATURES].values)


def build_sequences(
    df: pd.DataFrame,
    scaler: StandardScaler,
    sequence_length: int = SEQUENCE_LENGTH,
):
    """
    Build fixed-length LSTM input sequences from a (chronologically sorted,
    single-split) dataframe, grouped per asset so sequences never cross
    asset boundaries.

    Returns
    -------
    X : np.ndarray, shape (n_sequences, sequence_length, n_features)
        Scaled sensor readings.
    y : np.ndarray, shape (n_sequences,)
        The `trend_risk_label` of the LAST timestep in each sequence
        (i.e. "how degraded is the asset right now, given its last
        `sequence_length` readings").
    meta : pd.DataFrame
        One row per sequence with asset_id / timestamp of the last reading,
        useful for traceability during evaluation.
    """
    X_list, y_list, meta_rows = [], [], []

    df = df.sort_values(["asset_id", "timestamp"])
    for asset_id, group in df.groupby("asset_id"):
        group = group.reset_index(drop=True)
        scaled = scale_features(group, scaler)
        labels = group["trend_risk_label"].values
        timestamps = group["timestamp"].values

        for end in range(sequence_length - 1, len(group)):
            start = end - sequence_length + 1
            X_list.append(scaled[start : end + 1])
            y_list.append(labels[end])
            meta_rows.append({"asset_id": asset_id, "timestamp": timestamps[end]})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def sequence_payload_to_array(sequence: list[dict], scaler: StandardScaler) -> np.ndarray:
    """
    Convert a list of raw sensor-reading dicts (as received by the API,
    already trimmed/padded to SEQUENCE_LENGTH by the caller) into a scaled
    numpy array of shape (sequence_length, n_features), in the fixed
    FEATURES order.
    """
    raw = np.array([[reading[f] for f in FEATURES] for reading in sequence], dtype=np.float64)
    return scaler.transform(raw)
