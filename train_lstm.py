"""
train_lstm.py
Trains an LSTM that looks at the last WINDOW_SIZE hourly sensor readings
and forecasts the anomaly-risk level 24 hours and 72 hours ahead.

The "risk" target is derived from the already-trained Isolation Forest,
so the LSTM learns to forecast the TREND of that risk score forward in time.

Saves: models/lstm_model.keras, models/feature_scaler.pkl
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

FEATURES = ["current", "vibration", "temperature"]
WINDOW_SIZE = 24     # past 24 hourly readings used as input
HORIZON_24 = 24      # steps ahead for the 24hr forecast
HORIZON_72 = 72      # steps ahead for the 72hr forecast

df = pd.read_csv("sensor_data.csv")

iso_model = joblib.load("models/isoforest.pkl")
iso_scaler = joblib.load("models/scaler.pkl")

# Compute a continuous risk score (0-1) for every row using the Isolation Forest.
X_all = iso_scaler.transform(df[FEATURES])
raw_scores = iso_model.decision_function(X_all)
df["risk"] = np.clip((0.25 - raw_scores) / 0.5, 0, 1)

# Fit a separate scaler for the LSTM's raw sensor inputs
feature_scaler = StandardScaler()
feature_scaler.fit(df[FEATURES])

X_seq, y_seq = [], []

for asset_id, group in df.groupby("asset_id"):
    group = group.reset_index(drop=True)
    feats = feature_scaler.transform(group[FEATURES])
    risk = group["risk"].values

    n = len(group)
    for t in range(WINDOW_SIZE, n - HORIZON_72):
        X_seq.append(feats[t - WINDOW_SIZE:t])
        y_seq.append([risk[t + HORIZON_24], risk[t + HORIZON_72]])

X_seq = np.array(X_seq, dtype="float32")
y_seq = np.array(y_seq, dtype="float32")
print("Training sequences:", X_seq.shape, "Targets:", y_seq.shape)

# simple train/validation split (last 15% of sequences held out)
split = int(len(X_seq) * 0.85)
X_train, X_val = X_seq[:split], X_seq[split:]
y_train, y_val = y_seq[:split], y_seq[split:]

model = keras.Sequential([
    layers.Input(shape=(WINDOW_SIZE, len(FEATURES))),
    layers.LSTM(32, return_sequences=False),
    layers.Dense(16, activation="relu"),
    layers.Dense(2, activation="sigmoid")  # outputs: [risk_in_24hr, risk_in_72hr], both 0-1
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=64,
    verbose=2
)

model.save("models/lstm_model.keras")
joblib.dump(feature_scaler, "models/feature_scaler.pkl")

print("Saved models/lstm_model.keras and models/feature_scaler.pkl")
