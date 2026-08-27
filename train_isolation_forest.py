"""
train_isolation_forest.py
Trains an Isolation Forest on healthy sensor data to learn what "normal" looks like.
Saves: models/isoforest.pkl, models/scaler.pkl
"""

import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = ["current", "vibration", "temperature"]

df = pd.read_csv("sensor_data.csv")

# Isolation Forest should mostly learn from HEALTHY data (label == 0)
healthy_df = df[df["label"] == 0]

scaler = StandardScaler()
X_train = scaler.fit_transform(healthy_df[FEATURES])

# contamination = expected fraction of outliers in a typical batch; 0.05 is a safe demo default
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)
model.fit(X_train)

joblib.dump(model, "models/isoforest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# quick sanity check: score a few healthy vs degraded rows
degraded_df = df[df["label"] == 1].head(5)
healthy_sample = healthy_df.head(5)

for name, sample in [("HEALTHY sample", healthy_sample), ("DEGRADED sample", degraded_df)]:
    X = scaler.transform(sample[FEATURES])
    scores = model.decision_function(X)   # higher = more normal
    risk = 1 - (scores - (-0.5)) / 1.0    # rough squashing just for a sanity print
    print(name, "raw decision_function scores:", scores.round(3))

print("Saved models/isoforest.pkl and models/scaler.pkl")
