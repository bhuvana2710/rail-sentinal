"""
generate_data.py
Creates a synthetic dataset of point-machine sensor readings.
Simulates many "assets" (point machines). Most stay healthy the whole time.
Some assets degrade: current creeps up -> vibration follows -> temperature rises.
This mirrors the real-world failure order described in the project brief.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_ASSETS = 40
HOURS_PER_ASSET = 24 * 20  # 20 days of hourly readings per asset
DEGRADING_FRACTION = 0.35  # ~35% of assets will show a degradation curve

rows = []

for asset_id in range(N_ASSETS):
    is_degrading = np.random.rand() < DEGRADING_FRACTION

    # baseline healthy values
    base_current = np.random.normal(1.4, 0.05)      # amps
    base_vibration = np.random.normal(0.30, 0.03)   # g
    base_temp = np.random.normal(38.0, 1.0)         # deg C

    if is_degrading:
        # pick a random point in the timeline where degradation starts
        onset_hour = np.random.randint(int(HOURS_PER_ASSET * 0.4), int(HOURS_PER_ASSET * 0.8))
    else:
        onset_hour = None

    for hour in range(HOURS_PER_ASSET):
        noise_c = np.random.normal(0, 0.02)
        noise_v = np.random.normal(0, 0.015)
        noise_t = np.random.normal(0, 0.3)

        if is_degrading and hour >= onset_hour:
            progress = (hour - onset_hour) / max(1, (HOURS_PER_ASSET - onset_hour))
            # current rises first, vibration follows with a lag, temperature rises last
            current = base_current + 0.9 * progress + noise_c
            vibration = base_vibration + 0.6 * max(0, progress - 0.15) + noise_v
            temperature = base_temp + 15 * max(0, progress - 0.35) + noise_t
            label = 1 if progress > 0.5 else 0  # "1" = confirmed fault window
        else:
            current = base_current + noise_c
            vibration = base_vibration + noise_v
            temperature = base_temp + noise_t
            label = 0

        rows.append({
            "asset_id": f"PM-{asset_id:03d}",
            "hour": hour,
            "current": round(float(current), 4),
            "vibration": round(float(vibration), 4),
            "temperature": round(float(temperature), 4),
            "label": label
        })

df = pd.DataFrame(rows)
df.to_csv("sensor_data.csv", index=False)
print(f"Saved sensor_data.csv with {len(df)} rows across {N_ASSETS} assets.")
print(df.head())
