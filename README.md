h# Person 3 — ML, Prediction API & Explainability
Railway point-machine predictive maintenance — anomaly detection, degradation trend prediction, and explainability, exposed as an HTTP API for the n8n automation layer.

This README is written for the n8n owner (integration) and for anyone re-running/retraining the ML system. Scope: **only** Person 3's work (Isolation Forest, LSTM, SHAP, FastAPI). Person 1 (hardware) and Person 4 (GenAI explanation) are out of scope and not implemented here.

---

## 0. Read this first — data & threshold disclaimers

- **No real point-machine dataset was supplied.** `preprocessing.py` generates a **synthetic** dataset with normal, anomalous, and degrading conditions. All labels (`is_anomaly_label`, `trend_risk_label`) are synthetic proxies, not measured failures.
- All evaluation metrics (precision/recall/F1, MAE/RMSE) in this README and in `models/config.json` are **demonstration results on synthetic data**, not evidence of real-world railway safety performance.
- **Risk thresholds (LOW/MEDIUM/HIGH) and `forecast_hours` are prototype/demo values**, not scientifically validated railway safety thresholds. See §6.
- Replace `generate_synthetic_dataset()` in `preprocessing.py` with a loader for real sensor logs when real data is available — everything downstream (scaling, sequences, training, API) consumes the same DataFrame schema and does not need to change.

---

## 1. Architecture

```
Sensor Data (asset_id, timestamp, sequence of motor_current_A / vibration_g / temperature_C)
     │
     ▼
Preprocessing  (preprocessing.py: cleaning, StandardScaler, fixed-length sequences)
     │
     ├──────────────► Isolation Forest ──► anomaly_score, is_anomaly
     │
     ├──────────────► LSTM ──────────────► trend_risk
     │
     └──────────────► SHAP (on Isolation Forest) ──► feature contributions
     │
     ▼
Risk Assessment (predict.py: combine anomaly_score + trend_risk → risk_level, forecast_hours)
     │
     ▼
FastAPI (app.py)
     │
     ▼
POST /predict  ──►  n8n HTTP Request node
```

Models are **trained once** (`train.py`) and **loaded, not retrained**, on every `/predict` call (`predict.py` loads artifacts at process startup).

---

## 2. Project structure

```
person3_ml/
├── app.py                 # FastAPI app: POST /predict, GET /health
├── train.py                # Full training pipeline (run this first)
├── predict.py               # Loads saved artifacts once, runs combined prediction
├── preprocessing.py         # Synthetic data gen, cleaning, scaling, sequencing
├── explainability.py        # SHAP KernelExplainer wrapper around the Isolation Forest
├── test_api.py               # Runnable script: calls /health and /predict on a live API
├── colab_deploy.ipynb        # Colab notebook: train + launch API + ngrok, end to end
├── data/                      # Generated synthetic dataset (train.py writes here)
│   └── synthetic_sensor_data.csv
├── models/                    # Trained artifacts (train.py writes here; app.py reads here)
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── lstm_model.keras
│   ├── shap_background.npy
│   └── config.json
├── requirements.txt
└── README.md
```

---

## 3. How each model works

### 3.1 Isolation Forest (anomaly detection)

**Training:** trained on the training split's rows that are *not* injected spike-anomalies (`is_anomaly_label == 0`), scaled with a `StandardScaler` fit on that same data.

**How it works:** Isolation Forest builds many random binary trees that isolate each point by repeatedly splitting on a random feature at a random value. Points that are "few and different" (anomalies) typically need **fewer** random splits to end up alone in a leaf than normal points, which are surrounded by similar points and need many splits to separate. The model's raw output is therefore related to the average path length to isolate a point across all trees: **short path → anomalous, long path → normal**.

**Turning that into `anomaly_score` (0–1, higher = worse):**
```
raw = -IsolationForest.decision_function(x)     # flip sklearn's convention (higher=normal) so higher=worse
anomaly_score = clip((raw - raw_min) / (raw_max - raw_min), 0, 1)
```
`raw_min`/`raw_max` are the 5th/95th percentile of `raw` over the **training** distribution, computed once in `train.py` and saved in `models/config.json` — never recomputed per request, so scores stay comparable across requests.

`is_anomaly = anomaly_score > 0.6` (demo threshold, see §6).

**Evaluation (synthetic):** Because the "normal" training population includes both healthy and heavily-degraded-but-not-spiked readings, the Isolation Forest also (correctly) flags severely degraded readings as unusual — even though the synthetic ground truth only labels the injected *spikes* as `is_anomaly_label=1`. This means **recall against the spike-only synthetic label is high, but precision looks low** — this is a labeling-definition artifact of the demo, not a modeling bug, and is documented here rather than hidden. Real deployments should evaluate against real confirmed-fault labels.

### 3.2 LSTM (degradation trend)

- **Input:** the last `SEQUENCE_LENGTH = 12` scaled readings of `[motor_current_A, vibration_g, temperature_C]`.
- **Target:** `trend_risk_label` — a **synthetic** 0 (just serviced) → 1 (end of degradation cycle) proxy built into the synthetic data generator.
- **Architecture:** `LSTM(32) → Dense(16, relu) → Dropout(0.1) → Dense(1, sigmoid)`.
- **Loss / metrics:** MAE (mean absolute error), directly interpretable in the same 0–1 units as the target.
- **Training:** Adam optimizer, early stopping on validation loss (patience 8), up to 60 epochs.
- **Output interpretation:** `trend_risk` close to 0 → recent trajectory looks healthy; close to 1 → recent trajectory looks like the end of a degradation cycle in the synthetic data. **This is a synthetic degradation proxy, not a measured failure probability.**

### 3.3 SHAP (explainability)

We explain the **Isolation Forest's** anomaly score (not the LSTM) — see the full rationale in `explainability.py`'s docstring; in short, the Isolation Forest's output is over the 3 *current* sensor features, which maps directly onto "why is this reading flagged", while decomposing the LSTM's output across all 12 timesteps × 3 features is far less actionable for a maintenance engineer.

- **Method:** `shap.KernelExplainer`, a model-agnostic Shapley-value estimator, run against a saved background sample (60 rows) from real training data. Chosen over `TreeExplainer` because TreeExplainer's support for `IsolationForest` is version-fragile; KernelExplainer works with any callable and is robust across library versions.
- **Meaning:** `anomaly_score(x) ≈ base_value + Σ shap_values`. A **positive** SHAP value means that feature pushed the score **up** (more anomalous); **negative** means it pushed the score **down** (more normal). Magnitude = influence, in the same 0–1-ish units as `anomaly_score`.
- **Limitations:** KernelExplainer values are local approximations based on a finite background sample; they describe this model's behavior on this instance, not a guaranteed physical root cause.

---

## 4. Risk assessment (combining everything)

```
combined_risk = 0.5 * anomaly_score + 0.5 * trend_risk

combined_risk <  0.35            → risk_level = "LOW"
0.35 <= combined_risk < 0.70     → risk_level = "MEDIUM"
combined_risk >= 0.70            → risk_level = "HIGH"
```

**These thresholds and the 50/50 weighting are prototype/demo values** chosen for a readable 3-tier output. They are **not** derived from real railway safety data or any regulatory standard. A real deployment must recalibrate them against confirmed maintenance/failure events (e.g. via ROC analysis).

`forecast_hours` is a simple **demo heuristic**, not a fitted remaining-useful-life model: it assumes `trend_risk` keeps growing at the same average rate it grows at in the synthetic training cycles, and estimates hours until it would reach 1.0. It must be replaced with a model fitted on real degradation-to-failure data before it informs any real scheduling decision.

---

## 5. API contract

| | |
|---|---|
| **HTTP Method** | `POST` |
| **Endpoint** | `/predict` |
| **Content-Type** | `application/json` |
| **Authentication** | None (demo). See §11 for how to add one. |

### Request schema

```json
{
  "asset_id": "PM-001",
  "timestamp": "2026-08-26T20:10:00+05:30",
  "sequence": [
    { "motor_current_A": 1.82, "vibration_g": 0.34, "temperature_C": 34.8 },
    { "motor_current_A": 1.90, "vibration_g": 0.41, "temperature_C": 35.9 },
    { "motor_current_A": 1.95, "vibration_g": 0.47, "temperature_C": 36.7 },
    { "motor_current_A": 2.05, "vibration_g": 0.55, "temperature_C": 38.1 },
    { "motor_current_A": 2.15, "vibration_g": 0.63, "temperature_C": 39.8 },
    { "motor_current_A": 2.28, "vibration_g": 0.72, "temperature_C": 41.5 },
    { "motor_current_A": 2.42, "vibration_g": 0.81, "temperature_C": 43.4 },
    { "motor_current_A": 2.55, "vibration_g": 0.90, "temperature_C": 45.0 },
    { "motor_current_A": 2.68, "vibration_g": 1.00, "temperature_C": 46.6 },
    { "motor_current_A": 2.75, "vibration_g": 1.10, "temperature_C": 47.5 },
    { "motor_current_A": 2.80, "vibration_g": 1.22, "temperature_C": 48.0 },
    { "motor_current_A": 2.85, "vibration_g": 1.35, "temperature_C": 48.9 }
  ]
}
```

Rules (enforced by pydantic in `app.py`, return `422` on violation):
- `asset_id`: non-empty string.
- `timestamp`: ISO-8601 string.
- `sequence`: array of `{motor_current_A, vibration_g, temperature_C}` (all numeric), **ordered oldest → newest**, **minimum 12 entries** (`SEQUENCE_LENGTH`). If more than 12 are sent, only the most recent 12 are used. The **last** entry is treated as the current/latest reading (used for the Isolation Forest score and SHAP).

### Response schema

```json
{
  "asset_id": "PM-001",
  "timestamp": "2026-08-26T20:10:00+05:30",
  "anomaly_score": 0.78,
  "is_anomaly": true,
  "trend_risk": 0.81,
  "risk_level": "HIGH",
  "forecast_hours": 3.1,
  "shap": {
    "motor_current_A": 0.31,
    "vibration_g": 0.24,
    "temperature_C": 0.09
  }
}
```

| Field | Meaning |
|---|---|
| `anomaly_score` | 0 (normal) – 1 (highly anomalous), from Isolation Forest on the latest reading |
| `is_anomaly` | `anomaly_score > 0.6` (demo threshold) |
| `trend_risk` | 0 (healthy trajectory) – 1 (end-of-cycle degraded trajectory), from LSTM on the 12-reading window |
| `risk_level` | `"LOW"` / `"MEDIUM"` / `"HIGH"`, from `combined_risk` (§4) |
| `forecast_hours` | Demo heuristic: estimated hours until `trend_risk` would reach 1.0 if it keeps growing at the synthetic training rate |
| `shap` | Per-feature SHAP contribution to `anomaly_score` (sign = direction, magnitude = influence) |

> ⚠️ The numeric values above (`0.78`, `0.81`, `3.1`, …) are **illustrative** — this response schema is fixed, but the actual numbers depend on the trained model, which is non-deterministic in minor ways (random forest, random weight init) across training runs. Run `python test_api.py <your-api-url>` after deployment (§9) to see a real, live response.

---

## 6. Error handling

| Situation | Response |
|---|---|
| Missing/invalid field (e.g. missing `asset_id`, non-numeric sensor value, bad timestamp) | `422 Unprocessable Entity` with a JSON body describing which field failed (standard FastAPI/pydantic validation error) |
| `sequence` shorter than 12 readings | `422 Unprocessable Entity`, `"sequence must contain at least 12 readings, got N"` |
| Models not trained yet (`models/` missing) | `503 Service Unavailable` |
| Any other unexpected error | `500 Internal Server Error` with `{"error": "internal_server_error", "detail": "..."}` (JSON, never an HTML traceback) |

## 7. Health check

```
GET /health
```
```json
{ "status": "ok", "model_loaded": true }
```

---

## 8. Running locally (no Colab)

```bash
cd person3_ml
pip install -r requirements.txt
python train.py                 # trains + saves models/ (run once, or whenever retraining is needed)
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test it:
```bash
python test_api.py http://localhost:8000
```

---

## 9. Colab + ngrok deployment (recommended for the demo)

Open `colab_deploy.ipynb` in Google Colab and run the cells **in order**:

1. **Install dependencies** — `pip install fastapi uvicorn[standard] scikit-learn tensorflow shap joblib pyngrok pandas numpy requests`
2. **Upload the project** — put the whole `person3_ml/` folder at `/content/person3_ml/` in Colab's file browser (or `git clone` it there).
3. **Train** — runs `train.py`, which generates the synthetic dataset and saves `models/`.
4. **Confirm artifacts** — lists `models/` to confirm `isolation_forest.pkl`, `scaler.pkl`, `lstm_model.keras`, `shap_background.npy`, `config.json` exist.
5. **Start FastAPI** — runs `uvicorn.run("app:app", ...)` in a background thread inside the notebook process, on port 8000.
6. **Local sanity check** — calls `http://localhost:8000/health` inside Colab.
7. **Start ngrok** — `ngrok.connect(8000, "http")`, prints the public URL.
8. **Public test** — calls `/health` and `/predict` through the public ngrok URL and prints the exact URL to give to the n8n workflow.

**The public URL can only be known after Cell 7 runs** (ngrok assigns it dynamically). Everywhere in this README a URL is shown as `https://XXXXXXXX.ngrok-free.app`, replace it with whatever Cell 7/8 prints.

Keep the Colab runtime alive for as long as n8n needs to reach the API — closing the tab disconnects ngrok.

---

## 10. n8n HTTP Request node configuration

```
Method:          POST
URL:             https://XXXXXXXX.ngrok-free.app/predict     ← replace with the URL printed by Colab Cell 7/8
Authentication:  None (see §11 to add one)
Send Body:       JSON
Content-Type:    application/json
```

**Body (JSON, "Using JSON" mode in n8n):**
```json
{
  "asset_id": "={{ $json.asset_id }}",
  "timestamp": "={{ $json.timestamp }}",
  "sequence": "={{ $json.sequence }}"
}
```
Map `$json.asset_id`, `$json.timestamp`, `$json.sequence` to whatever upstream node/field in your n8n workflow holds the last 12 (or more) historical sensor readings for that asset, in the exact `{motor_current_A, vibration_g, temperature_C}` shape shown in §5. If your upstream data only has flat rolling-window fields, use an n8n **Function**/**Set** node just before this HTTP Request node to assemble the `sequence` array in that shape.

Downstream, the response fields (`anomaly_score`, `is_anomaly`, `trend_risk`, `risk_level`, `forecast_hours`, `shap.*`) are available as `{{$json.risk_level}}` etc. for your alerting logic or for Person 4's GenAI node to consume.

---

## 11. Adding authentication (optional, recommended for anything beyond a demo)

The demo API has **no authentication** (`Authentication: None` above) to keep the n8n setup simple. To add a shared-secret header check:

```python
# in app.py
API_KEY = os.environ.get("PREDICT_API_KEY")

@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return await call_next(request)
```
Then in n8n's HTTP Request node, add a header `x-api-key: <the same secret>`.

---

## 12. Working sample request/response

```bash
curl -X POST "https://XXXXXXXX.ngrok-free.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "PM-001",
    "timestamp": "2026-08-26T20:10:00+05:30",
    "sequence": [
      {"motor_current_A": 1.82, "vibration_g": 0.34, "temperature_C": 34.8},
      {"motor_current_A": 1.90, "vibration_g": 0.41, "temperature_C": 35.9},
      {"motor_current_A": 1.95, "vibration_g": 0.47, "temperature_C": 36.7},
      {"motor_current_A": 2.05, "vibration_g": 0.55, "temperature_C": 38.1},
      {"motor_current_A": 2.15, "vibration_g": 0.63, "temperature_C": 39.8},
      {"motor_current_A": 2.28, "vibration_g": 0.72, "temperature_C": 41.5},
      {"motor_current_A": 2.42, "vibration_g": 0.81, "temperature_C": 43.4},
      {"motor_current_A": 2.55, "vibration_g": 0.90, "temperature_C": 45.0},
      {"motor_current_A": 2.68, "vibration_g": 1.00, "temperature_C": 46.6},
      {"motor_current_A": 2.75, "vibration_g": 1.10, "temperature_C": 47.5},
      {"motor_current_A": 2.80, "vibration_g": 1.22, "temperature_C": 48.0},
      {"motor_current_A": 2.85, "vibration_g": 1.35, "temperature_C": 48.9}
    ]
  }'
```

**Example response** (illustrative — the input sequence shows a clear rising trend in all three sensors, which the pipeline is designed to flag; run `python test_api.py <url>` for the exact live numbers from your trained model):

```json
{
  "asset_id": "PM-001",
  "timestamp": "2026-08-26T20:10:00+05:30",
  "anomaly_score": 0.78,
  "is_anomaly": true,
  "trend_risk": 0.81,
  "risk_level": "HIGH",
  "forecast_hours": 3.1,
  "shap": {
    "motor_current_A": 0.31,
    "vibration_g": 0.24,
    "temperature_C": 0.09
  }
}
```

This exact request body is also encoded in `test_api.py`, so you can get a real, live version of this response by running:
```bash
python test_api.py https://XXXXXXXX.ngrok-free.app
```

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `ngrok URL changed` | Free ngrok URLs are re-assigned every time the tunnel restarts. Re-run Colab Cell 7/8, get the new URL, update the n8n HTTP Request node. |
| `API not running` / connection refused | Check the Colab runtime hasn't disconnected/timed out; re-run cells 5–7. For local/Docker, confirm the `uvicorn`/container process is up. |
| `model file missing` (`FileNotFoundError` on startup, `/health` returns `model_loaded: false`) | Run `python train.py` before starting the API — models are not trained automatically at boot. |
| `wrong JSON format` | Compare your payload to §5's schema exactly — field names are case-sensitive (`motor_current_A`, not `Motor_Current`). |
| `insufficient sequence length` | `sequence` must have at least 12 entries. Pad your n8n workflow's rolling window or wait until 12 historical readings exist for that asset. |
| `422 Unprocessable Entity` | The response body's `detail` field names the exact offending field — check for missing fields, non-numeric values, or a malformed `timestamp`. |
| `connection refused` from n8n specifically | Confirm the ngrok URL is reachable from a browser first; some corporate/n8n-cloud networks block plain HTTP — use the `https://` ngrok URL, not `http://`. |
| `Colab runtime stopped` | Colab free tier disconnects after inactivity or ~12h. Re-run the notebook top to bottom (train.py doesn't need to be re-run if `models/` files were downloaded and re-uploaded, but must be re-run if the runtime was fully reset and `models/` is gone). |

---

## 14. Model evaluation summary

Full numbers are written to `models/config.json` after `train.py` runs. Both blocks below are computed on the **synthetic** test split (held out by asset, no leakage) and are demonstration results only:

- **Isolation Forest**: precision / recall / F1 against the synthetic injected-spike labels (see §3.1 for why precision reads low against this narrow definition — recall against true spikes is the more meaningful number here).
- **LSTM**: MAE / RMSE against the synthetic `trend_risk_label`.

> These evaluation results are demonstration results based on synthetic data and are not evidence of production railway safety performance.

---

# FIVE THINGS I MUST SEND TO THE n8n PERSON

1. **ML API URL:** `https://XXXXXXXX.ngrok-free.app/predict`  ← exact value only known after running Colab Cell 7/8 (or after your `localhost`/Docker deployment is up); insert it here once known.
2. **HTTP method:** `POST`
3. **Exact request JSON:**
   ```json
   {
     "asset_id": "PM-001",
     "timestamp": "2026-08-26T20:10:00+05:30",
     "sequence": [
       {"motor_current_A": 1.82, "vibration_g": 0.34, "temperature_C": 34.8},
       {"motor_current_A": 1.90, "vibration_g": 0.41, "temperature_C": 35.9},
       {"motor_current_A": 1.95, "vibration_g": 0.47, "temperature_C": 36.7},
       {"motor_current_A": 2.05, "vibration_g": 0.55, "temperature_C": 38.1},
       {"motor_current_A": 2.15, "vibration_g": 0.63, "temperature_C": 39.8},
       {"motor_current_A": 2.28, "vibration_g": 0.72, "temperature_C": 41.5},
       {"motor_current_A": 2.42, "vibration_g": 0.81, "temperature_C": 43.4},
       {"motor_current_A": 2.55, "vibration_g": 0.90, "temperature_C": 45.0},
       {"motor_current_A": 2.68, "vibration_g": 1.00, "temperature_C": 46.6},
       {"motor_current_A": 2.75, "vibration_g": 1.10, "temperature_C": 47.5},
       {"motor_current_A": 2.80, "vibration_g": 1.22, "temperature_C": 48.0},
       {"motor_current_A": 2.85, "vibration_g": 1.35, "temperature_C": 48.9}
     ]
   }
   ```
   (minimum 12 entries in `sequence`, oldest → newest; more than 12 is accepted, only the most recent 12 are used)
4. **Exact response JSON:**
   ```json
   {
     "asset_id": "PM-001",
     "timestamp": "2026-08-26T20:10:00+05:30",
     "anomaly_score": 0.78,
     "is_anomaly": true,
     "trend_risk": 0.81,
     "risk_level": "HIGH",
     "forecast_hours": 3.1,
     "shap": {
       "motor_current_A": 0.31,
       "vibration_g": 0.24,
       "temperature_C": 0.09
     }
   }
   ```
   (numbers illustrative — schema is fixed; run `test_api.py` against the live URL for real numbers)
5. **Deployment:** `Colab + ngrok` (recommended for the demo — see `colab_deploy.ipynb`). `localhost` and `Docker` are also fully supported (§8) with no code changes.

**Working sample request:** see §12's `curl` command.

**Working sample response:** see §12's example JSON (or run `python test_api.py <url>` for a real live one).
