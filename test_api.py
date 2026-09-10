"""
test_api.py
===========
A small runnable script that exercises the live API exactly the way n8n
will: an HTTP POST to /predict with the documented JSON schema.

Usage:
    python test_api.py http://localhost:8000
    python test_api.py https://XXXXXXXX.ngrok-free.app
"""

import sys
import json
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

SAMPLE_SEQUENCE = [
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
    {"motor_current_A": 2.85, "vibration_g": 1.35, "temperature_C": 48.9},
]

PAYLOAD = {
    "asset_id": "PM-001",
    "timestamp": "2026-08-26T20:10:00+05:30",
    "sequence": SAMPLE_SEQUENCE,
}


def main():
    print(f"--- GET {BASE_URL}/health ---")
    r = requests.get(f"{BASE_URL}/health", timeout=30)
    print(r.status_code, r.json())

    print(f"\n--- POST {BASE_URL}/predict ---")
    print("Request body:")
    print(json.dumps(PAYLOAD, indent=2))

    r = requests.post(f"{BASE_URL}/predict", json=PAYLOAD, timeout=60)
    print(f"\nStatus: {r.status_code}")
    print("Response body:")
    print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    main()
