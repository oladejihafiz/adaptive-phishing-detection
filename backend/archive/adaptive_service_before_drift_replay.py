
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

from river.drift import ADWIN


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
STATE_DIR = BASE_DIR / "results" / "adaptive_state"

STATE_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_MODEL = MODEL_DIR / "adaptive_model.joblib"
INITIAL_SCALER = MODEL_DIR / "scaler.joblib"

STATE_FILE = STATE_DIR / "learning_state.joblib"


class AdaptiveService:
    def __init__(self):
        self.lock = Lock()

        if STATE_FILE.exists():
            saved = joblib.load(STATE_FILE)

            self.model = saved["model"]
            self.scaler = saved["scaler"]
            self.detector = saved["detector"]
            self.updates = saved["updates"]
            self.errors = saved["errors"]
            self.drift_events = saved["drift_events"]

            print("Restored adaptive learning state.")

        else:
            self.model = joblib.load(INITIAL_MODEL)
            self.scaler = joblib.load(INITIAL_SCALER)

            self.detector = ADWIN(delta=0.002)

            self.updates = 0
            self.errors = 0
            self.drift_events = []

            print("Initialized adaptive learning service.")

        self.feature_names = list(
            self.scaler.feature_names_in_
        )

    def _prepare(self, feature_values):
        frame = pd.DataFrame([feature_values])

        frame = frame[self.feature_names]

        return self.scaler.transform(frame)

    def predict(self, feature_values):
        with self.lock:
            X = self._prepare(feature_values)

            prediction = int(
                self.model.predict(X)[0]
            )

            return {
                "prediction": prediction,
                "status": (
                    "phishing" if prediction == 1
                    else "legitimate"
                ),
                "model": "Passive-Aggressive",
                "learning_updates": self.updates,
                "drift_events": len(self.drift_events)
            }

    def learn(self, feature_values, verified_label):
        if verified_label not in (0, 1):
            raise ValueError(
                "Verified label must be 0 or 1."
            )

        with self.lock:
            X = self._prepare(feature_values)

            # Predict before learning the verified label.
            prediction_before = int(
                self.model.predict(X)[0]
            )

            error = int(
                prediction_before != verified_label
            )

            # Monitor prediction error.
            self.detector.update(error)

            drift_detected = bool(
                self.detector.drift_detected
            )

            # Update the classifier incrementally.
            self.model.partial_fit(
                X,
                np.array([verified_label]),
                classes=np.array([0, 1])
            )

            self.updates += 1
            self.errors += error

            if drift_detected:
                self.drift_events.append({
                    "update": self.updates,
                    "error_count": self.errors
                })

            # Save state for subsequent API restarts.
            joblib.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "detector": self.detector,
                    "updates": self.updates,
                    "errors": self.errors,
                    "drift_events": self.drift_events
                },
                STATE_FILE
            )

            return {
                "prediction_before_update": prediction_before,
                "verified_label": verified_label,
                "prediction_error": error,
                "learning_completed": True,
                "total_updates": self.updates,
                "total_errors": self.errors,
                "adwin_drift_detected": drift_detected,
                "total_drift_events": len(self.drift_events)
            }

    def status(self):
        with self.lock:
            return {
                "model": "Passive-Aggressive",
                "learning_updates": self.updates,
                "prediction_errors": self.errors,
                "drift_events": len(self.drift_events),
                "recent_drift_events": self.drift_events[-5:],
                "adwin_window_width": self.detector.width
            }


adaptive_service = AdaptiveService()
