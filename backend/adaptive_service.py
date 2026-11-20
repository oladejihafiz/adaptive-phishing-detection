
# Path helps us locate project files reliably.
from pathlib import Path

# Lock prevents two API requests from updating the model simultaneously.
from threading import Lock

# deque stores a limited number of recently verified examples.
from collections import deque

# os.replace helps us save the learning state more safely.
import os
import tempfile

# joblib saves and loads trained machine-learning objects.
import joblib

# NumPy and pandas prepare feature values for the classifier.
import numpy as np
import pandas as pd

# ADWIN detects changes in the stream of prediction errors.
from river.drift import ADWIN


# ------------------------------------------------------------
# 1. PROJECT PATHS AND MODEL CONFIGURATION
# ------------------------------------------------------------

# Find the project's root directory.
BASE_DIR = Path(__file__).resolve().parent.parent

# Locate the models directory.
MODEL_DIR = BASE_DIR / "models"

# Locate the directory used to save adaptive learning state.
STATE_DIR = BASE_DIR / "results" / "adaptive_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Load the separately trained 29-feature adaptive model.
INITIAL_MODEL = MODEL_DIR / "adaptive_model_29.joblib"
INITIAL_SCALER = MODEL_DIR / "scaler_29.joblib"

# IMPORTANT: Keep the 29-feature learning history separate.
# This prevents accidentally loading the old 15-feature state.
STATE_FILE = STATE_DIR / "learning_state_29.joblib"

# The candidate model must use exactly 29 input features.
EXPECTED_FEATURE_COUNT = 29


# ------------------------------------------------------------
# 2. ADAPTIVE LEARNING CONFIGURATION
# ------------------------------------------------------------

# Keep only the 128 most recent verified examples.
BUFFER_SIZE = 128

# On detected drift, replay at most 32 previous examples.
REPLAY_SIZE = 32

# Require at least 16 verified examples before replay.
MIN_REPLAY_EXAMPLES = 16


class AdaptiveService:

    def __init__(self):
        # Protect predictions, training, and state updates.
        self.lock = Lock()

        if STATE_FILE.exists():

            # Restore the dedicated 29-feature learning state.
            # Only load joblib files created by this trusted application.
            saved = joblib.load(STATE_FILE)

            self.model = saved["model"]
            self.scaler = saved["scaler"]
            self.detector = saved["detector"]

            self.updates = saved["updates"]
            self.errors = saved["errors"]
            self.drift_events = saved["drift_events"]

            self.replay_updates = saved.get("replay_updates", 0)

            # Restore recent verified examples.
            self.recent_examples = deque(
                saved.get("recent_examples", []),
                maxlen=BUFFER_SIZE
            )

            # Restore the saved feature order.
            saved_feature_names = saved.get("feature_names")

            if saved_feature_names is None:
                raise RuntimeError(
                    "The saved 29-feature learning state has no "
                    "feature-order information. Archive it and "
                    "initialize a fresh state."
                )

            self.feature_names = list(saved_feature_names)

            # Confirm that the restored state belongs to this model.
            self._validate_model_and_scaler()

            # Confirm that the saved feature order is still compatible.
            scaler_names = list(self.scaler.feature_names_in_)

            if self.feature_names != scaler_names:
                raise RuntimeError(
                    "Saved learning-state feature order does not "
                    "match the scaler's feature order."
                )

            print("Restored 29-feature adaptive learning state.")

        else:

            # Confirm the expected model and scaler files exist.
            if not INITIAL_MODEL.exists():
                raise FileNotFoundError(
                    f"29-feature adaptive model not found: {INITIAL_MODEL}"
                )

            if not INITIAL_SCALER.exists():
                raise FileNotFoundError(
                    f"29-feature adaptive scaler not found: {INITIAL_SCALER}"
                )

            # Load the separately trained 29-feature classifier.
            self.model = joblib.load(INITIAL_MODEL)

            # Load the scaler fitted during the same training process.
            self.scaler = joblib.load(INITIAL_SCALER)

            # Preserve the exact feature order used during training.
            self.feature_names = list(
                self.scaler.feature_names_in_
            )

            # Verify the model and scaler before accepting requests.
            self._validate_model_and_scaler()

            # Monitor prediction errors for concept drift.
            self.detector = ADWIN(delta=0.002)

            # Counters for new verified feedback.
            self.updates = 0
            self.errors = 0
            self.drift_events = []

            # Counters and storage for drift-triggered replay.
            self.replay_updates = 0
            self.recent_examples = deque(maxlen=BUFFER_SIZE)

            print("Initialized fresh 29-feature adaptive service.")

    # --------------------------------------------------------
    # 3. MODEL AND SCALER COMPATIBILITY CHECKS
    # --------------------------------------------------------

    def _validate_model_and_scaler(self):
        # Ensure the scaler exposes the feature names used in training.
        if not hasattr(self.scaler, "feature_names_in_"):
            raise RuntimeError(
                "The adaptive scaler does not contain feature names."
            )

        # Read the scaler's expected feature order.
        scaler_names = list(self.scaler.feature_names_in_)

        # Ensure the expected feature count is exactly 29.
        if len(scaler_names) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_FEATURE_COUNT} scaler features, "
                f"but found {len(scaler_names)}."
            )

        # Ensure all feature names are unique.
        if len(set(scaler_names)) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "The adaptive scaler contains duplicate feature names."
            )

        # Ensure the classifier expects the same number of features.
        if self.model.n_features_in_ != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "The adaptive classifier does not expect 29 features."
            )

        # Ensure the model supports both legitimate and phishing labels.
        if not np.array_equal(
            self.model.classes_,
            np.array([0, 1])
        ):
            raise RuntimeError(
                "The adaptive classifier must support classes 0 and 1."
            )

    # --------------------------------------------------------
    # 4. FEATURE PREPARATION
    # --------------------------------------------------------

    def _prepare(self, feature_values):
        # Ensure the backend supplied a dictionary of URL features.
        if not isinstance(feature_values, dict):
            raise TypeError(
                "Adaptive features must be supplied as a dictionary."
            )

        # Detect missing or unexpected features.
        supplied_names = set(feature_values.keys())
        expected_names = set(self.feature_names)

        missing = expected_names - supplied_names
        unexpected = supplied_names - expected_names

        if missing or unexpected:
            raise ValueError(
                "Adaptive feature mismatch. "
                f"Missing: {sorted(missing)}; "
                f"Unexpected: {sorted(unexpected)}"
            )

        # Convert URL features into a one-row DataFrame.
        frame = pd.DataFrame([feature_values])

        # Match the exact feature order used during training.
        frame = frame[self.feature_names]

        # Apply the existing fitted scaler.
        return self.scaler.transform(frame)

    # --------------------------------------------------------
    # 5. SAVE LEARNING STATE SAFELY
    # --------------------------------------------------------

    def _save_state(self):
        # Store everything needed to resume adaptive learning.
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "detector": self.detector,
            "updates": self.updates,
            "errors": self.errors,
            "drift_events": self.drift_events,
            "replay_updates": self.replay_updates,
            "recent_examples": list(self.recent_examples),

            # Save feature order to detect incompatible state files.
            "feature_names": self.feature_names
        }

        # Create a temporary file in the same directory.
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".joblib",
                prefix="learning_state_29_",
                dir=STATE_DIR,
                delete=False
            ) as temporary_file:

                temporary_path = Path(temporary_file.name)

            # Save the complete new state to the temporary file.
            joblib.dump(state, temporary_path)

            # Replace the previous state only after saving succeeds.
            os.replace(temporary_path, STATE_FILE)

        finally:
            # Clean up the temporary file if saving failed.
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    # --------------------------------------------------------
    # 6. REPLAY RECENT VERIFIED EXAMPLES
    # --------------------------------------------------------

    def _replay_recent_examples(self):
        # Do not replay if we have too little verified feedback.
        if len(self.recent_examples) < MIN_REPLAY_EXAMPLES:
            return 0

        # Select up to 32 previous verified examples.
        examples = list(self.recent_examples)[-REPLAY_SIZE:]

        # Extract feature dictionaries and verified labels.
        feature_rows = [
            example["features"]
            for example in examples
        ]

        labels = np.array([
            example["label"]
            for example in examples
        ])

        # Build a DataFrame using the original feature order.
        frame = pd.DataFrame(feature_rows)
        frame = frame[self.feature_names]

        # Scale using the existing fitted scaler.
        X = self.scaler.transform(frame)

        # Perform one additional incremental training pass.
        self.model.partial_fit(
            X,
            labels,
            classes=np.array([0, 1])
        )

        # Count replayed examples separately from new feedback.
        replayed_count = len(examples)
        self.replay_updates += replayed_count

        return replayed_count

    # --------------------------------------------------------
    # 7. PREDICTION WITHOUT TRAINING
    # --------------------------------------------------------

    def predict(self, feature_values):
        with self.lock:

            # Prepare features without changing the model.
            X = self._prepare(feature_values)

            # Predict 0 (legitimate) or 1 (phishing).
            prediction = int(
                self.model.predict(X)[0]
            )

            return {
                "prediction": prediction,
                "status": (
                    "phishing"
                    if prediction == 1
                    else "legitimate"
                ),
                "model": "Passive-Aggressive (29 features)",
                "learning_updates": self.updates,
                "drift_events": len(self.drift_events)
            }

    # --------------------------------------------------------
    # 8. LEARN FROM VERIFIED FEEDBACK
    # --------------------------------------------------------

    def learn(self, feature_values, verified_label):

        # Accept only the two supported class labels.
        if verified_label not in (0, 1):
            raise ValueError(
                "Verified label must be 0 or 1."
            )

        with self.lock:

            # Prepare the verified URL's features.
            X = self._prepare(feature_values)

            # Predict BEFORE learning the correct label.
            prediction_before = int(
                self.model.predict(X)[0]
            )

            # An error is 1 when prediction and label differ.
            error = int(
                prediction_before != verified_label
            )

            # Feed the pre-update error to ADWIN.
            self.detector.update(error)

            # Check whether ADWIN detected drift.
            drift_detected = bool(
                self.detector.drift_detected
            )

            # Always learn from the newly verified example.
            self.model.partial_fit(
                X,
                np.array([verified_label]),
                classes=np.array([0, 1])
            )

            # Count this as one new verified learning update.
            self.updates += 1
            self.errors += error

            replayed_count = 0

            if drift_detected:
                # Replay older examples before adding the current one.
                # This prevents the current example from being trained
                # twice during the same learning operation.
                replayed_count = self._replay_recent_examples()

                # Record the detected drift and response.
                self.drift_events.append({
                    "update": self.updates,
                    "error_count": self.errors,
                    "replayed_examples": replayed_count
                })

            # Remember the current verified example for future replay.
            self.recent_examples.append({
                "features": {
                    name: feature_values[name]
                    for name in self.feature_names
                },
                "label": int(verified_label)
            })

            # Save the updated model, detector, and replay buffer.
            self._save_state()

            return {
                "prediction_before_update": prediction_before,
                "verified_label": int(verified_label),
                "prediction_error": error,
                "learning_completed": True,
                "total_updates": self.updates,
                "total_errors": self.errors,
                "adwin_drift_detected": drift_detected,
                "replayed_examples": replayed_count,
                "total_replay_updates": self.replay_updates,
                "total_drift_events": len(self.drift_events)
            }

    # --------------------------------------------------------
    # 9. ADAPTIVE MODEL STATUS
    # --------------------------------------------------------

    def status(self):
        with self.lock:

            # Return statistics without modifying the model.
            return {
                "model": "Passive-Aggressive (29 features)",
                "learning_updates": self.updates,
                "prediction_errors": self.errors,
                "drift_events": len(self.drift_events),
                "recent_drift_events": self.drift_events[-5:],
                "adwin_window_width": self.detector.width,
                "replay_updates": self.replay_updates,
                "recent_verified_examples": len(
                    self.recent_examples
                )
            }


# Create the service instance used by the FastAPI backend.
adaptive_service = AdaptiveService()
