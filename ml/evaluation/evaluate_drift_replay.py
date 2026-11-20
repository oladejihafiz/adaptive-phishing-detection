
# Compare ordinary online learning with ADWIN-triggered replay.
# This experiment does not modify the saved API learning state.

from collections import deque
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from river.drift import ADWIN
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Locate the project files.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "processed" / "validation.csv"
MODEL_FILE = BASE_DIR / "models" / "adaptive_model.joblib"
SCALER_FILE = BASE_DIR / "models" / "scaler.joblib"

RESULT_DIR = BASE_DIR / "results" / "metrics"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Use a reproducible experiment.
# Change this value between runs to test different samples
# and ordering while keeping the experiment repeatable.
# sys lets us receive a random seed from the terminal.
import sys

# Use the supplied seed, or default to 42.
RANDOM_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

# Create two phases with different phishing proportions.
# These are artificial changes in class prevalence, not real
# chronological phishing observations.
PHASE_SIZE = 3000
PHASE_1_PHISHING_RATE = 0.05
PHASE_2_PHISHING_RATE = 0.40

BUFFER_SIZE = 128
REPLAY_SIZE = 32
MIN_REPLAY_EXAMPLES = 16


def sample_phase(data, phishing_rate, random_seed):
    """Sample a phase with a specified phishing proportion."""

    phishing_count = round(PHASE_SIZE * phishing_rate)
    legitimate_count = PHASE_SIZE - phishing_count

    # Sample both classes without replacement.
    phishing = data[data["label"] == 1].sample(
        n=phishing_count,
        random_state=random_seed,
    )

    legitimate = data[data["label"] == 0].sample(
        n=legitimate_count,
        random_state=random_seed + 1,
    )

    # Shuffle within the phase.
    return pd.concat(
        [phishing, legitimate],
        ignore_index=True,
    ).sample(
        frac=1,
        random_state=random_seed + 2,
    ).reset_index(drop=True)


def calculate_metrics(actual, predicted):
    """Calculate classification metrics from recorded predictions."""

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        ),
        "false_positive_rate": (
            fp / (fp + tn) if fp + tn else 0.0
        ),
    }


# Load the validation dataset.
data = pd.read_csv(DATA_FILE)

# Load the scaler fitted during initial training.
scaler = joblib.load(SCALER_FILE)

# Preserve the exact feature order used by the model.
feature_names = list(scaler.feature_names_in_)

# Construct two phases with an artificial prevalence shift.
phase_1 = sample_phase(
    data,
    PHASE_1_PHISHING_RATE,
    RANDOM_SEED,
)

phase_2 = sample_phase(
    data,
    PHASE_2_PHISHING_RATE,
    RANDOM_SEED + 100,
)

# Join both phases into one ordered evaluation stream.
stream = pd.concat(
    [phase_1, phase_2],
    ignore_index=True,
)

# Transform features once, using the original fitted scaler.
X = scaler.transform(stream[feature_names])
y = stream["label"].to_numpy(dtype=int)

# Load two independent copies of the SAME pretrained model.
baseline_model = joblib.load(MODEL_FILE)
replay_model = joblib.load(MODEL_FILE)

# Monitor only the replay model's pre-update prediction errors.
detector = ADWIN(delta=0.002)

# Store recent scaled feature vectors and their verified labels.
recent_examples = deque(maxlen=BUFFER_SIZE)

baseline_predictions = []
replay_predictions = []

drift_events = []
total_replay_updates = 0

# Measure elapsed time for the full paired experiment.
start_time = perf_counter()

for index in range(len(y)):
    # Get one example and its correct label.
    example = X[index:index + 1]
    label = int(y[index])

    # Predict BEFORE either model learns this example.
    baseline_prediction = int(
        baseline_model.predict(example)[0]
    )

    replay_prediction = int(
        replay_model.predict(example)[0]
    )

    baseline_predictions.append(baseline_prediction)
    replay_predictions.append(replay_prediction)

    # Calculate the replay model's pre-update prediction error.
    replay_error = int(replay_prediction != label)

    # Update the drift detector.
    detector.update(replay_error)
    drift_detected = bool(detector.drift_detected)

    # Both models receive exactly one normal learning update.
    baseline_model.partial_fit(
        example,
        np.array([label]),
        classes=np.array([0, 1]),
    )

    replay_model.partial_fit(
        example,
        np.array([label]),
        classes=np.array([0, 1]),
    )

    # Add this newly verified example to the replay buffer.
    recent_examples.append(
        (example[0].copy(), label)
    )

    # Only the replay strategy performs additional training.
    if drift_detected:
        replayed_count = 0

        if len(recent_examples) >= MIN_REPLAY_EXAMPLES:
            selected = list(recent_examples)[-REPLAY_SIZE:]

            replay_X = np.vstack([
                features for features, _ in selected
            ])

            replay_y = np.array([
                verified_label for _, verified_label in selected
            ])

            replay_model.partial_fit(
                replay_X,
                replay_y,
                classes=np.array([0, 1]),
            )

            replayed_count = len(selected)
            total_replay_updates += replayed_count

        drift_events.append({
            "sample": index + 1,
            "phase": (
                "Before shift"
                if index < PHASE_SIZE
                else "After shift"
            ),
            "replayed_examples": replayed_count,
        })

elapsed_seconds = perf_counter() - start_time

# Calculate overall and phase-specific results.
results = []

for name, predictions in [
    ("Baseline online learning", baseline_predictions),
    ("ADWIN-triggered replay", replay_predictions),
]:
    for phase_name, start, end in [
        ("Overall", 0, len(y)),
        ("Before shift", 0, PHASE_SIZE),
        ("After shift", PHASE_SIZE, len(y)),
    ]:
        metrics = calculate_metrics(
            y[start:end],
            predictions[start:end],
        )

        results.append({
            "strategy": name,
            "phase": phase_name,
            **metrics,
        })

results_df = pd.DataFrame(results)

# Save the comparison for your project report.
# Give each experiment its own filename to avoid overwriting results.
output_file = RESULT_DIR / f"drift_replay_comparison_seed_{RANDOM_SEED}.csv"
results_df.to_csv(output_file, index=False)

# Save the detected drift events separately.
# Save drift events separately for each random seed.
drift_file = RESULT_DIR / f"drift_replay_events_seed_{RANDOM_SEED}.csv"
pd.DataFrame(
    drift_events,
    columns=["sample", "phase", "replayed_examples"],
).to_csv(drift_file, index=False)

# Display the main results.
print("\nCOMPARATIVE EVALUATION")
print(results_df.round(4).to_string(index=False))

print("\nDRIFT SUMMARY")
print("Total stream samples:", len(y))
print("Drift events:", len(drift_events))
print("Replay training examples:", total_replay_updates)
print("Drift details:", drift_events)
print("Elapsed seconds:", round(elapsed_seconds, 2))

print("\nResults saved to:")
print(output_file)
print(drift_file)
