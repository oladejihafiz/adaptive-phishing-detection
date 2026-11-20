from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from river.drift import ADWIN
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "dataset" / "processed"
MODELS = BASE / "models"
RESULTS = BASE / "results" / "stream"

RESULTS.mkdir(parents=True, exist_ok=True)

# Use validation data for development experiments.
# Do not touch the final test dataset yet.
MAX_SAMPLES = 3000
BATCH_SIZE = 1000

print("Loading validation data...")

df = pd.read_csv(DATA / "validation.csv")

# Random order is used only for a preliminary
# streaming simulation, NOT a chronological drift test.
df = df.sample(
    frac=1,
    random_state=42
).head(MAX_SAMPLES).reset_index(drop=True)

static_model = joblib.load(
    MODELS / "static_model.joblib"
)

# Disable parallel processing for individual URL predictions
static_model.n_jobs = 1

online_model = joblib.load(
    MODELS / "adaptive_model.joblib"
)

scaler = joblib.load(
    MODELS / "scaler.joblib"
)

feature_names = list(
    static_model.feature_names_in_
)

X = df[feature_names]
y = df["label"].to_numpy()

X_scaled = scaler.transform(X)

# Separate baseline and online predictions
static_predictions = []
online_predictions = []

adwin = ADWIN(delta=0.002)

drift_positions = []
online_errors = []
running_accuracy = []

print("Processing simulated URL stream...")

for i in range(len(df)):

    raw_sample = X.iloc[[i]]
    scaled_sample = X_scaled[i:i + 1]
    true_label = int(y[i])

    # 1. Predict BEFORE updating
    static_pred = int(
        static_model.predict(raw_sample)[0]
    )

    online_pred = int(
        online_model.predict(scaled_sample)[0]
    )

    static_predictions.append(static_pred)
    online_predictions.append(online_pred)

    # 2. Record prediction error
    error = int(online_pred != true_label)
    online_errors.append(error)

    # 3. Monitor error with ADWIN
    adwin.update(error)

    if adwin.drift_detected:
        drift_positions.append(i + 1)
        print(
            f"ADWIN change detected at sample {i + 1}"
        )

    # 4. Learn AFTER receiving the true label
    online_model.partial_fit(
        scaled_sample,
        np.array([true_label])
    )

    # 5. Track running performance
    running_accuracy.append(
        1 - np.mean(online_errors)
    )

    if (i + 1) % BATCH_SIZE == 0:
        print(
            f"Processed {i + 1}/{len(df)} URLs"
        )

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "false_positive_rate": (
            fp / (fp + tn) if fp + tn else 0
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }

static_metrics = calculate_metrics(
    y, static_predictions
)

online_metrics = calculate_metrics(
    y, online_predictions
)

print("\nSTATIC RANDOM FOREST")
print(json.dumps(static_metrics, indent=2))

print("\nONLINE PASSIVE-AGGRESSIVE")
print(json.dumps(online_metrics, indent=2))

print("\nADWIN detections:", len(drift_positions))
print("Detection positions:", drift_positions)

# Save results
summary = {
    "samples": len(df),
    "static_random_forest": static_metrics,
    "online_passive_aggressive": online_metrics,
    "adwin_detection_positions": drift_positions,
    "note": (
        "Randomized validation stream; ADWIN detections "
        "do not establish real-world temporal drift."
    )
}

with open(
    RESULTS / "stream_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

pd.DataFrame({
    "sample": np.arange(1, len(df) + 1),
    "actual_label": y,
    "static_prediction": static_predictions,
    "online_prediction": online_predictions,
    "online_error": online_errors,
    "running_accuracy": running_accuracy
}).to_csv(
    RESULTS / "stream_predictions.csv",
    index=False
)

# Plot running accuracy
plt.figure(figsize=(10, 5))

plt.plot(
    np.arange(1, len(df) + 1),
    running_accuracy,
    label="Online model running accuracy"
)

for position in drift_positions:
    plt.axvline(
        position,
        color="red",
        alpha=0.5,
        linestyle="--"
    )

plt.xlabel("Number of processed URLs")
plt.ylabel("Cumulative accuracy")
plt.title("Online Learning and ADWIN Detections")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS / "online_accuracy.png",
    dpi=200
)

plt.close()

print("\nStream evaluation complete!")
print("Results saved in:", RESULTS)