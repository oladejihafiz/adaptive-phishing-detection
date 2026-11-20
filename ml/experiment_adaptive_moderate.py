
"""
Test moderate class weighting for adaptive phishing detection.

This experiment uses SGDClassifier with Passive-Aggressive-style
learning behaviour and a modest phishing sample weight.

It saves separate experimental files and does not overwrite
the existing adaptive model, scaler or learning state.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.utils import shuffle


# ---------------------------------------------------------
# 1. Configure paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "processed_v3_grouped"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "adaptive_model_moderate_experiment.joblib"
SCALER_FILE = MODEL_DIR / "adaptive_scaler_moderate_experiment.joblib"
REPORT_FILE = METRICS_DIR / "adaptive_moderate_experiment.json"


# ---------------------------------------------------------
# 2. Load the feature definition and datasets
# ---------------------------------------------------------

with open(DATA_DIR / "feature_sets.json", "r", encoding="utf-8") as file:
    feature_sets = json.load(file)

feature_names = feature_sets["15"]

required_columns = feature_names + ["label"]

train = pd.read_csv(DATA_DIR / "train.csv", usecols=required_columns)
validation = pd.read_csv(DATA_DIR / "validation.csv", usecols=required_columns)

X_train = train[feature_names].copy()
y_train = train["label"].astype(int)

X_validation = validation[feature_names].copy()
y_validation = validation["label"].astype(int)

if set(y_train.unique()) != {0, 1}:
    raise ValueError("Training data must contain both classes.")

if set(y_validation.unique()) != {0, 1}:
    raise ValueError("Validation data must contain both classes.")

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Training data contains invalid feature values.")

if not np.isfinite(X_validation.to_numpy()).all():
    raise ValueError("Validation data contains invalid feature values.")


# ---------------------------------------------------------
# 3. Shuffle and scale the data
# ---------------------------------------------------------

X_train, y_train = shuffle(
    X_train,
    y_train,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)


# ---------------------------------------------------------
# 4. Configure moderate sample weighting
# ---------------------------------------------------------

# Legitimate examples retain their normal weight.
# Phishing examples receive a modest 25% increase.
sample_weights = np.where(y_train.to_numpy() == 1, 1.25, 1.0)

print("Sample weights:")
print("Legitimate: 1.00")
print("Phishing:   1.25")


# ---------------------------------------------------------
# 5. Train the experimental classifier
# ---------------------------------------------------------

model = SGDClassifier(
    loss="hinge",
    penalty=None,
    learning_rate="pa1",
    eta0=1.0,
    random_state=42
)

batch_size = 5000

print("\nTraining moderate-weight classifier...")

for start in range(0, len(X_train_scaled), batch_size):

    end = min(start + batch_size, len(X_train_scaled))

    # Select the current training batch.
    X_batch = X_train_scaled[start:end]
    y_batch = y_train.iloc[start:end]

    # Apply the corresponding sample weights.
    batch_weights = sample_weights[start:end]

    model.partial_fit(
        X_batch,
        y_batch,
        classes=np.array([0, 1]),
        sample_weight=batch_weights
    )

    print(
        f"Processed {end:,}/{len(X_train_scaled):,} records",
        end="\r"
    )

print("\nTraining completed.")


# ---------------------------------------------------------
# 6. Evaluate the default decision boundary
# ---------------------------------------------------------

scores = model.decision_function(X_validation_scaled)

predictions = (scores >= 0).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_validation,
    predictions,
    labels=[0, 1]
).ravel()

metrics = {
    "accuracy": float(accuracy_score(y_validation, predictions)),
    "precision": float(precision_score(
        y_validation, predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_validation, predictions, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_validation, predictions, zero_division=0
    )),
    "false_positive_rate": float(fp / (fp + tn)),
    "false_negative_rate": float(fn / (fn + tp))
}

print("\nModerate-weight results:")

for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

print("\nConfusion matrix:")
print([[int(tn), int(fp)], [int(fn), int(tp)]])


# ---------------------------------------------------------
# 7. Evaluate alternative decision thresholds
# ---------------------------------------------------------

thresholds = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

threshold_results = []

for threshold in thresholds:

    threshold_predictions = (scores >= threshold).astype(int)

    t_n, f_p, f_n, t_p = confusion_matrix(
        y_validation,
        threshold_predictions,
        labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "threshold": threshold,
        "precision": float(precision_score(
            y_validation, threshold_predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_validation, threshold_predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_validation, threshold_predictions, zero_division=0
        )),
        "false_positive_rate": float(f_p / (f_p + t_n))
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nThreshold comparison:")
print(threshold_df.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))


# ---------------------------------------------------------
# 8. Save the experimental artifacts
# ---------------------------------------------------------

joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

report = {
    "experiment": "moderate_phishing_sample_weight",
    "dataset": "processed_v3_grouped",
    "features": feature_names,
    "phishing_sample_weight": 1.25,
    "legitimate_sample_weight": 1.0,
    "classifier": "SGDClassifier",
    "learning_rate": "pa1",
    "metrics": metrics,
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    },
    "threshold_results": threshold_results
}

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nExperimental model saved:", MODEL_FILE)
print("Experimental scaler saved:", SCALER_FILE)
print("Experimental report saved:", REPORT_FILE)

