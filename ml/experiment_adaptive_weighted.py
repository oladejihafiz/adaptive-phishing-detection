
"""
Experiment with class-weighted Passive-Aggressive training.

This is an isolated experiment. It does not overwrite the
current adaptive model or scaler.

The validation dataset is used for evaluation only.
The v3 test dataset remains untouched.
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

# Separate experimental files prevent overwriting the
# existing adaptive model and scaler.
MODEL_FILE = MODEL_DIR / "adaptive_model_weighted_experiment.joblib"
SCALER_FILE = MODEL_DIR / "adaptive_scaler_weighted_experiment.joblib"

REPORT_FILE = METRICS_DIR / "adaptive_weighted_experiment.json"


# ---------------------------------------------------------
# 2. Load feature definition and datasets
# ---------------------------------------------------------

with open(DATA_DIR / "feature_sets.json", "r", encoding="utf-8") as file:
    feature_sets = json.load(file)

feature_names = feature_sets["15"]

required_columns = feature_names + ["label"]

print("Loading v3 datasets...")

train = pd.read_csv(
    DATA_DIR / "train.csv",
    usecols=required_columns
)

validation = pd.read_csv(
    DATA_DIR / "validation.csv",
    usecols=required_columns
)

X_train = train[feature_names].copy()
y_train = train["label"].astype(int)

X_validation = validation[feature_names].copy()
y_validation = validation["label"].astype(int)

# Check that the data contains both expected classes.
if set(y_train.unique()) != {0, 1}:
    raise ValueError("Training data must contain both classes.")

if set(y_validation.unique()) != {0, 1}:
    raise ValueError("Validation data must contain both classes.")

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Training features contain invalid values.")

if not np.isfinite(X_validation.to_numpy()).all():
    raise ValueError("Validation features contain invalid values.")


# ---------------------------------------------------------
# 3. Calculate class weights
# ---------------------------------------------------------

# Balanced class weights give each class equal total
# influence, regardless of its number of examples.
class_counts = y_train.value_counts().sort_index()

total_samples = len(y_train)
number_of_classes = len(class_counts)

class_weights = {
    int(label): total_samples / (number_of_classes * int(count))
    for label, count in class_counts.items()
}

print("\nClass weights:")
print(class_weights)


# ---------------------------------------------------------
# 4. Shuffle and scale the training data
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
# 5. Train a weighted Passive-Aggressive classifier
# ---------------------------------------------------------

print("\nTraining weighted classifier...")

# Use SGDClassifier with Passive-Aggressive learning behaviour.
# This is the supported replacement for the deprecated classifier.
model = SGDClassifier(
    loss="hinge",
    penalty=None,
    learning_rate="pa1",
    eta0=1.0,
    random_state=42
)

batch_size = 5000

for start in range(0, len(X_train_scaled), batch_size):

    end = min(start + batch_size, len(X_train_scaled))

    X_batch = X_train_scaled[start:end]
    y_batch = y_train.iloc[start:end]

    # Apply the global class weight to each training example.
    sample_weights = np.array([
        class_weights[int(label)]
        for label in y_batch
    ])

    model.partial_fit(
        X_batch,
        y_batch,
        classes=np.array([0, 1]),
        sample_weight=sample_weights
    )

    print(
        f"Processed {end:,}/{len(X_train_scaled):,} records",
        end="\r"
    )

print("\nWeighted training completed.")


# ---------------------------------------------------------
# 6. Evaluate at the default decision boundary
# ---------------------------------------------------------

print("\nEvaluating weighted classifier...")

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

print("\nWeighted model results:")
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

print("\nConfusion matrix:")
print([[tn, fp], [fn, tp]])


# ---------------------------------------------------------
# 7. Examine alternative thresholds
# ---------------------------------------------------------

print("\nThreshold comparison:")

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
        "precision": precision_score(
            y_validation, threshold_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_validation, threshold_predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_validation, threshold_predictions, zero_division=0
        ),
        "false_positive_rate": f_p / (f_p + t_n)
    })

threshold_df = pd.DataFrame(threshold_results)

print(threshold_df.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))


# ---------------------------------------------------------
# 8. Save experimental model and results separately
# ---------------------------------------------------------

joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

report = {
    "experiment": "class_weighted_passive_aggressive",
    "dataset": "processed_v3_grouped",
    "feature_count": len(feature_names),
    "features": feature_names,
    "class_weights": class_weights,
    "parameters": {
        "C": 0.1,
        "max_iter": 1000,
        "random_state": 42,
        "batch_size": batch_size
    },
    "metrics_at_default_threshold": metrics,
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

print("\nExperimental model saved:")
print(MODEL_FILE)

print("\nExperimental scaler saved:")
print(SCALER_FILE)

print("\nExperimental report saved:")
print(REPORT_FILE)
