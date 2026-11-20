
"""
Train the initial Passive-Aggressive adaptive phishing classifier.

This version uses the domain-disjoint v3 grouped dataset and
the official 15-feature adaptive feature definition.

The trained model and scaler retain the filenames expected by
backend/adaptive_service.py.

The v3 test dataset is deliberately not used in this script.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils import shuffle


# ---------------------------------------------------------
# 1. Configure project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Use the domain-disjoint v3 grouped dataset.
DATA_DIR = BASE_DIR / "dataset" / "processed_v3_grouped"

MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = DATA_DIR / "feature_sets.json"

MODEL_FILE = MODEL_DIR / "adaptive_model.joblib"
SCALER_FILE = MODEL_DIR / "scaler.joblib"

REPORT_FILE = METRICS_DIR / "adaptive_v3_validation.json"


# ---------------------------------------------------------
# 2. Load the official adaptive feature definition
# ---------------------------------------------------------

print("Loading adaptive feature definition...")

with open(FEATURE_FILE, "r", encoding="utf-8") as file:
    feature_sets = json.load(file)

FEATURE_NAMES = feature_sets["15"]

# Ensure that exactly 15 unique features are defined.
if len(FEATURE_NAMES) != 15 or len(set(FEATURE_NAMES)) != 15:
    raise ValueError("Invalid 15-feature definition.")

print("Adaptive feature count:", len(FEATURE_NAMES))
print("Features:", FEATURE_NAMES)


# ---------------------------------------------------------
# 3. Load only the required CSV columns
# ---------------------------------------------------------

print("\nLoading v3 grouped training and validation datasets...")

# Loading only the required columns reduces memory usage.
required_columns = FEATURE_NAMES + ["label"]

train = pd.read_csv(
    DATA_DIR / "train.csv",
    usecols=required_columns
)

validation = pd.read_csv(
    DATA_DIR / "validation.csv",
    usecols=required_columns
)

# Explicitly arrange columns in the official feature order.
X_train = train[FEATURE_NAMES].copy()
y_train = train["label"].astype(int)

X_validation = validation[FEATURE_NAMES].copy()
y_validation = validation["label"].astype(int)

# Check the binary classification labels.
if not set(y_train.unique()).issubset({0, 1}):
    raise ValueError("Unexpected labels in training data.")

if not set(y_validation.unique()).issubset({0, 1}):
    raise ValueError("Unexpected labels in validation data.")

if set(y_train.unique()) != {0, 1}:
    raise ValueError("Training data must contain both classes.")

if set(y_validation.unique()) != {0, 1}:
    raise ValueError("Validation data must contain both classes.")

# Reject missing or infinite feature values.
if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Training features contain invalid values.")

if not np.isfinite(X_validation.to_numpy()).all():
    raise ValueError("Validation features contain invalid values.")

print("Training records:", len(X_train))
print("Validation records:", len(X_validation))

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation label distribution:")
print(y_validation.value_counts().sort_index())


# ---------------------------------------------------------
# 4. Shuffle training records reproducibly
# ---------------------------------------------------------

print("\nShuffling training records...")

# Randomising record order reduces dependence on the
# original ordering when incremental batches are processed.
X_train, y_train = shuffle(
    X_train,
    y_train,
    random_state=42
)


# ---------------------------------------------------------
# 5. Scale the features
# ---------------------------------------------------------

print("\nFitting feature scaler...")

# Fit the scaler exclusively on the training dataset.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Apply the same fitted scaler to validation data.
X_validation_scaled = scaler.transform(X_validation)


# ---------------------------------------------------------
# 6. Train the adaptive classifier
# ---------------------------------------------------------

print("\nTraining Passive-Aggressive classifier...")

model = PassiveAggressiveClassifier(
    C=0.1,
    max_iter=1000,
    random_state=42
)

# Process the training dataset incrementally in batches.
batch_size = 5000

for start in range(0, len(X_train_scaled), batch_size):

    end = min(start + batch_size, len(X_train_scaled))

    X_batch = X_train_scaled[start:end]
    y_batch = y_train.iloc[start:end]

    # Supply both possible classes during incremental training.
    model.partial_fit(
        X_batch,
        y_batch,
        classes=np.array([0, 1])
    )

    print(
        f"Processed {end:,}/{len(X_train_scaled):,} records",
        end="\r"
    )

print("\nTraining completed.")


# ---------------------------------------------------------
# 7. Evaluate using validation data
# ---------------------------------------------------------

print("\nEvaluating the trained adaptive classifier...")

# Generate predictions without updating the classifier.
predictions = model.predict(X_validation_scaled)

accuracy = accuracy_score(y_validation, predictions)

precision = precision_score(
    y_validation,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_validation,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_validation,
    predictions,
    zero_division=0
)

# Use an explicit label order:
# 0 = legitimate, 1 = phishing.
matrix = confusion_matrix(
    y_validation,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = matrix.ravel()

false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

class_report = classification_report(
    y_validation,
    predictions,
    labels=[0, 1],
    target_names=["legitimate", "phishing"],
    output_dict=True,
    zero_division=0
)

print("\nValidation results")
print("------------------")
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"FPR:       {false_positive_rate:.6f}")
print(f"FNR:       {false_negative_rate:.6f}")

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")
print(
    classification_report(
        y_validation,
        predictions,
        labels=[0, 1],
        target_names=["legitimate", "phishing"],
        zero_division=0
    )
)


# ---------------------------------------------------------
# 8. Verify model and scaler compatibility
# ---------------------------------------------------------

print("\nChecking model compatibility...")

if list(scaler.feature_names_in_) != FEATURE_NAMES:
    raise RuntimeError("Scaler feature ordering is incorrect.")

if model.n_features_in_ != len(FEATURE_NAMES):
    raise RuntimeError("Model feature count is incorrect.")

if not np.array_equal(model.classes_, np.array([0, 1])):
    raise RuntimeError("Unexpected classifier classes.")

print("Model and scaler compatibility checks passed.")


# ---------------------------------------------------------
# 9. Save validation results separately
# ---------------------------------------------------------

# Record enough information to reproduce and explain
# the adaptive model's validation evaluation.
report = {
    "dataset": "processed_v3_grouped",
    "split_method": "StratifiedGroupKFold",
    "domain_disjoint": True,
    "feature_count": len(FEATURE_NAMES),
    "features": FEATURE_NAMES,
    "classifier": "PassiveAggressiveClassifier",
    "parameters": {
        "C": 0.1,
        "max_iter": 1000,
        "random_state": 42,
        "batch_size": batch_size,
        "training_passes": 1
    },
    "training_records": len(X_train),
    "validation_records": len(X_validation),
    "training_class_distribution": {
        str(label): int(count)
        for label, count in train["label"].value_counts().sort_index().items()
    },
    "validation_class_distribution": {
        str(label): int(count)
        for label, count in validation["label"].value_counts().sort_index().items()
    },
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate)
    },
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    },
    "classification_report": class_report
}

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nValidation report saved:", REPORT_FILE)


# ---------------------------------------------------------
# 10. Save the trained model and scaler
# ---------------------------------------------------------

joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

print("\nAdaptive model saved successfully!")
print("Model:", MODEL_FILE)

print("\nScaler saved successfully!")
print("Scaler:", SCALER_FILE)

print("\nV3 adaptive training completed successfully.")
