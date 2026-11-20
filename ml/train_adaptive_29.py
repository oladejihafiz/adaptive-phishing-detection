
"""
Train a separate 29-feature Passive-Aggressive adaptive phishing classifier.

This experiment uses:
- The existing domain-disjoint v3 grouped dataset.
- The official 29-feature definition.
- The same classifier and training parameters as the 15-feature baseline.

IMPORTANT:
- Does not overwrite the existing 15-feature model or scaler.
- Does not modify the adaptive learning state.
- Does not use the v3 test dataset.
- Saves its own model, scaler and validation report.
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

DATA_DIR = BASE_DIR / "dataset" / "processed_v3_grouped"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = DATA_DIR / "feature_sets.json"

# Separate filenames protect the existing 15-feature model.
MODEL_FILE = MODEL_DIR / "adaptive_model_29.joblib"
SCALER_FILE = MODEL_DIR / "scaler_29.joblib"
REPORT_FILE = METRICS_DIR / "adaptive_v3_29_validation.json"

# Existing baseline report is read only, if available.
BASELINE_REPORT = METRICS_DIR / "adaptive_v3_validation.json"


# ---------------------------------------------------------
# 2. Load the official 29-feature definition
# ---------------------------------------------------------

print("Loading 29-feature definition...")

with open(FEATURE_FILE, "r", encoding="utf-8") as file:
    feature_sets = json.load(file)

FEATURE_NAMES = feature_sets["29"]

# Verify the expected number of unique features.
if len(FEATURE_NAMES) != 29 or len(set(FEATURE_NAMES)) != 29:
    raise ValueError("Invalid 29-feature definition.")

print("Feature count:", len(FEATURE_NAMES))
print("Features:", FEATURE_NAMES)


# ---------------------------------------------------------
# 3. Load training and validation data
# ---------------------------------------------------------

print("\nLoading v3 grouped datasets...")

required_columns = FEATURE_NAMES + ["label"]

train = pd.read_csv(
    DATA_DIR / "train.csv",
    usecols=required_columns
)

validation = pd.read_csv(
    DATA_DIR / "validation.csv",
    usecols=required_columns
)

# Keep features in exactly the official defined order.
X_train = train[FEATURE_NAMES].copy()
y_train = train["label"].astype(int)

X_validation = validation[FEATURE_NAMES].copy()
y_validation = validation["label"].astype(int)

# Verify that both datasets contain only binary labels.
for name, labels in [
    ("training", y_train),
    ("validation", y_validation)
]:
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {name} data.")

    if set(labels.unique()) != {0, 1}:
        raise ValueError(f"{name.capitalize()} data must contain both classes.")

# Reject missing and infinite feature values.
if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Training features contain invalid values.")

if not np.isfinite(X_validation.to_numpy()).all():
    raise ValueError("Validation features contain invalid values.")

print("Training records:", len(X_train))
print("Validation records:", len(X_validation))


# ---------------------------------------------------------
# 4. Shuffle training records reproducibly
# ---------------------------------------------------------

print("\nShuffling training records...")

X_train, y_train = shuffle(
    X_train,
    y_train,
    random_state=42
)


# ---------------------------------------------------------
# 5. Scale features
# ---------------------------------------------------------

print("\nFitting 29-feature scaler...")

# Fit exclusively on training data to avoid data leakage.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)


# ---------------------------------------------------------
# 6. Train the Passive-Aggressive classifier
# ---------------------------------------------------------

print("\nTraining 29-feature Passive-Aggressive classifier...")

# Keep the same settings as the existing 15-feature baseline.
model = PassiveAggressiveClassifier(
    C=0.1,
    max_iter=1000,
    random_state=42
)

# Train incrementally using the same batch size.
batch_size = 5000

for start in range(0, len(X_train_scaled), batch_size):

    end = min(start + batch_size, len(X_train_scaled))

    X_batch = X_train_scaled[start:end]
    y_batch = y_train.iloc[start:end]

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
# 7. Evaluate on validation data
# ---------------------------------------------------------

print("\nEvaluating the 29-feature model...")

# Predict without updating the trained model.
predictions = model.predict(X_validation_scaled)

accuracy = accuracy_score(y_validation, predictions)
precision = precision_score(
    y_validation, predictions, zero_division=0
)
recall = recall_score(
    y_validation, predictions, zero_division=0
)
f1 = f1_score(
    y_validation, predictions, zero_division=0
)

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

print("\n29-feature validation results")
print("-----------------------------")
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

print("Compatibility checks passed.")


# ---------------------------------------------------------
# 9. Save the independent validation report
# ---------------------------------------------------------

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
# 10. Save the separate model and scaler
# ---------------------------------------------------------

joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

print("\n29-feature model saved:", MODEL_FILE)
print("29-feature scaler saved:", SCALER_FILE)


# ---------------------------------------------------------
# 11. Display the existing baseline for comparison
# ---------------------------------------------------------

print("\nComparing with the existing 15-feature baseline...")

if BASELINE_REPORT.exists():

    with open(BASELINE_REPORT, "r", encoding="utf-8") as file:
        baseline = json.load(file)

    baseline_metrics = baseline.get("metrics", {})
    new_metrics = report["metrics"]

    print("\nMETRIC              15 FEATURES       29 FEATURES")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "false_negative_rate"
    ]:

        old_value = baseline_metrics.get(metric)

        if old_value is not None:
            print(
                f"{metric:<20} "
                f"{old_value:<17.6f} "
                f"{new_metrics[metric]:.6f}"
            )

        else:
            print(
                f"{metric:<20} "
                f"{'Unavailable':<17} "
                f"{new_metrics[metric]:.6f}"
            )

else:
    print("Existing baseline report was not found.")
    print("The 29-feature results have still been saved.")

print("\n29-feature experiment completed.")
print("The existing 15-feature model was not modified.")
