
"""
Train a separate 40-feature Passive-Aggressive adaptive phishing classifier.

This experiment uses:
- The domain-disjoint v3 grouped dataset.
- The official 40-feature definition.
- The same classifier and training parameters as the 15-feature
  and 29-feature experiments.

IMPORTANT:
- Does not overwrite the existing 15-feature or 29-feature models.
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

# Use the existing domain-disjoint v3 grouped dataset.
DATA_DIR = BASE_DIR / "dataset" / "processed_v3_grouped"

MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = DATA_DIR / "feature_sets.json"

# Separate filenames protect the existing models.
MODEL_FILE = MODEL_DIR / "adaptive_model_40.joblib"
SCALER_FILE = MODEL_DIR / "scaler_40.joblib"

REPORT_FILE = METRICS_DIR / "adaptive_v3_40_validation.json"

# Existing experiment reports are read only.
BASELINE_REPORTS = {
    "15 features": METRICS_DIR / "adaptive_v3_validation.json",
    "29 features": METRICS_DIR / "adaptive_v3_29_validation.json"
}


# ---------------------------------------------------------
# 2. Load the official 40-feature definition
# ---------------------------------------------------------

print("Loading 40-feature definition...")

with open(FEATURE_FILE, "r", encoding="utf-8") as file:
    feature_sets = json.load(file)

FEATURE_NAMES = feature_sets["40"]

# Verify that exactly 40 unique features are defined.
if len(FEATURE_NAMES) != 40 or len(set(FEATURE_NAMES)) != 40:
    raise ValueError("Invalid 40-feature definition.")

print("Feature count:", len(FEATURE_NAMES))
print("Features:", FEATURE_NAMES)


# ---------------------------------------------------------
# 3. Load training and validation datasets
# ---------------------------------------------------------

print("\nLoading v3 grouped datasets...")

# Load only the features required by this experiment.
required_columns = FEATURE_NAMES + ["label"]

train = pd.read_csv(
    DATA_DIR / "train.csv",
    usecols=required_columns
)

validation = pd.read_csv(
    DATA_DIR / "validation.csv",
    usecols=required_columns
)

# Arrange features in exactly the official feature order.
X_train = train[FEATURE_NAMES].copy()
y_train = train["label"].astype(int)

X_validation = validation[FEATURE_NAMES].copy()
y_validation = validation["label"].astype(int)

# Confirm that both datasets contain only binary labels.
for name, labels in [
    ("training", y_train),
    ("validation", y_validation)
]:
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {name} data.")

    if set(labels.unique()) != {0, 1}:
        raise ValueError(
            f"{name.capitalize()} data must contain both classes."
        )

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

# Use the same random seed as the previous experiments.
X_train, y_train = shuffle(
    X_train,
    y_train,
    random_state=42
)


# ---------------------------------------------------------
# 5. Scale the features
# ---------------------------------------------------------

print("\nFitting 40-feature scaler...")

# Fit the scaler exclusively on training data.
# This prevents information leakage from validation data.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Transform validation data using the training scaler.
X_validation_scaled = scaler.transform(X_validation)


# ---------------------------------------------------------
# 6. Train the Passive-Aggressive classifier
# ---------------------------------------------------------

print("\nTraining 40-feature Passive-Aggressive classifier...")

# Preserve the same parameters used in the previous experiments.
model = PassiveAggressiveClassifier(
    C=0.1,
    max_iter=1000,
    random_state=42
)

# Process training records incrementally in batches.
batch_size = 5000

for start in range(0, len(X_train_scaled), batch_size):

    end = min(start + batch_size, len(X_train_scaled))

    X_batch = X_train_scaled[start:end]
    y_batch = y_train.iloc[start:end]

    # partial_fit supports incremental model training.
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

print("\nEvaluating the 40-feature model...")

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

# Explicitly define:
# 0 = legitimate
# 1 = phishing
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

print("\n40-feature validation results")
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

print("Model and scaler compatibility checks passed.")


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

print("\n40-feature model saved:", MODEL_FILE)
print("40-feature scaler saved:", SCALER_FILE)


# ---------------------------------------------------------
# 11. Compare all available feature experiments
# ---------------------------------------------------------

print("\nComparing adaptive model experiments...")

all_reports = {
    "40 features": report
}

# Load earlier reports without modifying them.
for experiment_name, report_path in BASELINE_REPORTS.items():

    if report_path.exists():

        with open(report_path, "r", encoding="utf-8") as file:
            all_reports[experiment_name] = json.load(file)

    else:
        print(f"Warning: {experiment_name} report not found.")

# Display all metrics side by side.
metric_names = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "false_negative_rate"
]

print("\nMETRIC              15 FEATURES   29 FEATURES   40 FEATURES")

for metric in metric_names:

    values = []

    for experiment_name in [
        "15 features",
        "29 features",
        "40 features"
    ]:

        experiment = all_reports.get(experiment_name, {})
        metric_value = experiment.get("metrics", {}).get(metric)

        values.append(
            f"{metric_value:.6f}" if metric_value is not None
            else "N/A"
        )

    print(
        f"{metric:<20} "
        f"{values[0]:<14} "
        f"{values[1]:<14} "
        f"{values[2]}"
    )

print("\n40-feature experiment completed.")
print("Existing models and adaptive learning state were not modified.")
