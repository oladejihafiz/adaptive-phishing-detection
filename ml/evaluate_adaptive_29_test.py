
"""
Evaluate the experimental 29-feature adaptive model on the
untouched v3 grouped test dataset.

IMPORTANT:
- Does not train or update the model.
- Does not modify the model or scaler.
- Does not tune the decision threshold.
- Does not use the test data for model selection.
- Saves a separate test evaluation report.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# 1. Configure project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "processed_v3_grouped"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"

FEATURE_FILE = DATA_DIR / "feature_sets.json"

MODEL_FILE = MODEL_DIR / "adaptive_model_29.joblib"
SCALER_FILE = MODEL_DIR / "scaler_29.joblib"

REPORT_FILE = METRICS_DIR / "adaptive_v3_29_test.json"

METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Load the official 29-feature definition
# ---------------------------------------------------------

print("Loading official 29-feature definition...")

with open(FEATURE_FILE, "r", encoding="utf-8") as file:
    feature_sets = json.load(file)

FEATURE_NAMES = feature_sets["29"]

if len(FEATURE_NAMES) != 29 or len(set(FEATURE_NAMES)) != 29:
    raise ValueError("Invalid 29-feature definition.")

print("Feature count:", len(FEATURE_NAMES))


# ---------------------------------------------------------
# 3. Load the saved experimental model and scaler
# ---------------------------------------------------------

print("\nLoading saved 29-feature model and scaler...")

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

if not SCALER_FILE.exists():
    raise FileNotFoundError(f"Scaler not found: {SCALER_FILE}")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# Verify that the saved artifacts match the official features.
if list(scaler.feature_names_in_) != FEATURE_NAMES:
    raise RuntimeError("Scaler feature ordering does not match.")

if model.n_features_in_ != len(FEATURE_NAMES):
    raise RuntimeError("Model feature count does not match.")

if not np.array_equal(model.classes_, np.array([0, 1])):
    raise RuntimeError("Unexpected model classes.")

print("Model and scaler compatibility checks passed.")


# ---------------------------------------------------------
# 4. Load the untouched test dataset
# ---------------------------------------------------------

print("\nLoading untouched v3 grouped test dataset...")

required_columns = FEATURE_NAMES + ["label"]

test = pd.read_csv(
    DATA_DIR / "test.csv",
    usecols=required_columns
)

X_test = test[FEATURE_NAMES].copy()
y_test = test["label"].astype(int)

if set(y_test.unique()) != {0, 1}:
    raise ValueError("Test dataset must contain both classes.")

# Reject missing and infinite feature values.
if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError("Test features contain invalid values.")

print("Test records:", len(X_test))

print("\nTest label distribution:")
print(y_test.value_counts().sort_index())


# ---------------------------------------------------------
# 5. Transform test features
# ---------------------------------------------------------

print("\nScaling test features...")

# Use the scaler fitted on training data.
# Never fit or refit a scaler on the test dataset.
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 6. Generate predictions
# ---------------------------------------------------------

print("\nEvaluating the 29-feature model...")

# Predict without modifying the saved classifier.
predictions = model.predict(X_test_scaled)


# ---------------------------------------------------------
# 7. Calculate test metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = matrix.ravel()

false_positive_rate = (
    fp / (fp + tn) if (fp + tn) else 0.0
)

false_negative_rate = (
    fn / (fn + tp) if (fn + tp) else 0.0
)

class_report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["legitimate", "phishing"],
    output_dict=True,
    zero_division=0
)


# ---------------------------------------------------------
# 8. Display final test results
# ---------------------------------------------------------

print("\n29-FEATURE ADAPTIVE MODEL - TEST RESULTS")
print("========================================")

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
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["legitimate", "phishing"],
        zero_division=0
    )
)


# ---------------------------------------------------------
# 9. Save the independent test evaluation report
# ---------------------------------------------------------

report = {
    "dataset": "processed_v3_grouped",
    "evaluation_split": "test",
    "split_method": "StratifiedGroupKFold",
    "domain_disjoint": True,
    "feature_count": len(FEATURE_NAMES),
    "features": FEATURE_NAMES,
    "classifier": "PassiveAggressiveClassifier",
    "model_file": str(MODEL_FILE.relative_to(BASE_DIR)),
    "scaler_file": str(SCALER_FILE.relative_to(BASE_DIR)),
    "test_records": len(X_test),
    "test_class_distribution": {
        str(label): int(count)
        for label, count in y_test.value_counts().sort_index().items()
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

print("\nTest report saved:", REPORT_FILE)

print("\n29-feature test evaluation completed.")
print("The saved model and scaler were not modified.")
