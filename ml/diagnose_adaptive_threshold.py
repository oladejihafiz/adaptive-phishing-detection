
"""
Diagnose the Passive-Aggressive classifier's decision threshold.

This script does not retrain or modify the saved model.
It evaluates alternative decision thresholds on the v3
validation dataset only.

The test dataset remains untouched.
"""

from pathlib import Path

import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ---------------------------------------------------------
# 1. Configure project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "processed_v3_grouped"
MODEL_DIR = BASE_DIR / "models"

FEATURE_FILE = DATA_DIR / "feature_sets.json"

MODEL_FILE = MODEL_DIR / "adaptive_model.joblib"
SCALER_FILE = MODEL_DIR / "scaler.joblib"


# ---------------------------------------------------------
# 2. Load the saved model, scaler and validation data
# ---------------------------------------------------------

print("Loading adaptive model and scaler...")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# Read the feature configuration as a normal JSON dictionary.
with open(FEATURE_FILE, "r", encoding="utf-8") as file:
    feature_sets = json.load(file)

# Retrieve the exact 15 features used by the adaptive model.
feature_names = feature_sets["15"]

validation = pd.read_csv(
    DATA_DIR / "validation.csv",
    usecols=feature_names + ["label"]
)

X_validation = validation[feature_names]
y_validation = validation["label"].astype(int)

X_validation_scaled = scaler.transform(X_validation)


# ---------------------------------------------------------
# 3. Obtain decision scores
# ---------------------------------------------------------

# Decision scores show how far each example is from
# the classifier's decision boundary.
scores = model.decision_function(X_validation_scaled)

print("\nDecision score distribution:")
print(pd.Series(scores).describe())


# ---------------------------------------------------------
# 4. Evaluate alternative thresholds
# ---------------------------------------------------------

# The default Passive-Aggressive prediction boundary is 0.
# Positive scores above a selected threshold are classified
# as phishing.

thresholds = [
    -1.00,
    -0.75,
    -0.50,
    -0.25,
     0.00,
     0.25,
     0.50,
     0.75,
     1.00,
     1.50,
     2.00
]

results = []

for threshold in thresholds:

    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_validation, predictions),
        "precision": precision_score(
            y_validation, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_validation, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_validation, predictions, zero_division=0
        ),
        "false_positive_rate": fp / (fp + tn),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp
    })


# ---------------------------------------------------------
# 5. Display the threshold comparison
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\nThreshold comparison:")
print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ---------------------------------------------------------
# 6. Save diagnostic results
# ---------------------------------------------------------

REPORT_DIR = BASE_DIR / "results" / "metrics"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "adaptive_threshold_diagnostic.csv"

results_df.to_csv(REPORT_FILE, index=False)

print("\nDiagnostic report saved:")
print(REPORT_FILE)
