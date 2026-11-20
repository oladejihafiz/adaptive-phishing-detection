
"""
THRESHOLD SENSITIVITY EXPERIMENT - VERSION 3

Purpose:
    Compare nearby decision thresholds for the selected
    40-feature Random Forest using validation predictions only.

Important:
    - Does NOT retrain the model.
    - Does NOT use the independent test set.
    - Does NOT overwrite existing experiment results.
    - Keeps the original threshold of 0.747170 as a reference.
    - Positive class: 1 = phishing.
    - Negative class: 0 = legitimate.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics_v3_grouped"
    / "validation_probabilities_v3.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics_v3_grouped"
)

OUTPUT_FILE = OUTPUT_DIR / "threshold_sensitivity_v3.csv"

# Original threshold selected using validation data.
ORIGINAL_THRESHOLD = 0.747170

# Test thresholds around the original operating point.
THRESHOLDS = [
    0.70,
    0.71,
    0.72,
    0.73,
    0.74,
    ORIGINAL_THRESHOLD,
    0.75,
    0.76,
    0.77,
    0.78,
    0.79,
    0.80,
]


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def find_label_column(df):
    """Identify the actual phishing/legitimate label column."""

    candidates = [
        "label",
        "y_true",
        "true_label",
        "actual_label",
        "target",
    ]

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    raise KeyError(
        "Could not identify the validation label column.\n"
        f"Available columns: {list(df.columns)}"
    )


def find_probability_column(df):
    """
    Identify the 40-feature Random Forest probability column.

    Prefer a column explicitly identifying the 40-feature model.
    If the CSV contains only one probability column, use it.
    """

    candidates = [
        "phishing_probability_40",
        "probability_40",
        "probability_40_features",
        "phishing_score_40",
        "score_40",
        "prob_40",
        "y_score_40",
        "rf_40_probability",
        "40_feature_probability",
        "phishing_probability",
        "probability",
        "y_score",
    ]

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    # Try to identify a probability column containing "40".
    possible = [
        column
        for column in df.columns
        if "40" in str(column).lower()
        and any(
            word in str(column).lower()
            for word in ["prob", "score"]
        )
    ]

    if len(possible) == 1:
        return possible[0]

    # If only one probability/score column exists, use it.
    all_probability_columns = [
        column
        for column in df.columns
        if any(
            word in str(column).lower()
            for word in ["probability", "prob", "score"]
        )
    ]

    if len(all_probability_columns) == 1:
        return all_probability_columns[0]

    raise KeyError(
        "Could not uniquely identify the 40-feature model's "
        "phishing probability column.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Possible probability columns: {all_probability_columns}"
    )


# ---------------------------------------------------------
# 3. LOAD VALIDATION PREDICTIONS
# ---------------------------------------------------------

print("=" * 72)
print("THRESHOLD SENSITIVITY EXPERIMENT")
print("=" * 72)

print("\nLoading validation predictions...")

df = pd.read_csv(INPUT_FILE)

print("Validation records:", f"{len(df):,}")

label_column = find_label_column(df)
probability_column = find_probability_column(df)

print("Label column:", label_column)
print("Probability column:", probability_column)

# Convert the actual labels and probability scores.
y_true = pd.to_numeric(
    df[label_column],
    errors="coerce"
)

scores = pd.to_numeric(
    df[probability_column],
    errors="coerce"
)

# Reject missing or invalid values rather than silently dropping rows.
if y_true.isna().any() or scores.isna().any():
    raise ValueError(
        "Missing or invalid labels/probabilities found."
    )

if not y_true.isin([0, 1]).all():
    raise ValueError(
        "Labels must follow the project convention: "
        "0 = legitimate, 1 = phishing."
    )

if not scores.between(0, 1).all():
    raise ValueError(
        "Probability scores must be between 0 and 1."
    )

y_true = y_true.astype(int).to_numpy()
scores = scores.to_numpy()


# ---------------------------------------------------------
# 4. CALCULATE METRICS FOR EACH THRESHOLD
# ---------------------------------------------------------

results = []

for threshold in THRESHOLDS:

    # Classify URLs using the current threshold.
    y_pred = (scores >= threshold).astype(int)

    # Confusion matrix:
    # TN = legitimate correctly classified.
    # FP = legitimate incorrectly classified as phishing.
    # FN = phishing incorrectly classified as legitimate.
    # TP = phishing correctly classified.
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Calculate the main evaluation metrics.
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # Calculate false positive and false negative rates.
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "is_original_threshold": (
            threshold == ORIGINAL_THRESHOLD
        ),
    })


# ---------------------------------------------------------
# 5. SAVE AND DISPLAY RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

# Sort by threshold for easier comparison.
results_df = results_df.sort_values(
    "threshold"
).reset_index(drop=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# Display percentages for readability.
percentage_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "false_negative_rate",
]

display_df = results_df.copy()

for column in percentage_columns:
    display_df[column] = (
        display_df[column] * 100
    ).round(3)

display_df["threshold"] = (
    display_df["threshold"].round(6)
)

print("\n" + "=" * 72)
print("VALIDATION THRESHOLD COMPARISON")
print("=" * 72)

print(
    display_df.to_string(index=False)
)

# Identify the best F1 result among the tested thresholds.
best_f1_row = results_df.loc[
    results_df["f1_score"].idxmax()
]

# Identify the best recall among thresholds achieving validation
# FPR <= 1%.
eligible = results_df[
    results_df["false_positive_rate"] <= 0.01
]

print("\n" + "=" * 72)
print("EXPERIMENT SUMMARY")
print("=" * 72)

print(
    "Best F1 threshold:",
    f"{best_f1_row['threshold']:.6f}"
)

print(
    "Best F1 score:",
    f"{best_f1_row['f1_score']:.4%}"
)

if not eligible.empty:
    best_recall_row = eligible.loc[
        eligible["recall"].idxmax()
    ]

    print("\nBest recall with validation FPR <= 1%:")

    print(
        "Threshold:",
        f"{best_recall_row['threshold']:.6f}"
    )

    print(
        "Recall:",
        f"{best_recall_row['recall']:.4%}"
    )

    print(
        "FPR:",
        f"{best_recall_row['false_positive_rate']:.4%}"
    )
else:
    print(
        "\nNo tested threshold achieved validation FPR <= 1%."
    )

print("\nOriginal threshold:", ORIGINAL_THRESHOLD)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nNo model training or test-set evaluation was performed.")

print("=" * 72)
