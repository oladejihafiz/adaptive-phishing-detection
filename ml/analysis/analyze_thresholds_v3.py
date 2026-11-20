
"""
Analyse decision thresholds for the three domain-disjoint RF models.

This script:
1. Loads the saved validation probabilities.
2. Evaluates thresholds from 0.05 to 0.99.
3. Finds thresholds that meet selected false-positive-rate targets.
4. Compares precision, recall, F1, and accuracy.
5. Saves reports without modifying or retraining any model.

IMPORTANT:
Thresholds are selected using validation data only.
The test set remains untouched.
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
    roc_curve,
)


# ---------------------------------------------------------
# 1. PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "metrics_v3_grouped"

INPUT_FILE = RESULTS_DIR / "validation_probabilities_v3.csv"

# Evaluate these false-positive-rate constraints.
FPR_TARGETS = [0.05, 0.02, 0.01, 0.005, 0.001]

# Evaluate thresholds at regular intervals.
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.01), 2)

# Map each model to its saved phishing probability column.
MODEL_COLUMNS = {
    "15_features": "phishing_probability_15_features",
    "29_features": "phishing_probability_29_features",
    "40_features": "phishing_probability_40_features",
}


# ---------------------------------------------------------
# 2. LOAD VALIDATION PROBABILITIES
# ---------------------------------------------------------

def load_validation_data():
    """Load and validate the saved validation probabilities."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Validation probability file not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    required_columns = {"label", *MODEL_COLUMNS.values()}

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    y_true = data["label"].to_numpy(dtype=np.int8)

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1.")

    for model_name, column in MODEL_COLUMNS.items():

        probabilities = data[column].to_numpy(dtype=np.float64)

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"Invalid probabilities found for {model_name}."
            )

        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(
                f"Probabilities outside [0, 1] for {model_name}."
            )

    print(f"Loaded {len(data):,} validation records.")

    return data, y_true


# ---------------------------------------------------------
# 3. CALCULATE METRICS FOR A THRESHOLD
# ---------------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification metrics at a chosen threshold."""

    # Classify a URL as phishing when its probability meets
    # or exceeds the selected threshold.
    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    # Extract the four confusion-matrix values.
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    # Calculate the false-positive rate.
    fpr = fp / max(tn + fp, 1)

    # Return all relevant evaluation measurements.
    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "false_positive_rate": float(fpr),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "flagged_percentage": float(
            predictions.mean() * 100
        ),
    }


# ---------------------------------------------------------
# 4. FIND THE BEST THRESHOLD FOR AN FPR TARGET
# ---------------------------------------------------------

def find_threshold_for_fpr(
    y_true,
    probabilities,
    target_fpr,
):
    """
    Find the threshold with the highest recall while keeping
    the validation false-positive rate at or below the target.
    """

    # Generate the complete ROC curve without dropping
    # intermediate thresholds.
    fpr_values, recall_values, thresholds = roc_curve(
        y_true,
        probabilities,
        drop_intermediate=False,
    )

    # Identify thresholds satisfying the FPR constraint.
    eligible_indices = np.where(
        fpr_values <= target_fpr
    )[0]

    if len(eligible_indices) == 0:
        return None

    # Select the eligible threshold with maximum recall.
    best_index = max(
        eligible_indices,
        key=lambda index: (
            recall_values[index],
            thresholds[index],
        ),
    )

    selected_threshold = float(thresholds[best_index])

    # Calculate the complete metrics at that threshold.
    metrics = calculate_metrics(
        y_true,
        probabilities,
        selected_threshold,
    )

    metrics["target_false_positive_rate"] = target_fpr

    return metrics


# ---------------------------------------------------------
# 5. RUN THE THRESHOLD SWEEP
# ---------------------------------------------------------

def main():
    """Run threshold analysis for all three models."""

    print("=" * 70)
    print("PHISHING DETECTION THRESHOLD ANALYSIS")
    print("=" * 70)

    data, y_true = load_validation_data()

    all_sweep_results = []
    all_target_results = []

    # Analyse each model independently.
    for model_name, column in MODEL_COLUMNS.items():

        print("\n" + "=" * 60)
        print(f"ANALYSING {model_name.upper()}")
        print("=" * 60)

        probabilities = data[column].to_numpy(dtype=np.float64)

        # -------------------------------------------------
        # A. REGULAR THRESHOLD SWEEP
        # -------------------------------------------------

        print("\nEvaluating thresholds...")

        for threshold in THRESHOLDS:

            metrics = calculate_metrics(
                y_true,
                probabilities,
                threshold,
            )

            metrics["model"] = model_name

            all_sweep_results.append(metrics)

        # -------------------------------------------------
        # B. FPR-CONSTRAINED THRESHOLD SELECTION
        # -------------------------------------------------

        print("\nThresholds for selected FPR targets:")

        for target in FPR_TARGETS:

            result = find_threshold_for_fpr(
                y_true,
                probabilities,
                target,
            )

            if result is None:
                continue

            result["model"] = model_name

            all_target_results.append(result)

            print(
                f"Target FPR: {target * 100:.2f}% | "
                f"Threshold: {result['threshold']:.6f} | "
                f"Actual FPR: "
                f"{result['false_positive_rate'] * 100:.3f}% | "
                f"Recall: {result['recall'] * 100:.2f}% | "
                f"Precision: {result['precision'] * 100:.2f}% | "
                f"F1: {result['f1_score']:.4f}"
            )

    # -----------------------------------------------------
    # 6. SAVE THE THRESHOLD SWEEP
    # -----------------------------------------------------

    sweep_df = pd.DataFrame(all_sweep_results)

    sweep_path = RESULTS_DIR / "threshold_sweep_v3.csv"

    sweep_df.to_csv(sweep_path, index=False)

    # -----------------------------------------------------
    # 7. SAVE THE FPR TARGET RESULTS
    # -----------------------------------------------------

    target_df = pd.DataFrame(all_target_results)

    target_path = RESULTS_DIR / "fpr_target_analysis_v3.csv"

    target_df.to_csv(target_path, index=False)

    # -----------------------------------------------------
    # 8. DISPLAY THE 40-FEATURE RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("40-FEATURE MODEL: FPR TARGET COMPARISON")
    print("=" * 75)

    enhanced_results = target_df[
        target_df["model"] == "40_features"
    ]

    display_columns = [
        "target_false_positive_rate",
        "threshold",
        "false_positive_rate",
        "precision",
        "recall",
        "f1_score",
        "false_positive",
        "false_negative",
    ]

    if not enhanced_results.empty:

        print(
            enhanced_results[display_columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.5f}",
            )
        )

    print("\nReports saved:")
    print(f"  {sweep_path}")
    print(f"  {target_path}")

    print("\nThreshold analysis completed.")
    print("No model was retrained.")
    print("The test dataset was not used.")


# Execute the analysis.
if __name__ == "__main__":
    main()
