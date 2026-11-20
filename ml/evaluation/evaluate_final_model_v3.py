
"""
Final independent evaluation of the selected 40-feature phishing detector.

Uses:
- The already-trained 40-feature Random Forest.
- The threshold selected using validation data.
- The untouched domain-disjoint test dataset.

This script does not retrain the model or tune the threshold.
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
    roc_auc_score,
    average_precision_score,
)


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset" / "processed_v3_grouped"
MODEL_DIR = ROOT / "models_v3_grouped"
RESULTS_DIR = ROOT / "results" / "metrics_v3_grouped"

MODEL_FILE = MODEL_DIR / "random_forest_40_features.joblib"
FEATURE_FILE = MODEL_DIR / "random_forest_40_features_features.joblib"

TEST_FILE = DATA_DIR / "test.csv"
TEST_URLS_FILE = DATA_DIR / "test_urls.csv"

# This threshold was selected using validation data.
# Do not change it after inspecting test results.
FINAL_THRESHOLD = 0.747170

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. CONFIDENCE INTERVAL CALCULATION
# ---------------------------------------------------------

def wilson_interval(successes, total, z=1.96):
    """Calculate a 95% Wilson confidence interval for a proportion."""

    if total == 0:
        return [None, None]

    proportion = successes / total
    denominator = 1 + (z ** 2 / total)

    centre = (
        proportion + (z ** 2 / (2 * total))
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (proportion * (1 - proportion) / total)
            + (z ** 2 / (4 * total ** 2))
        )
        / denominator
    )

    return [
        float(max(0, centre - margin)),
        float(min(1, centre + margin)),
    ]


# ---------------------------------------------------------
# 3. LOAD MODEL AND TEST DATA
# ---------------------------------------------------------

def load_inputs():
    """Load the saved model, feature names, and untouched test data."""

    required_files = [
        MODEL_FILE,
        FEATURE_FILE,
        TEST_FILE,
        TEST_URLS_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("Loading selected Random Forest model...")

    model = joblib.load(MODEL_FILE)
    feature_names = joblib.load(FEATURE_FILE)

    print("Loading untouched test dataset...")

    test_data = pd.read_csv(TEST_FILE, low_memory=False)
    test_urls = pd.read_csv(TEST_URLS_FILE, low_memory=False)

    # Check that the feature matrix contains all expected columns.
    missing = set(feature_names) - set(test_data.columns)

    if missing:
        raise ValueError(
            f"Test dataset is missing features: {sorted(missing)}"
        )

    if "label" not in test_data.columns:
        raise ValueError("Test dataset has no label column.")

    # Extract the features in the exact training order.
    X_test = test_data[feature_names].to_numpy(dtype=np.float32)

    # Extract the true labels.
    y_test = test_data["label"].to_numpy(dtype=np.int8)

    # Confirm valid binary labels and finite feature values.
    if not set(np.unique(y_test)).issubset({0, 1}):
        raise ValueError("Unexpected test labels.")

    if not np.isfinite(X_test).all():
        raise ValueError("Non-finite test feature values detected.")

    # Verify URL metadata alignment.
    if len(test_urls) != len(y_test):
        raise ValueError(
            "Test URL metadata and feature row counts differ."
        )

    if not np.array_equal(
        test_urls["label"].to_numpy(dtype=np.int8),
        y_test,
    ):
        raise ValueError(
            "Test URL metadata labels do not align with test features."
        )

    # Confirm that the saved model expects the same number of features.
    if X_test.shape[1] != model.n_features_in_:
        raise ValueError(
            "Test feature count does not match the trained model."
        )

    print(f"Test records: {len(y_test):,}")
    print(f"Features: {X_test.shape[1]}")

    return model, feature_names, X_test, y_test, test_urls


# ---------------------------------------------------------
# 4. FINAL TEST EVALUATION
# ---------------------------------------------------------

def evaluate(model, X_test, y_test):
    """Evaluate the selected model at the locked threshold."""

    print("\nGenerating test predictions...")

    # Get the probability associated with the phishing class.
    phishing_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(X_test)[:, phishing_index]

    # Apply the threshold selected on validation data.
    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(np.int8)

    # Generate the confusion matrix.
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    # Calculate classification metrics.
    fpr = fp / max(tn + fp, 1)
    fnr = fn / max(fn + tp, 1)

    metrics = {
        "model": "random_forest_40_features",
        "feature_count": int(X_test.shape[1]),
        "decision_threshold": FINAL_THRESHOLD,
        "test_records": int(len(y_test)),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "false_positive_rate_95_percent_ci": wilson_interval(
            fp,
            tn + fp,
        ),
        "recall_95_percent_ci": wilson_interval(
            tp,
            tp + fn,
        ),
    }

    return metrics, probabilities, predictions


# ---------------------------------------------------------
# 5. SAVE TEST RESULTS
# ---------------------------------------------------------

def save_results(metrics, test_urls, probabilities, predictions):
    """Save the final report and per-URL test predictions."""

    # Save the full evaluation report.
    report_path = RESULTS_DIR / "final_test_metrics_v3.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save each test URL alongside its actual and predicted labels.
    prediction_data = pd.DataFrame({
        "url": test_urls["url"].to_numpy(),
        "actual_label": test_urls["label"].to_numpy(),
        "phishing_probability": probabilities,
        "predicted_label": predictions,
        "decision_threshold": FINAL_THRESHOLD,
    })

    predictions_path = RESULTS_DIR / "final_test_predictions_v3.csv"

    prediction_data.to_csv(
        predictions_path,
        index=False,
    )

    print(f"\nMetrics saved: {report_path}")
    print(f"Predictions saved: {predictions_path}")


# ---------------------------------------------------------
# 6. DISPLAY FINAL RESULTS
# ---------------------------------------------------------

def display_results(metrics):
    """Print the final test performance in a readable format."""

    matrix = metrics["confusion_matrix"]

    print("\n" + "=" * 65)
    print("FINAL TEST SET RESULTS")
    print("=" * 65)

    print(f"Model:              {metrics['model']}")
    print(f"Features:           {metrics['feature_count']}")
    print(f"Threshold:          {metrics['decision_threshold']:.6f}")

    print("\nClassification metrics:")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1-score:           {metrics['f1_score']:.4f}")
    print(f"False-positive rate:{metrics['false_positive_rate']:.4f}")
    print(f"False-negative rate:{metrics['false_negative_rate']:.4f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:             {metrics['pr_auc']:.4f}")

    print("\nConfusion matrix:")
    print(
        np.array([
            [matrix["true_negative"], matrix["false_positive"]],
            [matrix["false_negative"], matrix["true_positive"]],
        ])
    )

    fpr_ci = metrics["false_positive_rate_95_percent_ci"]
    recall_ci = metrics["recall_95_percent_ci"]

    print("\n95% confidence intervals:")
    print(
        f"FPR:    {fpr_ci[0] * 100:.3f}% - "
        f"{fpr_ci[1] * 100:.3f}%"
    )
    print(
        f"Recall: {recall_ci[0] * 100:.3f}% - "
        f"{recall_ci[1] * 100:.3f}%"
    )

    print("\nFinal evaluation completed.")
    print("No model retraining or threshold tuning was performed.")


# ---------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------

def main():
    """Run the locked final evaluation."""

    print("=" * 65)
    print("FINAL PHISHING DETECTOR EVALUATION")
    print("=" * 65)

    print(f"Locked decision threshold: {FINAL_THRESHOLD}")

    model, feature_names, X_test, y_test, test_urls = load_inputs()

    metrics, probabilities, predictions = evaluate(
        model,
        X_test,
        y_test,
    )

    save_results(
        metrics,
        test_urls,
        probabilities,
        predictions,
    )

    display_results(metrics)


if __name__ == "__main__":
    main()
