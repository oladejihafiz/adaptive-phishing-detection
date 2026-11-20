
# STEP 41: FINAL HELD-OUT TEST EVALUATION
#
# Compare the original and enhanced Random Forest models
# using the same test URLs.
#
# The enhanced threshold is fixed at 0.50.
# This script does NOT train or modify any model.

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from ml.enhanced_features import enhanced_features


# Locate project folders.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(name, actual, predictions):
    """Calculate metrics from the actual and predicted labels."""

    tn, fp, fn, tp = confusion_matrix(
        actual, predictions, labels=[0, 1]
    ).ravel()

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    fpr = fp / (fp + tn) if fp + tn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    result = {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print(f"\n{name}")
    print("-" * 55)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"False-positive rate: {fpr:.4f}")
    print(f"TN: {tn:,} | FP: {fp:,}")
    print(f"FN: {fn:,} | TP: {tp:,}")

    return result


def main():
    print("Loading held-out test datasets...")

    # Raw URLs and labels for the test set.
    test_urls = pd.read_csv(
        DATA_DIR / "test_urls.csv"
    )

    # Original 15-feature test records.
    original_test = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    # Verify that both files refer to aligned records.
    if len(test_urls) != len(original_test):
        raise ValueError("Test dataset row counts differ.")

    if not np.array_equal(
        test_urls["label"].to_numpy(),
        original_test["label"].to_numpy()
    ):
        raise ValueError("Test dataset labels are not aligned.")

    actual = test_urls["label"].astype(int).to_numpy()

    print("Loading original and enhanced models...")

    original_model = joblib.load(
        MODEL_DIR / "static_model.joblib"
    )

    enhanced_model = joblib.load(
        MODEL_DIR / "static_model_enhanced.joblib"
    )

    # Load the feature order used during enhanced training.
    enhanced_features_order = joblib.load(
        MODEL_DIR / "static_enhanced_features.joblib"
    )

    # Prepare the original model inputs.
    X_original = original_test.drop(columns=["label"])

    # Prepare enhanced inputs from the same test URLs.
    print("Extracting 40 features from test URLs...")

    X_enhanced = pd.DataFrame([
        enhanced_features(url)
        for url in test_urls["url"]
    ])[enhanced_features_order]

    print("Generating test predictions...")

    # Use the original model's standard prediction method.
    original_predictions = original_model.predict(
        X_original
    )

    # Get phishing scores from the enhanced model.
    phishing_column = list(
        enhanced_model.classes_
    ).index(1)

    enhanced_scores = enhanced_model.predict_proba(
        X_enhanced
    )[:, phishing_column]

    # Apply the threshold chosen BEFORE test evaluation.
    enhanced_predictions = (
        enhanced_scores >= 0.50
    ).astype(int)

    print("\nFINAL TEST RESULTS")
    print("=" * 55)

    original_results = evaluate(
        "ORIGINAL RANDOM FOREST",
        actual,
        original_predictions
    )

    enhanced_results = evaluate(
        "ENHANCED RANDOM FOREST (THRESHOLD 0.50)",
        actual,
        enhanced_predictions
    )

    # Save summary results for the final report.
    output_file = (
        RESULTS_DIR / "final_rf_test_comparison.csv"
    )

    pd.DataFrame([
        original_results,
        enhanced_results
    ]).to_csv(output_file, index=False)

    print("\n" + "=" * 55)
    print("Final test evaluation completed.")
    print(f"Results saved to: {output_file}")
    print("No models or datasets were modified.")


if __name__ == "__main__":
    main()

