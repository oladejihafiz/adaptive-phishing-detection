
# STEP 42: ENHANCED RANDOM FOREST - FRSC CASE STUDY
#
# Purpose:
# Compare the original and enhanced Random Forest models
# using the same 24 independently verified phishing URLs.
#
# This script does not train, update, or modify either model.

from pathlib import Path

import joblib
import pandas as pd

# Original 15-feature extraction function.
from ml.prepare_data import features

# Enhanced 40-feature extraction function.
from ml.enhanced_features import enhanced_features


# Locate the main project directory.
BASE_DIR = Path(__file__).resolve().parents[1]

# Locate the independent FRSC case-study dataset.
DATA_FILE = BASE_DIR / "dataset" / "raw" / "dataset_5.csv"

# Locate the trained Random Forest models.
ORIGINAL_MODEL_FILE = (
    BASE_DIR / "models" / "static_model.joblib"
)

ENHANCED_MODEL_FILE = (
    BASE_DIR / "models" / "static_model_enhanced.joblib"
)

# Locate the saved enhanced feature order.
ENHANCED_FEATURES_FILE = (
    BASE_DIR / "models" / "static_enhanced_features.joblib"
)

# Locate the results directory.
RESULT_DIR = BASE_DIR / "results" / "metrics"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# Preserve the original model's 15-feature order.
ORIGINAL_FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_at",
    "num_question",
    "num_equals",
    "num_slashes",
    "is_https",
    "has_ip_host",
    "num_host_parts",
    "has_punycode",
    "suspicious_words",
]


def main():

    print("Loading independent FRSC case-study dataset...")

    # Read the same dataset used in the earlier evaluation.
    data = pd.read_csv(DATA_FILE)

    # Verify that all 24 records are phishing examples.
    assert len(data) == 24, (
        "Expected exactly 24 FRSC phishing URLs."
    )

    assert (data["label_code"] == 1).all(), (
        "The FRSC case study must contain phishing-only labels."
    )

    print("Loading trained Random Forest models...")

    original_model = joblib.load(ORIGINAL_MODEL_FILE)
    enhanced_model = joblib.load(ENHANCED_MODEL_FILE)

    # Load the exact feature order used in enhanced training.
    enhanced_feature_names = joblib.load(
        ENHANCED_FEATURES_FILE
    )

    print("Extracting original 15 features...")

    original_features = pd.DataFrame([
        features(str(url))
        for url in data["url"]
    ])[ORIGINAL_FEATURE_NAMES]

    print("Extracting enhanced 40 features...")

    enhanced_feature_frame = pd.DataFrame([
        enhanced_features(str(url))
        for url in data["url"]
    ])[enhanced_feature_names]

    print("Generating predictions...")

    # Predict using the original Random Forest.
    original_predictions = original_model.predict(
        original_features
    ).astype(int)

    # Obtain enhanced-model phishing probabilities.
    phishing_column = list(
        enhanced_model.classes_
    ).index(1)

    enhanced_scores = enhanced_model.predict_proba(
        enhanced_feature_frame
    )[:, phishing_column]

    # Use the threshold selected before this case study.
    enhanced_predictions = (
        enhanced_scores >= 0.50
    ).astype(int)

    # Count correctly detected phishing URLs.
    total = len(data)

    original_detected = int(
        (original_predictions == 1).sum()
    )

    enhanced_detected = int(
        (enhanced_predictions == 1).sum()
    )

    # Recall is the proportion of phishing URLs detected.
    original_recall = original_detected / total
    enhanced_recall = enhanced_detected / total

    # Save predictions without changing the original results file.
    results = pd.DataFrame({
        "url": data["url"],
        "actual_label": data["label_code"],
        "original_rf_prediction": original_predictions,
        "enhanced_rf_prediction": enhanced_predictions,
        "enhanced_phishing_score": enhanced_scores,
    })

    results["both_detected"] = (
        (original_predictions == 1)
        & (enhanced_predictions == 1)
    )

    results["both_missed"] = (
        (original_predictions == 0)
        & (enhanced_predictions == 0)
    )

    output_file = (
        RESULT_DIR / "enhanced_frsc_case_study.csv"
    )

    results.to_csv(output_file, index=False)

    # Display the summary without printing phishing URLs.
    print("\n" + "=" * 55)
    print("INDEPENDENT FRSC PHISHING CASE STUDY")
    print("=" * 55)

    print(f"Total phishing URLs: {total}")

    print("\nORIGINAL RANDOM FOREST")
    print(f"Detected: {original_detected}")
    print(f"Missed: {total - original_detected}")
    print(f"Recall: {original_recall:.2%}")

    print("\nENHANCED RANDOM FOREST")
    print(f"Detected: {enhanced_detected}")
    print(f"Missed: {total - enhanced_detected}")
    print(f"Recall: {enhanced_recall:.2%}")

    print("\nMODEL AGREEMENT")
    print(
        "Both detected:",
        int(results["both_detected"].sum())
    )
    print(
        "Both missed:",
        int(results["both_missed"].sum())
    )
    print(
        "Models disagreed:",
        int(
            (original_predictions != enhanced_predictions).sum()
        )
    )

    print("\nResults saved to:", output_file)
    print("No trained models were modified.")


if __name__ == "__main__":
    main()
