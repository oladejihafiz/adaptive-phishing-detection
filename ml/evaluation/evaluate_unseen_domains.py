
# STEP 39: EVALUATE RANDOM FOREST MODELS ON UNSEEN DOMAINS
#
# Compare the original 15-feature model and enhanced 40-feature
# model on the same validation URLs.
#
# This script does NOT retrain or modify either model.

from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import tldextract

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from ml.enhanced_features import enhanced_features


# Locate project folders.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use the bundled Public Suffix List without downloading data.
extractor = tldextract.TLDExtract(suffix_list_urls=())


def get_registered_domain(url):
    """
    Return the registered domain of a URL.

    Example:
    https://mail.google.com -> google.com
    """

    url = str(url).strip()

    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url

    hostname = (urlsplit(url).hostname or "").lower()

    extracted = extractor(hostname)

    return extracted.top_domain_under_public_suffix or hostname


def evaluate(name, actual, predictions):
    """
    Calculate metrics for one model on one group of URLs.
    """

    # labels=[0, 1] keeps the confusion matrix order consistent.
    tn, fp, fn, tp = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1]
    ).ravel()

    # Avoid division by zero if a category has no examples.
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    result = {
        "group_and_model": name,
        "total_urls": len(actual),
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "f1": f1_score(
            actual, predictions, zero_division=0
        ),
        "fpr": fpr,
        "fnr": fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print(f"\n{name}")
    print("-" * 55)
    print(f"Total URLs: {len(actual):,}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-score: {result['f1']:.4f}")
    print(f"False-positive rate: {result['fpr']:.4f}")
    print(f"False-negative rate: {result['fnr']:.4f}")
    print(f"TN: {tn:,} | FP: {fp:,}")
    print(f"FN: {fn:,} | TP: {tp:,}")

    return result


def main():

    print("Loading training and validation datasets...")

    train_urls = pd.read_csv(DATA_DIR / "train_urls.csv")
    validation_urls = pd.read_csv(
        DATA_DIR / "validation_urls.csv"
    )

    # These are the original 15-feature validation records.
    original_validation = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    # Verify that both validation files contain matching labels.
    if len(validation_urls) != len(original_validation):
        raise ValueError("Validation row counts do not match.")

    if not np.array_equal(
        validation_urls["label"].to_numpy(),
        original_validation["label"].to_numpy()
    ):
        raise ValueError("Validation labels are not aligned.")

    print("Identifying domains encountered during training...")

    # Build the set of domains seen by the models during training.
    training_domains = set(
        train_urls["url"].map(get_registered_domain)
    )

    # Mark validation URLs that share a training domain.
    validation_domains = validation_urls["url"].map(
        get_registered_domain
    )

    seen_mask = validation_domains.isin(
        training_domains
    ).to_numpy()

    unseen_mask = ~seen_mask

    print(f"Seen-domain validation URLs: {seen_mask.sum():,}")
    print(f"Unseen-domain validation URLs: {unseen_mask.sum():,}")

    print("\nLoading trained models...")

    original_model = joblib.load(
        MODEL_DIR / "static_model.joblib"
    )

    enhanced_model = joblib.load(
        MODEL_DIR / "static_model_enhanced.joblib"
    )

    # Load the exact feature order used by the enhanced model.
    enhanced_feature_names = joblib.load(
        MODEL_DIR / "static_enhanced_features.joblib"
    )

    # Prepare the original model's numerical inputs.
    X_original = original_validation.drop(
        columns=["label"]
    )

    # Prepare the enhanced model's 40-feature inputs.
    print("Extracting enhanced validation features...")

    X_enhanced = pd.DataFrame([
        enhanced_features(url)
        for url in validation_urls["url"]
    ])

    X_enhanced = X_enhanced[enhanced_feature_names]

    actual = validation_urls["label"].astype(int).to_numpy()

    print("\nGenerating predictions...")

    # Both models use their default prediction rules.
    # We are NOT tuning thresholds in this experiment.
    original_predictions = original_model.predict(X_original)
    enhanced_predictions = enhanced_model.predict(X_enhanced)

    groups = {
        "ALL VALIDATION URLS": np.ones(
            len(actual), dtype=bool
        ),
        "SEEN DOMAINS": seen_mask,
        "UNSEEN DOMAINS": unseen_mask,
    }

    results = []

    # Evaluate both models on each identical subset.
    for group_name, mask in groups.items():

        results.append(
            evaluate(
                group_name + " - ORIGINAL RF",
                actual[mask],
                original_predictions[mask]
            )
        )

        results.append(
            evaluate(
                group_name + " - ENHANCED RF",
                actual[mask],
                enhanced_predictions[mask]
            )
        )

    # Save summary metrics, not raw URL records.
    output_file = RESULTS_DIR / "unseen_domain_comparison.csv"

    pd.DataFrame(results).to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 55)
    print("Comparison completed successfully.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
