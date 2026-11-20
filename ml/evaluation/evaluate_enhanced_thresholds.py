
# STEP 40: ENHANCED MODEL THRESHOLD EVALUATION
#
# Purpose:
# Compare decision thresholds using validation data only.
# Show results for all URLs and unseen-domain URLs separately.
#
# This script does NOT retrain or modify the model.

from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import tldextract

from sklearn.metrics import confusion_matrix, f1_score
from ml.enhanced_features import enhanced_features


# Locate the project folders.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use the same offline domain extractor as earlier.
extractor = tldextract.TLDExtract(suffix_list_urls=())


def registered_domain(url):
    """Get the registered domain without visiting the website."""

    url = str(url).strip()

    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url

    hostname = (urlsplit(url).hostname or "").lower()

    result = extractor(hostname)

    return result.top_domain_under_public_suffix or hostname


def calculate_metrics(actual, predictions):
    """Calculate classification metrics from predictions."""

    tn, fp, fn, tp = confusion_matrix(
        actual, predictions, labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(actual, predictions, zero_division=0),
        "fpr": fpr,
        "fp": int(fp),
        "fn": int(fn),
    }


def main():
    print("Loading datasets...")

    train = pd.read_csv(DATA_DIR / "train_urls.csv")
    validation = pd.read_csv(DATA_DIR / "validation_urls.csv")

    print("Identifying unseen domains...")

    # Domains that appeared in the training dataset.
    training_domains = set(
        train["url"].map(registered_domain)
    )

    # True means the domain was absent from training.
    unseen_mask = ~validation["url"].map(
        registered_domain
    ).isin(training_domains).to_numpy()

    print(f"Unseen-domain URLs: {unseen_mask.sum():,}")

    print("Loading enhanced Random Forest...")

    model = joblib.load(
        MODEL_DIR / "static_model_enhanced.joblib"
    )

    feature_names = joblib.load(
        MODEL_DIR / "static_enhanced_features.joblib"
    )

    print("Extracting enhanced validation features...")

    X = pd.DataFrame([
        enhanced_features(url)
        for url in validation["url"]
    ])[feature_names]

    actual = validation["label"].astype(int).to_numpy()

    # Locate the probability column corresponding to phishing (1).
    phishing_column = list(model.classes_).index(1)

    scores = model.predict_proba(X)[:, phishing_column]

    # Thresholds to compare.
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    results = []

    for threshold in thresholds:

        # Predict phishing when the score reaches the threshold.
        predictions = (scores >= threshold).astype(int)

        # Evaluate on all validation URLs.
        all_metrics = calculate_metrics(actual, predictions)

        # Evaluate on URLs whose domains were not seen in training.
        unseen_metrics = calculate_metrics(
            actual[unseen_mask],
            predictions[unseen_mask]
        )

        print(f"\nTHRESHOLD: {threshold:.2f}")
        print("-" * 55)

        print(
            "All URLs:",
            f"F1={all_metrics['f1']:.4f}",
            f"Recall={all_metrics['recall']:.4f}",
            f"FPR={all_metrics['fpr']:.4f}",
            f"FP={all_metrics['fp']}",
            f"FN={all_metrics['fn']}"
        )

        print(
            "Unseen domains:",
            f"F1={unseen_metrics['f1']:.4f}",
            f"Recall={unseen_metrics['recall']:.4f}",
            f"FPR={unseen_metrics['fpr']:.4f}",
            f"FP={unseen_metrics['fp']}",
            f"FN={unseen_metrics['fn']}"
        )

        # Save the metrics for later documentation.
        for group, metrics in [
            ("all_validation", all_metrics),
            ("unseen_domains", unseen_metrics),
        ]:
            results.append({
                "threshold": threshold,
                "group": group,
                **metrics,
            })

    output_file = (
        RESULTS_DIR / "enhanced_threshold_comparison.csv"
    )

    pd.DataFrame(results).to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 55)
    print("Threshold evaluation completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
