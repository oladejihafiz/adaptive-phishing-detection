
# Import libraries for processing URLs, training, and evaluation.
from pathlib import Path
from urllib.parse import urlsplit
import ipaddress
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Import the original 15-feature extractor.
from ml.prepare_data import features

# Locate project folders.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "metrics"

# Ensure output directories exist.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_url(url):
    """
    Prepare a URL for feature extraction without visiting it.
    """
    url = str(url).strip()

    # Add a scheme if the URL does not already have one.
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url

    return url


def improved_features(url):
    """
    Combine the original 15 features with additional URL features.
    All features are calculated locally; no network requests occur.
    """
    url = normalize_url(url)

    # Extract URL components.
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    # Begin with the original feature set.
    result = features(url)

    # Separate hostname and path digit counts.
    result["hostname_digits"] = sum(c.isdigit() for c in host)
    result["path_digits"] = sum(c.isdigit() for c in path)

    # Measure the proportion of digits rather than only their count.
    result["hostname_digit_ratio"] = (
        result["hostname_digits"] / max(len(host), 1)
    )

    result["url_digit_ratio"] = (
        result["num_digits"] / max(len(url), 1)
    )

    # Distinguish hyphens in hostnames from hyphens in paths.
    result["hostname_hyphens"] = host.count("-")
    result["path_hyphens"] = path.count("-")

    # Identify common URL structural patterns.
    result["query_length"] = len(query)
    result["has_fragment"] = int(bool(parsed.fragment))
    result["has_percent_encoding"] = int(bool(re.search(r"%[0-9a-fA-F]{2}", url)))
    result["has_explicit_port"] = int(
        bool(re.search(r":\d+$", parsed.netloc))
    )

    # Count hostname labels beyond the final two.
    # This is only an approximation of subdomain depth,
    # because some public suffixes contain multiple labels.
    host_parts = [part for part in host.split(".") if part]
    result["approx_subdomain_depth"] = max(len(host_parts) - 2, 0)

    # Calculate path structure.
    result["path_segments"] = len(
        [segment for segment in path.split("/") if segment]
    )

    # Separate suspicious words by URL component.
    suspicious_words = (
        "verify",
        "login",
        "account",
        "secure",
        "update",
        "password",
        "bank",
    )

    result["hostname_suspicious_words"] = sum(
        word in host for word in suspicious_words
    )

    result["path_suspicious_words"] = sum(
        word in path.lower() for word in suspicious_words
    )

    return result


def load_urls(filename):
    """
    Load URL records and check the expected columns.
    """
    data = pd.read_csv(DATA_DIR / filename)

    if "url" not in data.columns or "label" not in data.columns:
        raise ValueError(
            f"{filename} must contain 'url' and 'label' columns. "
            f"Found: {list(data.columns)}"
        )

    return data


def build_features(data):
    """
    Convert every URL into a numerical feature row.
    """
    rows = []

    for url in data["url"]:
        rows.append(improved_features(url))

    return pd.DataFrame(rows)


def report_metrics(name, y_true, predictions):
    """
    Print evaluation results for one model.
    """
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_true, predictions), 4))
    print("Precision:", round(precision_score(y_true, predictions), 4))
    print("Recall:", round(recall_score(y_true, predictions), 4))
    print("F1-score:", round(f1_score(y_true, predictions), 4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, predictions))


def main():
    print("Loading training and validation URLs...")

    train_urls = load_urls("train_urls.csv")
    validation_urls = load_urls("validation_urls.csv")

    # Load the original feature datasets to verify row alignment.
    original_train = pd.read_csv(DATA_DIR / "train.csv")
    original_validation = pd.read_csv(DATA_DIR / "validation.csv")

    if len(train_urls) != len(original_train):
        raise ValueError("Training URL and feature row counts differ.")

    if len(validation_urls) != len(original_validation):
        raise ValueError("Validation URL and feature row counts differ.")

    # Check that URL labels match the existing feature datasets.
    if not np.array_equal(
        train_urls["label"].to_numpy(),
        original_train["label"].to_numpy(),
    ):
        raise ValueError("Training labels are not aligned.")

    if not np.array_equal(
        validation_urls["label"].to_numpy(),
        original_validation["label"].to_numpy(),
    ):
        raise ValueError("Validation labels are not aligned.")

    print("Extracting improved training features...")
    X_train = build_features(train_urls)
    y_train = train_urls["label"].astype(int)

    print("Extracting improved validation features...")
    X_validation = build_features(validation_urls)
    y_validation = validation_urls["label"].astype(int)

    # Train a separate experimental model using the same RF settings.
    print("\nTraining experimental Random Forest...")

    improved_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    improved_model.fit(X_train, y_train)

    # Evaluate the experimental model at the baseline threshold.
    improved_predictions = improved_model.predict(X_validation)

    # Load the original model for a fair comparison.
    baseline_model = joblib.load(
        MODEL_DIR / "static_model.joblib"
    )

    baseline_X = original_validation.drop(columns=["label"])
    baseline_predictions = baseline_model.predict(baseline_X)

    report_metrics(
        "ORIGINAL RANDOM FOREST",
        y_validation,
        baseline_predictions,
    )

    report_metrics(
        "IMPROVED EXPERIMENTAL RANDOM FOREST",
        y_validation,
        improved_predictions,
    )

    # Save the new model under a different filename.
    # The original production model remains unchanged.
    output_path = MODEL_DIR / "static_model_improved.joblib"
    joblib.dump(improved_model, output_path)

    print(f"\nExperimental model saved to: {output_path}")

    # Save the feature names to support later backend integration.
    feature_path = MODEL_DIR / "static_improved_features.joblib"
    joblib.dump(list(X_train.columns), feature_path)

    print(f"Feature names saved to: {feature_path}")


if __name__ == "__main__":
    main()
