
# Import libraries for training, evaluation, and saving the model.
from pathlib import Path

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

# Import the exact 40-feature extractor.
from ml.enhanced_features import enhanced_features

# Locate the project directories.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "metrics"

# Create output directories if necessary.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_feature_matrix(urls):
    """
    Convert a collection of URLs into a table of numerical features.

    Each row represents one URL.
    Each column represents one of our 40 features.
    """
    rows = []

    for url in urls:
        rows.append(enhanced_features(url))

    return pd.DataFrame(rows)


def evaluate_model(name, model, X, y):
    """
    Calculate the classification metrics for a trained model.
    """
    predictions = model.predict(X)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn)

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y, predictions), 4))
    print("Precision:", round(precision_score(y, predictions), 4))
    print("Recall:", round(recall_score(y, predictions), 4))
    print("F1-score:", round(f1_score(y, predictions), 4))
    print("False-positive rate:", round(fpr, 4))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))


def main():
    print("Loading URL datasets...")

    train_urls = pd.read_csv(DATA_DIR / "train_urls.csv")
    validation_urls = pd.read_csv(
        DATA_DIR / "validation_urls.csv"
    )

    # Load the original numerical datasets to verify alignment.
    original_train = pd.read_csv(DATA_DIR / "train.csv")
    original_validation = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    # Check that URL records match the existing feature datasets.
    if len(train_urls) != len(original_train):
        raise ValueError("Training records are not aligned.")

    if len(validation_urls) != len(original_validation):
        raise ValueError("Validation records are not aligned.")

    # Confirm that labels agree row by row.
    if not np.array_equal(
        train_urls["label"].to_numpy(),
        original_train["label"].to_numpy(),
    ):
        raise ValueError("Training labels do not match.")

    if not np.array_equal(
        validation_urls["label"].to_numpy(),
        original_validation["label"].to_numpy(),
    ):
        raise ValueError("Validation labels do not match.")

    # Extract the enhanced features.
    print("Extracting 40-feature training data...")
    X_train = build_feature_matrix(train_urls["url"])

    print("Extracting 40-feature validation data...")
    X_validation = build_feature_matrix(
        validation_urls["url"]
    )

    # Store the feature order for consistent predictions.
    feature_names = list(X_train.columns)

    X_validation = X_validation[feature_names]

    y_train = train_urls["label"].astype(int)
    y_validation = validation_urls["label"].astype(int)

    # Train a new model without changing the original models.
    print("\nTraining enhanced Random Forest...")

    enhanced_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    enhanced_model.fit(X_train, y_train)

    # Load the original Random Forest.
    original_model = joblib.load(
        MODEL_DIR / "static_model.joblib"
    )

    # Load the first experimental Random Forest.
    improved_model = joblib.load(
        MODEL_DIR / "static_model_improved.joblib"
    )

    # Prepare the original 15 features.
    X_original = original_validation.drop(
        columns=["label"]
    )

    # Prepare the first experimental model's 29 features.
    from ml.train_static_improved import improved_features

    improved_feature_names = joblib.load(
        MODEL_DIR / "static_improved_features.joblib"
    )

    X_improved = pd.DataFrame(
        [
            improved_features(url)
            for url in validation_urls["url"]
        ]
    )[improved_feature_names]

    # Compare all three models at the default threshold.
    evaluate_model(
        "ORIGINAL RANDOM FOREST (15 FEATURES)",
        original_model,
        X_original,
        y_validation,
    )

    evaluate_model(
        "IMPROVED RANDOM FOREST (29 FEATURES)",
        improved_model,
        X_improved,
        y_validation,
    )

    evaluate_model(
        "ENHANCED RANDOM FOREST (40 FEATURES)",
        enhanced_model,
        X_validation,
        y_validation,
    )

    # Save the enhanced model under a separate filename.
    model_path = MODEL_DIR / "static_model_enhanced.joblib"
    joblib.dump(enhanced_model, model_path)

    # Save its feature order.
    features_path = MODEL_DIR / "static_enhanced_features.joblib"
    joblib.dump(feature_names, features_path)

    print(f"\nEnhanced model saved to: {model_path}")
    print(f"Feature names saved to: {features_path}")

    print("\nAll original models remain unchanged.")


if __name__ == "__main__":
    main()
