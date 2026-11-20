
# Import Path to locate project files reliably.
from pathlib import Path

# Import joblib to load our previously trained models.
import joblib

# Import pandas to read the test dataset and save results.
import pandas as pd

# Import classification metrics for evaluating both models.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Locate the project's root directory.
ROOT = Path(__file__).resolve().parents[1]

# Define where our test data and trained models are stored.
TEST_FILE = ROOT / "dataset" / "processed" / "test.csv"
MODELS_DIR = ROOT / "models"

# Create the directory for evaluation results if necessary.
RESULTS_DIR = ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the untouched test dataset.
test_data = pd.read_csv(TEST_FILE)

# Separate the 15 URL features from the correct labels.
X_test = test_data.drop(columns=["label"])
y_test = test_data["label"].astype(int)

# Load the original trained Random Forest classifier.
static_model = joblib.load(MODELS_DIR / "static_model.joblib")

# Load the original trained Passive-Aggressive classifier.
adaptive_model = joblib.load(MODELS_DIR / "adaptive_model.joblib")

# Load the scaler fitted during adaptive model training.
scaler = joblib.load(MODELS_DIR / "scaler.joblib")

# Generate predictions from the Random Forest.
# This model uses the original, unscaled features.
static_predictions = static_model.predict(X_test)

# Scale features using the existing training scaler.
# We use transform(), NOT fit_transform(), to avoid data leakage.
X_test_scaled = scaler.transform(X_test)

# Generate predictions from the Passive-Aggressive model.
adaptive_predictions = adaptive_model.predict(X_test_scaled)


def evaluate_model(name, predictions):
    """
    Calculate classification metrics for one trained model.

    Label 0 represents legitimate URLs.
    Label 1 represents phishing URLs.
    """

    # Compute the confusion matrix in the order [0, 1].
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    # Calculate the false-positive rate.
    # This measures legitimate URLs incorrectly flagged as phishing.
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0

    # Collect all evaluation measurements.
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "false_positive_rate": false_positive_rate,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    # Display the evaluation results.
    print(f"\n{name.upper()}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"False-positive rate: {metrics['false_positive_rate']:.4f}")

    print("\nConfusion matrix:")
    print(f"True negatives: {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives: {tp}")

    return metrics


# Evaluate both models using the same test examples.
print("HELD-OUT TEST DATASET EVALUATION")
print(f"Total URLs: {len(test_data):,}")

static_results = evaluate_model(
    "Random Forest",
    static_predictions
)

adaptive_results = evaluate_model(
    "Passive-Aggressive",
    adaptive_predictions
)

# Save both sets of metrics for the project report.
results = pd.DataFrame([static_results, adaptive_results])

output_file = RESULTS_DIR / "test_model_comparison.csv"
results.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
