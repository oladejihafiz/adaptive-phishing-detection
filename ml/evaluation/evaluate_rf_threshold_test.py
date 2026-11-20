
# Import libraries for model loading and evaluation.
from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Locate the project root directory.
ROOT = Path(__file__).resolve().parents[1]

# Load the held-out test dataset.
test_data = pd.read_csv(
    ROOT / "dataset" / "processed" / "test.csv"
)

# Separate URL features and correct labels.
X_test = test_data.drop(columns=["label"])
y_test = test_data["label"].astype(int)

# Load the existing Random Forest without retraining it.
model = joblib.load(
    ROOT / "models" / "static_model.joblib"
)

# Identify the column corresponding to phishing (class 1).
phishing_index = list(model.classes_).index(1)

# Calculate phishing probabilities for the test URLs.
probabilities = model.predict_proba(X_test)[:, phishing_index]

# Compare the original threshold with the validation-selected
# threshold and the recall-preserving alternative.
thresholds = [0.50, 0.55, 0.60]

# Store results for the final project report.
results = []

print("RANDOM FOREST HELD-OUT THRESHOLD COMPARISON")
print(f"Test URLs: {len(test_data):,}")

for threshold in thresholds:

    # Convert phishing probabilities into class predictions.
    predictions = (probabilities >= threshold).astype(int)

    # Calculate the confusion matrix.
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    # Calculate the false-positive rate.
    fpr = fp / (fp + tn) if (fp + tn) else 0

    # Calculate classification metrics.
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    # Store all measurements.
    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    })

    # Display the results.
    print(
        f"Threshold={threshold:.2f} | "
        f"F1={f1:.4f} | "
        f"Recall={recall:.4f} | "
        f"FPR={fpr:.4f} | "
        f"FP={fp} | FN={fn}"
    )

# Save the comparison for your final report.
output_dir = ROOT / "results" / "metrics"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "rf_threshold_test_comparison.csv"

pd.DataFrame(results).to_csv(
    output_file,
    index=False
)

print(f"\nResults saved to: {output_file}")
