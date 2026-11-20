
# Import Path to locate project files.
from pathlib import Path

# Import joblib to load our trained Random Forest.
import joblib

# Import pandas for reading data and saving results.
import pandas as pd

# Import evaluation metrics.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Locate the project root directory.
ROOT = Path(__file__).resolve().parents[1]

# Load the validation dataset, not the test dataset.
data = pd.read_csv(
    ROOT / "dataset" / "processed" / "validation.csv"
)

# Separate URL features and correct labels.
X = data.drop(columns=["label"])
y = data["label"].astype(int)

# Load the existing trained Random Forest.
model = joblib.load(
    ROOT / "models" / "static_model.joblib"
)

# Get the estimated probability of phishing for every URL.
# Class 1 represents phishing.
phishing_class_index = list(model.classes_).index(1)
probabilities = model.predict_proba(X)[:, phishing_class_index]

# Test several possible decision thresholds.
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Store evaluation results for comparison.
results = []

print("RANDOM FOREST THRESHOLD EVALUATION")
print(f"Validation URLs: {len(data):,}")

for threshold in thresholds:

    # Classify a URL as phishing when its estimated
    # phishing probability reaches the chosen threshold.
    predictions = (probabilities >= threshold).astype(int)

    # Calculate the confusion matrix.
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    # Calculate the proportion of legitimate URLs
    # incorrectly classified as phishing.
    fpr = fp / (fp + tn) if fp + tn else 0

    # Calculate model performance.
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    accuracy = accuracy_score(y, predictions)

    # Save the results for later analysis.
    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "false_positives": int(fp),
        "false_negatives": int(fn),
    })

    # Display a summary for each threshold.
    print(
        f"Threshold={threshold:.2f} | "
        f"F1={f1:.4f} | "
        f"Recall={recall:.4f} | "
        f"FPR={fpr:.4f} | "
        f"FP={fp} | FN={fn}"
    )

# Create the results directory if it does not exist.
output_dir = ROOT / "results" / "metrics"
output_dir.mkdir(parents=True, exist_ok=True)

# Save the threshold comparison to a CSV file.
output_file = output_dir / "rf_threshold_comparison.csv"
pd.DataFrame(results).to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
