
# Load saved machine-learning models.
import joblib

# Read the held-out test dataset.
import pandas as pd

# Import Path to manage the results directory.
from pathlib import Path

# Calculate classification performance.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Load the same untouched test dataset used in Step 19.
data = pd.read_csv("dataset/processed/test.csv")

# Separate input features from their correct labels.
X = data.drop(columns=["label"])
y = data["label"].astype(int)

# Load the original pretrained models and scaler.
rf = joblib.load("models/static_model.joblib")
pa = joblib.load("models/adaptive_model.joblib")
scaler = joblib.load("models/scaler.joblib")

# Get predictions without training either model.
rf_pred = rf.predict(X)
pa_pred = pa.predict(scaler.transform(X))

# Compare three different ways to make a final decision.
strategies = {
    "Random Forest only": rf_pred,

    # Flag phishing when at least one model predicts phishing.
    "Either model (OR)": ((rf_pred == 1) | (pa_pred == 1)).astype(int),

    # Flag phishing only when both models predict phishing.
    "Both models (AND)": ((rf_pred == 1) & (pa_pred == 1)).astype(int),
}

# Store the evaluation results for the project report.
comparison_results = []

# Evaluate every strategy on the same test examples.
for name, predictions in strategies.items():
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    # False-positive rate: legitimate URLs incorrectly flagged.
    fpr = fp / (fp + tn) if fp + tn else 0

    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    print(f"Precision: {precision_score(y, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y, predictions, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y, predictions, zero_division=0):.4f}")
    print(f"False-positive rate: {fpr:.4f}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Store this strategy's performance measurements.
    comparison_results.append({
        "strategy": name,
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1_score": f1_score(y, predictions, zero_division=0),
        "false_positive_rate": fpr,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    })


# Create the output directory if it does not already exist.

output_dir = Path("results/metrics")
output_dir.mkdir(parents=True, exist_ok=True)

# Save all three strategies in one CSV file.
output_file = output_dir / "combined_model_comparison.csv"

pd.DataFrame(comparison_results).to_csv(
    output_file,
    index=False
)

print(f"\nCombined evaluation saved to: {output_file}")
