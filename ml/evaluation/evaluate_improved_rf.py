
# Import the libraries needed for evaluation.
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Import the same feature extractor used to train the improved model.
from ml.train_static_improved import improved_features

# Locate the project directories.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "metrics"

# Load the improved model and its feature names.
model = joblib.load(MODEL_DIR / "static_model_improved.joblib")
feature_names = joblib.load(
    MODEL_DIR / "static_improved_features.joblib"
)

# Load validation data.
validation = pd.read_csv(DATA_DIR / "validation.csv")
validation_urls = pd.read_csv(DATA_DIR / "validation_urls.csv")

# Verify that the URL rows and labels align.
if len(validation) != len(validation_urls):
    raise ValueError("Validation row counts do not match.")

if not np.array_equal(
    validation["label"].to_numpy(),
    validation_urls["label"].to_numpy(),
):
    raise ValueError("Validation labels do not match.")

# Recreate the improved features in their training order.
X = pd.DataFrame(
    [improved_features(url) for url in validation_urls["url"]]
)[feature_names]

y = validation["label"].astype(int)

# Calculate phishing probability estimates.
phishing_index = list(model.classes_).index(1)
scores = model.predict_proba(X)[:, phishing_index]

print("\nIMPROVED RANDOM FOREST THRESHOLD COMPARISON")

# Evaluate several decision thresholds on validation data only.
for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn)

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision_score(y, predictions):.4f} | "
        f"Recall={recall_score(y, predictions):.4f} | "
        f"F1={f1_score(y, predictions):.4f} | "
        f"FPR={fpr:.4f} | "
        f"FP={fp:,} | FN={fn:,}"
    )

# Use the default threshold to examine error categories.
predictions = (scores >= 0.50).astype(int)

analysis = validation.copy()
analysis["prediction"] = predictions

# Define the same URL-length categories used previously.
analysis["length_category"] = pd.cut(
    analysis["url_length"],
    bins=[0, 20, 30, 50, 100, float("inf")],
    labels=["1-20", "21-30", "31-50", "51-100", "101+"],
    include_lowest=True,
)

print("\nIMPROVED MODEL ERROR RATES BY CATEGORY")

# Analyze URL-length categories.
for category, group in analysis.groupby(
    "length_category", observed=True
):
    legitimate = group[group["label"] == 0]
    phishing = group[group["label"] == 1]

    fp = int((legitimate["prediction"] == 1).sum())
    fn = int((phishing["prediction"] == 0).sum())

    fpr = fp / len(legitimate) if len(legitimate) else 0
    fnr = fn / len(phishing) if len(phishing) else 0

    print(
        f"Length {category}: FPR={fpr:.4f}, FNR={fnr:.4f}"
    )

# Analyze HTTP and HTTPS separately.
for https_flag, group in analysis.groupby("is_https"):
    legitimate = group[group["label"] == 0]
    phishing = group[group["label"] == 1]

    fp = int((legitimate["prediction"] == 1).sum())
    fn = int((phishing["prediction"] == 0).sum())

    fpr = fp / len(legitimate) if len(legitimate) else 0
    fnr = fn / len(phishing) if len(phishing) else 0

    protocol = "HTTPS" if https_flag == 1 else "HTTP"

    print(
        f"{protocol}: FPR={fpr:.4f}, FNR={fnr:.4f}"
    )

# Check known legitimate examples without visiting them.
print("\nKNOWN LEGITIMATE URL CHECKS")

examples = [
    "https://example.com/",
    "https://chatgpt.com/",
    "https://chatgpt.com/c/00000000-0000-0000-0000-000000000000",
]

example_features = pd.DataFrame(
    [improved_features(url) for url in examples]
)[feature_names]

example_scores = model.predict_proba(
    example_features
)[:, phishing_index]

for url, score in zip(examples, example_scores):
    print(f"{url} -> phishing score: {score:.4f}")

# Save validation threshold comparison if needed later.
print("\nEvaluation completed. Production model unchanged.")
