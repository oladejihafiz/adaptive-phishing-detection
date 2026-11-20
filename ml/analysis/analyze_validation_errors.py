
# Import libraries for reading data, loading models, and saving results.
from pathlib import Path
import joblib
import pandas as pd

# Locate the project root directory.
ROOT = Path(__file__).resolve().parents[1]

# Load the validation dataset and its matching URL records.
data = pd.read_csv(
    ROOT / "dataset" / "processed" / "validation.csv"
)

url_data = pd.read_csv(
    ROOT / "dataset" / "processed" / "validation_urls.csv"
)

# Ensure both files contain the same number of examples.
if len(data) != len(url_data):
    raise ValueError("Validation files have different row counts.")

# Confirm that row labels agree before joining URLs to features.
# Both files were generated in the same order during preprocessing.
if not (
    data["label"].to_numpy() == url_data["label"].to_numpy()
).all():
    raise ValueError("Validation labels do not match row by row.")

# Load the existing Random Forest model.
model = joblib.load(
    ROOT / "models" / "static_model.joblib"
)

# Separate features and labels.
X = data.drop(columns=["label"])
y = data["label"].astype(int)

# Calculate phishing probability estimates.
phishing_index = list(model.classes_).index(1)
probabilities = model.predict_proba(X)[:, phishing_index]

# Use the original 0.50 threshold for baseline error analysis.
predictions = (probabilities >= 0.50).astype(int)

# Create a copy containing URLs, labels, and predictions.
analysis = url_data.copy()
analysis["phishing_score"] = probabilities
analysis["prediction"] = predictions

# Attach the original numerical features for analysis.
for column in X.columns:
    analysis[column] = X[column].to_numpy()

# Identify false positives: legitimate URLs flagged as phishing.
false_positives = analysis[
    (analysis["label"] == 0) &
    (analysis["prediction"] == 1)
].copy()

# Identify false negatives: phishing URLs classified as legitimate.
false_negatives = analysis[
    (analysis["label"] == 1) &
    (analysis["prediction"] == 0)
].copy()

print("\nVALIDATION ERROR ANALYSIS")
print("Total validation URLs:", len(analysis))
print("False positives:", len(false_positives))
print("False negatives:", len(false_negatives))

# Compare the average feature values for both error categories.
print("\nAVERAGE FEATURES OF MISCLASSIFIED URLS")

comparison = pd.DataFrame({
    "false_positive_mean": false_positives[X.columns].mean(),
    "false_negative_mean": false_negatives[X.columns].mean(),
})

print(comparison.round(3).to_string())

# Examine how frequently short URLs occur among false positives.
print("\nFALSE POSITIVES BY URL LENGTH")

for maximum_length in [20, 30, 50, 100]:
    count = int(
        (false_positives["url_length"] <= maximum_length).sum()
    )

    percentage = (
        100 * count / len(false_positives)
        if len(false_positives) else 0
    )

    print(
        f"Length <= {maximum_length}: "
        f"{count:,} ({percentage:.2f}%)"
    )

# Create a directory for analysis results.
output_dir = ROOT / "results" / "metrics"
output_dir.mkdir(parents=True, exist_ok=True)

# Save the summary without exposing individual URL strings.
comparison.to_csv(
    output_dir / "validation_error_feature_summary.csv"
)

# Save the full error records locally for further investigation.
# These files may contain phishing URLs; do not open them in a browser.
false_positives.to_csv(
    output_dir / "validation_false_positives.csv",
    index=False
)

false_negatives.to_csv(
    output_dir / "validation_false_negatives.csv",
    index=False
)

print("\nError analysis files saved successfully.")
