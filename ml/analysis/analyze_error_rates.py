
# Import libraries for loading data and calculating error rates.
from pathlib import Path
import joblib
import pandas as pd

# Locate the root directory of the project.
ROOT = Path(__file__).resolve().parents[1]

# Load the validation dataset.
data = pd.read_csv(
    ROOT / "dataset" / "processed" / "validation.csv"
)

# Load the original Random Forest model.
model = joblib.load(
    ROOT / "models" / "static_model.joblib"
)

# Separate features and correct labels.
X = data.drop(columns=["label"])
y = data["label"].astype(int)

# Generate baseline predictions using the original model.
predictions = model.predict(X)

# Store predictions alongside the original feature values.
analysis = data.copy()
analysis["prediction"] = predictions

# Define non-overlapping URL-length categories.
length_bins = [0, 20, 30, 50, 100, float("inf")]
length_labels = [
    "1-20",
    "21-30",
    "31-50",
    "51-100",
    "101+",
]

# Assign each URL to a length category.
analysis["length_category"] = pd.cut(
    analysis["url_length"],
    bins=length_bins,
    labels=length_labels,
    include_lowest=True
)


def calculate_error_rates(group):
    """
    Calculate error rates for one URL category.

    False-positive rate uses only legitimate URLs.
    False-negative rate uses only phishing URLs.
    """

    # Identify legitimate and phishing examples.
    legitimate = group[group["label"] == 0]
    phishing = group[group["label"] == 1]

    # Count legitimate URLs incorrectly flagged as phishing.
    false_positives = int(
        (legitimate["prediction"] == 1).sum()
    )

    # Count phishing URLs incorrectly classified as legitimate.
    false_negatives = int(
        (phishing["prediction"] == 0).sum()
    )

    # Calculate category-specific error rates.
    fpr = (
        false_positives / len(legitimate)
        if len(legitimate) else 0
    )

    fnr = (
        false_negatives / len(phishing)
        if len(phishing) else 0
    )

    return {
        "total_urls": len(group),
        "legitimate_urls": len(legitimate),
        "phishing_urls": len(phishing),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


# Store all category-level results.
results = []

# Analyze errors for each URL-length category.
for category, group in analysis.groupby(
    "length_category", observed=True
):
    metrics = calculate_error_rates(group)
    metrics["category_type"] = "URL length"
    metrics["category"] = str(category)
    results.append(metrics)

# Analyze HTTP and HTTPS URLs separately.
for is_https, group in analysis.groupby("is_https"):
    metrics = calculate_error_rates(group)
    metrics["category_type"] = "HTTPS"
    metrics["category"] = (
        "HTTPS" if is_https == 1 else "HTTP"
    )
    results.append(metrics)

# Convert the results into a table.
results_df = pd.DataFrame(results)

# Display category-level false-positive and false-negative rates.
print("\nERROR RATES BY URL CATEGORY")

for _, row in results_df.iterrows():
    print(
        f"{row['category_type']} - {row['category']}: "
        f"Legitimate={row['legitimate_urls']:,}, "
        f"Phishing={row['phishing_urls']:,}, "
        f"FPR={row['false_positive_rate']:.4f}, "
        f"FNR={row['false_negative_rate']:.4f}"
    )

# Save the analysis for your project report.
output_dir = ROOT / "results" / "metrics"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "validation_error_rates_by_category.csv"

results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
