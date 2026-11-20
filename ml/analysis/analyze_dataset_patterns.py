
# Import libraries for loading and analyzing the dataset.
from pathlib import Path
import pandas as pd

# Locate the project root directory.
ROOT = Path(__file__).resolve().parents[1]

# Load the validation dataset.
# We use validation data rather than the held-out test dataset
# because this analysis will help guide model improvements.
data = pd.read_csv(
    ROOT / "dataset" / "processed" / "validation.csv"
)

# Separate legitimate and phishing examples.
legitimate = data[data["label"] == 0]
phishing = data[data["label"] == 1]

print("\nDATASET PATTERN ANALYSIS")
print("Total URLs:", len(data))
print("Legitimate URLs:", len(legitimate))
print("Phishing URLs:", len(phishing))

# Calculate the proportion of HTTPS URLs in each class.
print("\nHTTPS DISTRIBUTION")

print(
    "Legitimate HTTPS:",
    round(legitimate["is_https"].mean() * 100, 2),
    "%"
)

print(
    "Phishing HTTPS:",
    round(phishing["is_https"].mean() * 100, 2),
    "%"
)

# Compare average URL feature values across both classes.
print("\nAVERAGE FEATURE VALUES BY CLASS")

columns = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_digits",
    "num_dots",
    "num_hyphens",
    "suspicious_words",
]

comparison = pd.DataFrame({
    "legitimate_mean": legitimate[columns].mean(),
    "phishing_mean": phishing[columns].mean(),
})

print(comparison.round(3).to_string())

# Compare URL lengths across legitimate and phishing examples.
print("\nURL LENGTH DISTRIBUTION")

# Each interval represents a distinct URL-length category.
ranges = [
    (0, 20),
    (21, 30),
    (31, 50),
    (51, 100),
    (101, float("inf")),
]

for lower, upper in ranges:

    # Count legitimate URLs within this length range.
    legit_count = int(
        legitimate["url_length"].between(
            lower, upper, inclusive="both"
        ).sum()
    )

    # Count phishing URLs within this length range.
    phishing_count = int(
        phishing["url_length"].between(
            lower, upper, inclusive="both"
        ).sum()
    )

    print(
        f"Length {lower}-{upper}: "
        f"Legitimate={legit_count:,}, "
        f"Phishing={phishing_count:,}"
    )

# Save the class-level feature comparison for documentation.
output_dir = ROOT / "results" / "metrics"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "dataset_pattern_comparison.csv"

comparison.to_csv(output_file)

print(f"\nResults saved to: {output_file}")
