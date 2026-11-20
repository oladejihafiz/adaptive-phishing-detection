
# Import the libraries needed to load the model and process URLs.
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd

# Import the exact feature extraction function used during training.
from ml.prepare_data import features

# Find the root directory of the project.
ROOT = Path(__file__).resolve().parents[1]

# Load the existing Random Forest model.
# This script will NOT modify or retrain the model.
model = joblib.load(ROOT / "models" / "static_model.joblib")

# Use the known legitimate examples for diagnosis.
# The ChatGPT URL is represented by its path structure,
# without including a private conversation identifier.
urls = [
    "https://example.com/",
    "https://chatgpt.com/c/00000000-0000-0000-0000-000000000000",
    "https://chatgpt.com/",
]

# Extract the same 15 features used during model training.
feature_data = pd.DataFrame([features(url) for url in urls])

# Calculate phishing probability estimates.
phishing_index = list(model.classes_).index(1)
probabilities = model.predict_proba(feature_data)[:, phishing_index]

print("\nLEGITIMATE URL DIAGNOSTIC")

# Display each URL's prediction without opening the website.
for url, probability in zip(urls, probabilities):
    print(f"\nHostname: {urlsplit(url).hostname}")
    print(f"URL: {url}")
    print(f"Phishing score: {probability:.4f}")
    print(f"Flagged at 0.50: {probability >= 0.50}")
    print(f"Flagged at 0.55: {probability >= 0.55}")
    print(f"Flagged at 0.60: {probability >= 0.60}")

# Display the numerical features used for each prediction.
print("\nFEATURE COMPARISON")
print(feature_data.to_string(index=False))

# Show the model's global feature importance values.
# These indicate which features the Random Forest generally
# relies on, not the exact cause of an individual prediction.
importance = pd.Series(
    model.feature_importances_,
    index=feature_data.columns
).sort_values(ascending=False)

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(importance.to_string())
