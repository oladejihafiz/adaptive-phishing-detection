from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd

# Locate the project and load the model used by the backend.
root = Path.cwd()
model = joblib.load(root / "models" / "static_model_enhanced.joblib")
feature_order = list(
    joblib.load(root / "models" / "static_enhanced_features.joblib")
)

# Import the exact enhanced feature extractor used by the project.
from ml.enhanced_features import enhanced_features

# Test the same legitimate URLs as the earlier diagnostic.
urls = [
    "https://example.com/",
    "https://chatgpt.com/c/00000000-0000-0000-0000-000000000000",
    "https://chatgpt.com/",
]

# Extract features in the exact order expected by the model.
feature_data = pd.DataFrame(
    [enhanced_features(url) for url in urls]
)[feature_order]

# Find the probability column corresponding to the phishing label (1).
phishing_index = list(model.classes_).index(1)
probabilities = model.predict_proba(feature_data)[:, phishing_index]

# Display scores and predictions without visiting the URLs.
print("\nENHANCED RANDOM FOREST DIAGNOSTIC")
for url, probability in zip(urls, probabilities):
    print(f"\nHostname: {urlsplit(url).hostname}")
    print(f"URL: {url}")
    print(f"Phishing score: {probability:.4f}")
    print(f"Flagged at 0.50: {probability >= 0.50}")

# Show the extracted feature values for further diagnosis.
print("\nENHANCED FEATURES")
print(feature_data.to_string(index=False))