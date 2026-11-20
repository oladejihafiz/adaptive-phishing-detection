# Work with project files and directories.
from pathlib import Path

# Load saved machine-learning models.
import joblib

# Work with tabular data.
import pandas as pd

# Use the exact feature extraction function from model training.
from ml.prepare_data import features

# Locate the project's root directory.
BASE_DIR = Path(__file__).resolve().parent.parent

# Locate the independent case-study dataset.
DATA_FILE = BASE_DIR / "dataset" / "raw" / "dataset_5.csv"

# Locate the original trained models and fitted scaler.
STATIC_MODEL_FILE = BASE_DIR / "models" / "static_model.joblib"
ADAPTIVE_MODEL_FILE = BASE_DIR / "models" / "adaptive_model.joblib"
SCALER_FILE = BASE_DIR / "models" / "scaler.joblib"

# Locate the output directory.
RESULT_DIR = BASE_DIR / "results" / "metrics"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# These must match the 15 features used during training.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_at",
    "num_question",
    "num_equals",
    "num_slashes",
    "is_https",
    "has_ip_host",
    "num_host_parts",
    "has_punycode",
    "suspicious_words",
]




# Load the independent case-study dataset.
data = pd.read_csv(DATA_FILE)

# Confirm that every row is labelled phishing.
assert len(data) == 24, "Expected 24 case-study URLs."
assert (data["label_code"] == 1).all(), (
    "This case study expects phishing-only labels."
)


# Apply the exact feature extraction function used during training.
# This avoids inconsistencies between training and evaluation.
feature_frame = pd.DataFrame([
    features(str(url))
    for url in data["url"]
])

# Preserve the original feature order expected by the models.
feature_frame = feature_frame[FEATURE_NAMES]


# Load the original pretrained models.
static_model = joblib.load(STATIC_MODEL_FILE)
adaptive_model = joblib.load(ADAPTIVE_MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# Confirm the feature order matches the fitted scaler.
assert list(scaler.feature_names_in_) == FEATURE_NAMES, (
    "Feature order does not match the trained scaler."
)

# Predict using the original Random Forest model.
static_predictions = static_model.predict(feature_frame).astype(int)

# Scale the URL features for the Passive-Aggressive classifier.
scaled_features = scaler.transform(feature_frame)
adaptive_predictions = adaptive_model.predict(
    scaled_features
).astype(int)


# Calculate the adaptive classifier's decision scores.
# Negative scores favor class 0 (legitimate).
# Positive scores favor class 1 (phishing).
# These scores are not calibrated probabilities.
adaptive_scores = adaptive_model.decision_function(
    scaled_features
)

# Display the range of decision scores.
print("\nADAPTIVE MODEL DECISION SCORES")
print("Minimum score:", round(float(adaptive_scores.min()), 4))
print("Maximum score:", round(float(adaptive_scores.max()), 4))
print("Average score:", round(float(adaptive_scores.mean()), 4))

# Count how many scores are close to the decision boundary.
near_boundary = int(
    ((adaptive_scores >= -1.0) & (adaptive_scores <= 1.0)).sum()
)

print("Scores between -1 and +1:", near_boundary)

# Display each score without printing potentially malicious URLs.
print("\nINDIVIDUAL DECISION SCORES")
for index, score in enumerate(adaptive_scores, start=1):
    print(f"Sample {index}: {float(score):.4f}")


# Record both models' predictions.
results = pd.DataFrame({
    "url": data["url"],
    "actual_label": data["label_code"],
    "random_forest_prediction": static_predictions,
    "adaptive_prediction": adaptive_predictions,
})

# Add indicators for model agreement.
results["both_detected"] = (
    (static_predictions == 1) &
    (adaptive_predictions == 1)
)

results["both_missed"] = (
    (static_predictions == 0) &
    (adaptive_predictions == 0)
)

# Count the number of detected phishing URLs.
total = len(results)
static_detected = int((static_predictions == 1).sum())
adaptive_detected = int((adaptive_predictions == 1).sum())
both_detected = int(results["both_detected"].sum())
both_missed = int(results["both_missed"].sum())

# Calculate phishing detection rates.
static_recall = static_detected / total
adaptive_recall = adaptive_detected / total

# Save individual predictions for further analysis.
output_file = RESULT_DIR / "frsc_case_study_predictions.csv"
results.to_csv(output_file, index=False)

# Display the case-study results.
print("\nFRSC PHISHING CASE STUDY")
print("Total phishing URLs:", total)

print("\nRANDOM FOREST")
print("Detected:", static_detected)
print("Missed:", total - static_detected)
print("Detection rate:", round(static_recall * 100, 2), "%")

print("\nPASSIVE-AGGRESSIVE")
print("Detected:", adaptive_detected)
print("Missed:", total - adaptive_detected)
print("Detection rate:", round(adaptive_recall * 100, 2), "%")

print("\nMODEL AGREEMENT")
print("Both detected:", both_detected)
print("Both missed:", both_missed)
print(
    "Models disagreed:",
    int((static_predictions != adaptive_predictions).sum())
)

print("\nResults saved to:", output_file)
