
"""
FINAL TEST ERROR ANALYSIS - VERSION 3

Purpose:
    Analyse false positives and false negatives from the locked,
    domain-disjoint test evaluation.

Important:
    - Does NOT train or modify any model.
    - Does NOT change the selected threshold.
    - Does NOT modify the original datasets.
    - Uses the saved final test predictions.
    - Saves diagnostic CSV and JSON reports.

Selected model:
    40-feature Random Forest

Locked threshold:
    0.747170
"""

from pathlib import Path
from urllib.parse import urlsplit
from collections import Counter
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "dataset" / "processed_v3_grouped"
MODEL_DIR = PROJECT_ROOT / "models_v3_grouped"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics_v3_grouped"

OUTPUT_DIR = METRICS_DIR / "error_analysis_v3"

TEST_FEATURES_FILE = DATA_DIR / "test.csv"
TEST_METADATA_FILE = DATA_DIR / "test_urls.csv"

PREDICTIONS_FILE = METRICS_DIR / "final_test_predictions_v3.csv"

FEATURE_NAMES_FILE = (
    MODEL_DIR / "random_forest_40_features_features.joblib"
)

# This is the threshold selected before the independent test evaluation.
LOCKED_THRESHOLD = 0.747170

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def find_column(dataframe, candidates, description):
    """
    Find a column using a list of possible column names.

    This allows the script to work with minor differences in
    column naming between saved CSV files.
    """
    column_lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in column_lookup:
            return column_lookup[key]

    raise KeyError(
        f"Could not identify the {description} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(dataframe.columns)}"
    )


def normalize_label(value):
    """
    Convert common binary label representations into integers.

    Project convention:
        1 = phishing
        0 = legitimate
    """
    if pd.isna(value):
        raise ValueError("A missing label was found.")

    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)

    if isinstance(value, (float, np.floating)):
        if float(value) in (0.0, 1.0):
            return int(value)

    text = str(value).strip().lower()

    phishing_values = {
        "1", "1.0", "phishing", "malicious", "bad", "true"
    }

    legitimate_values = {
        "0", "0.0", "legitimate", "benign", "good", "false"
    }

    if text in phishing_values:
        return 1

    if text in legitimate_values:
        return 0

    raise ValueError(f"Unrecognized binary label: {value}")


def parse_sources(value):
    """
    Convert a source field into a list of source names.

    Supports semicolon-separated, comma-separated, and
    pipe-separated source names.
    """
    if pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    parts = re.split(r"[;,|]", text)

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def safe_mean(values):
    """
    Calculate a mean while safely ignoring missing values.
    """
    numeric = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).dropna()

    if numeric.empty:
        return None

    return float(numeric.mean())


def safe_rate(numerator, denominator):
    """
    Calculate a rate without dividing by zero.
    """
    if denominator == 0:
        return None

    return float(numerator / denominator)


def get_url_patterns(url):
    """
    Extract simple URL-pattern information for error analysis.

    These values are diagnostic only. They are not used to
    train or modify the Random Forest model.
    """
    url = str(url)

    try:
        parsed = urlsplit(url)

        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""

        try:
            ip_host = int(
                bool(re.fullmatch(
                    r"\d{1,3}(?:\.\d{1,3}){3}",
                    hostname
                ))
            )
        except Exception:
            ip_host = 0

        return {
            "pattern_url_length": len(url),
            "pattern_hostname_length": len(hostname),
            "pattern_path_length": len(path),
            "pattern_query_length": len(query),
            "pattern_subdomain_count": max(
                hostname.count(".") - 1, 0
            ),
            "pattern_dot_count": url.count("."),
            "pattern_hyphen_count": url.count("-"),
            "pattern_digit_count": sum(
                character.isdigit() for character in url
            ),
            "pattern_slash_count": url.count("/"),
            "pattern_at_symbol": int("@" in url),
            "pattern_ip_host": ip_host,
            "pattern_https": int(parsed.scheme.lower() == "https"),
            "pattern_has_query": int(bool(query)),
            "pattern_has_fragment": int(bool(parsed.fragment)),
        }

    except Exception:
        return {
            "pattern_url_length": len(url),
            "pattern_hostname_length": 0,
            "pattern_path_length": 0,
            "pattern_query_length": 0,
            "pattern_subdomain_count": 0,
            "pattern_dot_count": url.count("."),
            "pattern_hyphen_count": url.count("-"),
            "pattern_digit_count": sum(
                character.isdigit() for character in url
            ),
            "pattern_slash_count": url.count("/"),
            "pattern_at_symbol": int("@" in url),
            "pattern_ip_host": 0,
            "pattern_https": 0,
            "pattern_has_query": 0,
            "pattern_has_fragment": 0,
        }


def classification_result(actual, predicted):
    """
    Identify the four possible classification outcomes.
    """
    if actual == 0 and predicted == 0:
        return "true_negative"

    if actual == 0 and predicted == 1:
        return "false_positive"

    if actual == 1 and predicted == 0:
        return "false_negative"

    return "true_positive"


def get_error_type(result):
    """
    Return whether a classification is an error.
    """
    if result in ("false_positive", "false_negative"):
        return result

    return None


# ---------------------------------------------------------
# 3. LOAD INPUT FILES
# ---------------------------------------------------------

print("=" * 72)
print("FINAL TEST ERROR ANALYSIS")
print("=" * 72)

print("\nLoading test features...")
test_features = pd.read_csv(TEST_FEATURES_FILE)

print("Loading test metadata...")
test_metadata = pd.read_csv(TEST_METADATA_FILE)

print("Loading saved predictions...")
predictions = pd.read_csv(PREDICTIONS_FILE)

print("Loading feature names...")
feature_names = joblib.load(FEATURE_NAMES_FILE)

feature_names = list(feature_names)


# ---------------------------------------------------------
# 4. IDENTIFY REQUIRED COLUMNS
# ---------------------------------------------------------

# Find the actual URL column in the metadata file.
metadata_url_column = find_column(
    test_metadata,
    ["url", "URL", "original_url"],
    "URL metadata"
)

# Find the label column in the test feature file.
feature_label_column = find_column(
    test_features,
    ["label", "Label", "target", "class"],
    "test label"
)

# Find the URL column in the saved prediction file.
prediction_url_column = find_column(
    predictions,
    ["url", "URL", "original_url"],
    "prediction URL"
)

# Find the actual label in the prediction file.
prediction_actual_column = find_column(
    predictions,
    [
        "y_true",
        "true_label",
        "actual_label",
        "actual",
        "label",
        "ground_truth",
    ],
    "prediction actual label"
)

# Find the predicted label in the prediction file.
prediction_result_column = find_column(
    predictions,
    [
        "y_pred",
        "predicted_label",
        "prediction",
        "predicted",
        "pred",
    ],
    "predicted label"
)

# Find the phishing score/probability column.
prediction_score_column = find_column(
    predictions,
    [
        "phishing_probability",
        "phishing_score",
        "probability",
        "score",
        "y_score",
        "positive_probability",
    ],
    "phishing probability"
)

# Identify source information if available.
source_column = find_column(
    test_metadata,
    ["sources", "source", "dataset_source"],
    "source metadata"
)

# Identify registered domain information if available.
domain_column = find_column(
    test_metadata,
    [
        "registered_domain",
        "domain",
        "root_domain",
        "registered domain",
    ],
    "registered domain"
)


# ---------------------------------------------------------
# 5. VALIDATE DATA ALIGNMENT
# ---------------------------------------------------------

print("\nValidated test records:", len(test_features))

if len(test_features) != len(test_metadata):
    raise ValueError(
        "Test features and test metadata have different row counts."
    )

if len(test_features) != len(predictions):
    raise ValueError(
        "Test features and saved predictions have different row counts."
    )

# Check URL alignment.
feature_urls = (
    test_metadata[metadata_url_column]
    .fillna("")
    .astype(str)
    .reset_index(drop=True)
)

prediction_urls = (
    predictions[prediction_url_column]
    .fillna("")
    .astype(str)
    .reset_index(drop=True)
)

if not feature_urls.equals(prediction_urls):
    raise ValueError(
        "URL alignment failed. The test metadata and saved predictions "
        "are not in the same order."
    )

print("URL alignment: PASSED")

# Check label alignment.
actual_labels = test_features[feature_label_column].apply(
    normalize_label
).reset_index(drop=True)

saved_actual_labels = predictions[
    prediction_actual_column
].apply(normalize_label).reset_index(drop=True)

if not actual_labels.equals(saved_actual_labels):
    raise ValueError(
        "Label alignment failed between the test file and predictions."
    )

print("Label alignment: PASSED")

# Check that all expected model features are available.
missing_features = [
    feature
    for feature in feature_names
    if feature not in test_features.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

print("Feature alignment: PASSED")

# Extract predictions and scores.
predicted_labels = predictions[
    prediction_result_column
].apply(normalize_label).reset_index(drop=True)

phishing_scores = pd.to_numeric(
    predictions[prediction_score_column],
    errors="coerce"
).reset_index(drop=True)

if phishing_scores.isna().any():
    raise ValueError(
        "Missing or invalid phishing probability values were found."
    )

if not phishing_scores.between(0, 1).all():
    raise ValueError(
        "Phishing scores must be between 0 and 1."
    )

# Verify the locked threshold against the saved predictions.
reconstructed_predictions = (
    phishing_scores >= LOCKED_THRESHOLD
).astype(int)

if not np.array_equal(
    reconstructed_predictions.to_numpy(),
    predicted_labels.to_numpy()
):
    raise ValueError(
        "Threshold verification failed. The saved predictions do not "
        "match the locked threshold of 0.747170."
    )

print("Threshold verification: PASSED")


# ---------------------------------------------------------
# 6. BUILD THE MAIN ANALYSIS DATAFRAME
# ---------------------------------------------------------

analysis = test_metadata.copy().reset_index(drop=True)

# Add the actual and predicted labels.
analysis["actual_label"] = actual_labels
analysis["predicted_label"] = predicted_labels
analysis["phishing_probability"] = phishing_scores

# Add the classification result.
analysis["result"] = [
    classification_result(actual, predicted)
    for actual, predicted in zip(
        actual_labels,
        predicted_labels
    )
]

# Add the error type.
analysis["error_type"] = analysis["result"].apply(
    get_error_type
)

# Add the locked threshold for traceability.
analysis["threshold"] = LOCKED_THRESHOLD

# Add the model's 40 feature values.
# Prefixing is unnecessary here because these are the actual model features.
for feature in feature_names:
    analysis[feature] = test_features[feature].reset_index(drop=True)


# ---------------------------------------------------------
# 7. CLASSIFICATION SUMMARY AND ERROR FILES
# ---------------------------------------------------------

print("\nClassification outcomes:")
print(analysis["result"].value_counts())

errors = analysis[
    analysis["error_type"].notna()
].copy()

false_positives = errors[
    errors["error_type"] == "false_positive"
].copy()

false_negatives = errors[
    errors["error_type"] == "false_negative"
].copy()

print("\nTotal test errors:", len(errors))
print("False positives:", len(false_positives))
print("False negatives:", len(false_negatives))

# Save all test errors.
errors.to_csv(
    OUTPUT_DIR / "all_test_errors.csv",
    index=False
)

false_positives.to_csv(
    OUTPUT_DIR / "false_positives.csv",
    index=False
)

false_negatives.to_csv(
    OUTPUT_DIR / "false_negatives.csv",
    index=False
)


# ---------------------------------------------------------
# 8. SOURCE-SPECIFIC ERROR COUNTS
# ---------------------------------------------------------

# Parse sources directly from the metadata.
# No source_list column is required anywhere in this script.

source_error_counts = Counter()

for _, row in errors.iterrows():
    source_names = parse_sources(row[source_column])

    for source_name in source_names:
        source_error_counts[
            (source_name, row["error_type"])
        ] += 1

source_rows = []

for (source_name, error_type), count in source_error_counts.items():
    source_rows.append({
        "source": source_name,
        "error_type": error_type,
        "error_count": int(count),
    })

source_error_df = pd.DataFrame(source_rows)

if not source_error_df.empty:
    source_error_df = source_error_df.sort_values(
        ["error_type", "error_count"],
        ascending=[True, False]
    )

source_error_df.to_csv(
    OUTPUT_DIR / "source_error_counts.csv",
    index=False
)


# ---------------------------------------------------------
# 9. TOP REGISTERED DOMAINS ASSOCIATED WITH ERRORS
# ---------------------------------------------------------

domain_rows = []

# Group by error type and registered domain.
for (error_type, domain), group in errors.groupby(
    ["error_type", domain_column],
    dropna=False
):
    # Parse sources directly from each row.
    # This avoids the previous KeyError involving source_list.
    associated_sources = set()

    for source_value in group[source_column]:
        for source_name in parse_sources(source_value):
            associated_sources.add(source_name)

    domain_rows.append({
        "error_type": error_type,
        "registered_domain": (
            str(domain) if pd.notna(domain) else "UNKNOWN"
        ),
        "error_count": int(len(group)),
        "sources": ";".join(sorted(associated_sources)),
        "example_urls": " || ".join(
            group[metadata_url_column]
            .astype(str)
            .head(3)
            .tolist()
        ),
    })

domain_df = pd.DataFrame(domain_rows)

if not domain_df.empty:
    domain_df = domain_df.sort_values(
        ["error_type", "error_count"],
        ascending=[True, False]
    )

domain_df.to_csv(
    OUTPUT_DIR / "errors_by_registered_domain.csv",
    index=False
)


# ---------------------------------------------------------
# 10. URL PATTERN COMPARISON
# ---------------------------------------------------------

pattern_records = []

for url in analysis[metadata_url_column]:
    pattern_records.append(get_url_patterns(url))

# Prefixes prevent collisions with the model's existing feature names.
pattern_df = pd.DataFrame(pattern_records)

analysis_with_patterns = pd.concat(
    [
        analysis.reset_index(drop=True),
        pattern_df.reset_index(drop=True),
    ],
    axis=1
)

pattern_summary_rows = []

pattern_columns = list(pattern_df.columns)

for result_name, group in analysis_with_patterns.groupby("result"):
    row = {
        "result": result_name,
        "record_count": int(len(group)),
    }

    for column in pattern_columns:
        row[f"mean_{column}"] = safe_mean(group[column])

    pattern_summary_rows.append(row)

pattern_summary = pd.DataFrame(pattern_summary_rows)

pattern_summary.to_csv(
    OUTPUT_DIR / "url_pattern_comparison.csv",
    index=False
)


# ---------------------------------------------------------
# 11. FEATURE COMPARISON BY CLASSIFICATION RESULT
# ---------------------------------------------------------

feature_result_rows = []

for result_name, group in analysis.groupby("result"):
    row = {
        "result": result_name,
        "record_count": int(len(group)),
    }

    for feature in feature_names:
        row[f"mean_{feature}"] = safe_mean(group[feature])

    feature_result_rows.append(row)

feature_result_df = pd.DataFrame(feature_result_rows)

feature_result_df.to_csv(
    OUTPUT_DIR / "feature_comparison_by_result.csv",
    index=False
)


# ---------------------------------------------------------
# 12. FEATURE COMPARISON BY ERROR CLASS
# ---------------------------------------------------------

feature_error_rows = []

for error_type, group in errors.groupby("error_type"):
    row = {
        "error_type": error_type,
        "record_count": int(len(group)),
    }

    for feature in feature_names:
        row[f"mean_{feature}"] = safe_mean(group[feature])

    feature_error_rows.append(row)

feature_error_df = pd.DataFrame(feature_error_rows)

feature_error_df.to_csv(
    OUTPUT_DIR / "feature_error_comparison_by_class.csv",
    index=False
)


# ---------------------------------------------------------
# 13. HIGHEST-SCORE FALSE POSITIVES AND
#     LOWEST-SCORE FALSE NEGATIVES
# ---------------------------------------------------------

# False positives are legitimate URLs incorrectly classified as phishing.
highest_score_fps = false_positives.sort_values(
    "phishing_probability",
    ascending=False
).head(100)

highest_score_fps.to_csv(
    OUTPUT_DIR / "top_100_highest_score_false_positives.csv",
    index=False
)

# False negatives are phishing URLs incorrectly classified as legitimate.
lowest_score_fns = false_negatives.sort_values(
    "phishing_probability",
    ascending=True
).head(100)

lowest_score_fns.to_csv(
    OUTPUT_DIR / "top_100_lowest_score_false_negatives.csv",
    index=False
)


# ---------------------------------------------------------
# 14. SAVE FINAL ERROR ANALYSIS SUMMARY
# ---------------------------------------------------------

true_negatives = int(
    (analysis["result"] == "true_negative").sum()
)

true_positives = int(
    (analysis["result"] == "true_positive").sum()
)

false_positive_count = int(len(false_positives))
false_negative_count = int(len(false_negatives))

legitimate_count = int((actual_labels == 0).sum())
phishing_count = int((actual_labels == 1).sum())

summary = {
    "model": "40-feature Random Forest",
    "threshold": LOCKED_THRESHOLD,
    "test_records": int(len(analysis)),
    "total_errors": int(len(errors)),
    "true_negatives": true_negatives,
    "true_positives": true_positives,
    "false_positives": false_positive_count,
    "false_negatives": false_negative_count,
    "legitimate_test_records": legitimate_count,
    "phishing_test_records": phishing_count,
    "false_positive_rate": safe_rate(
        false_positive_count,
        legitimate_count
    ),
    "false_negative_rate": safe_rate(
        false_negative_count,
        phishing_count
    ),
    "error_rate": safe_rate(
        len(errors),
        len(analysis)
    ),
    "source_error_rows": int(len(source_error_df)),
    "domain_error_groups": int(len(domain_df)),
    "feature_count": int(len(feature_names)),
    "outputs_directory": str(OUTPUT_DIR),
}

with open(
    OUTPUT_DIR / "error_analysis_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=4,
        allow_nan=False
    )


# ---------------------------------------------------------
# 15. FINAL REPORT
# ---------------------------------------------------------

print("\n" + "=" * 72)
print("ERROR ANALYSIS COMPLETED")
print("=" * 72)

print(f"Test records:       {len(analysis):,}")
print(f"True negatives:     {true_negatives:,}")
print(f"True positives:     {true_positives:,}")
print(f"False positives:    {false_positive_count:,}")
print(f"False negatives:    {false_negative_count:,}")

print(
    "False positive rate:",
    f"{summary['false_positive_rate']:.4%}"
)

print(
    "False negative rate:",
    f"{summary['false_negative_rate']:.4%}"
)

print("\nReports saved to:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for output_file in sorted(OUTPUT_DIR.glob("*")):
    if output_file.is_file():
        print(" -", output_file.name)

print("\nNo model training or threshold changes were performed.")
print("=" * 72)
