
"""
Evaluate the final grouped-test model separately for each dataset source.

IMPORTANT:
- This is a diagnostic evaluation only.
- It does not retrain or modify any model.
- It does not change the locked decision threshold.
- It uses the existing final test predictions.
- URLs belonging to multiple sources are counted in each applicable
  source group. Therefore, source totals may exceed the unique test total.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# ---------------------------------------------------------
# 1. PROJECT PATHS AND LOCKED CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT / "dataset" / "processed_v3_grouped" / "test_urls.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "metrics_v3_grouped"
    / "final_test_predictions_v3.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "metrics_v3_grouped"

# This must match the threshold used in the final test evaluation.
LOCKED_THRESHOLD = 0.747170

OUTPUT_CSV = OUTPUT_DIR / "source_performance_v3.csv"
OUTPUT_JSON = OUTPUT_DIR / "source_performance_v3.json"
OUTPUT_MEMBERSHIP_CSV = (
    OUTPUT_DIR / "source_membership_predictions_v3.csv"
)


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def parse_sources(source_value):
    """
    Convert a source field into a clean list of source names.

    The separator handling supports common formats such as:
    dataset_1;dataset_2
    dataset_1,dataset_2
    dataset_1|dataset_2
    """

    if pd.isna(source_value):
        return []

    # Split source names on common separators.
    sources = re.split(r"[;,|]+", str(source_value))

    # Remove whitespace and empty values.
    return sorted({
        source.strip()
        for source in sources
        if source.strip()
    })


def calculate_metrics(group, group_name):
    """
    Calculate classification metrics for one source group.

    The confusion matrix always uses:
        0 = legitimate
        1 = phishing

    ROC-AUC and PR-AUC are omitted when a group contains
    only one actual class because those metrics are undefined.
    """

    y_true = group["actual_label"].astype(int).to_numpy()
    y_pred = group["predicted_label"].astype(int).to_numpy()
    y_score = group["phishing_probability"].astype(float).to_numpy()

    # Explicitly preserve the legitimate/phishing matrix order.
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    legitimate_count = int(np.sum(y_true == 0))
    phishing_count = int(np.sum(y_true == 1))
    total_count = len(group)

    # Calculate FPR only when legitimate examples exist.
    fpr = (
        fp / legitimate_count
        if legitimate_count > 0
        else None
    )

    # Calculate FNR only when phishing examples exist.
    fnr = (
        fn / phishing_count
        if phishing_count > 0
        else None
    )

    # Ranking metrics require both actual classes.
    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
    else:
        roc_auc = None
        pr_auc = None

    result = {
        "source": group_name,
        "total_urls": int(total_count),
        "legitimate_urls": legitimate_count,
        "phishing_urls": phishing_count,

        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),

        "false_positive_rate": (
            float(fpr) if fpr is not None else None
        ),
        "false_negative_rate": (
            float(fnr) if fnr is not None else None
        ),

        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),

        "unique_registered_domains": int(
            group["registered_domain"].nunique()
        ) if "registered_domain" in group.columns else None,
    }

    return result


# ---------------------------------------------------------
# 3. LOAD THE EXISTING TEST METADATA AND PREDICTIONS
# ---------------------------------------------------------

print("=" * 72)
print("SOURCE-SPECIFIC MODEL EVALUATION")
print("=" * 72)

print("\nLoading test metadata...")
metadata = pd.read_csv(METADATA_PATH, low_memory=False)

print("Loading final test predictions...")
predictions = pd.read_csv(PREDICTIONS_PATH, low_memory=False)


# ---------------------------------------------------------
# 4. VALIDATE REQUIRED COLUMNS
# ---------------------------------------------------------

required_metadata = {
    "url",
    "label",
    "sources",
    "registered_domain",
}

required_predictions = {
    "url",
    "actual_label",
    "phishing_probability",
    "predicted_label",
    "decision_threshold",
}

missing_metadata = required_metadata - set(metadata.columns)
missing_predictions = required_predictions - set(predictions.columns)

if missing_metadata:
    raise ValueError(
        f"Missing metadata columns: {sorted(missing_metadata)}"
    )

if missing_predictions:
    raise ValueError(
        f"Missing prediction columns: {sorted(missing_predictions)}"
    )


# ---------------------------------------------------------
# 5. NORMALIZE TYPES AND VALIDATE THE LOCKED THRESHOLD
# ---------------------------------------------------------

metadata["label"] = pd.to_numeric(
    metadata["label"], errors="raise"
).astype(int)

predictions["actual_label"] = pd.to_numeric(
    predictions["actual_label"], errors="raise"
).astype(int)

predictions["predicted_label"] = pd.to_numeric(
    predictions["predicted_label"], errors="raise"
).astype(int)

predictions["phishing_probability"] = pd.to_numeric(
    predictions["phishing_probability"], errors="raise"
)

predictions["decision_threshold"] = pd.to_numeric(
    predictions["decision_threshold"], errors="raise"
)

if not set(metadata["label"].unique()).issubset({0, 1}):
    raise ValueError("Metadata contains labels outside 0 and 1.")

if not set(predictions["actual_label"].unique()).issubset({0, 1}):
    raise ValueError("Predictions contain invalid actual labels.")

if not set(predictions["predicted_label"].unique()).issubset({0, 1}):
    raise ValueError("Predictions contain invalid predicted labels.")

if not np.allclose(
    predictions["decision_threshold"].to_numpy(),
    LOCKED_THRESHOLD,
    atol=1e-6
):
    raise ValueError(
        "The prediction file does not use the locked threshold "
        f"{LOCKED_THRESHOLD}."
    )

if not predictions["phishing_probability"].between(0, 1).all():
    raise ValueError("Phishing scores must be between 0 and 1.")


# ---------------------------------------------------------
# 6. VERIFY URL AND LABEL ALIGNMENT
# ---------------------------------------------------------

# Each test metadata URL and prediction URL should appear once.
if metadata["url"].duplicated().any():
    raise ValueError("Duplicate URLs found in test metadata.")

if predictions["url"].duplicated().any():
    raise ValueError("Duplicate URLs found in prediction file.")

# Merge on URL, not row position.
merged = metadata.merge(
    predictions,
    left_on="url",
    right_on="url",
    how="outer",
    validate="one_to_one",
    indicator=True,
    suffixes=("_metadata", "_prediction"),
)

# Every metadata URL must have a matching prediction and vice versa.
if not (merged["_merge"] == "both").all():
    unmatched = merged.loc[
        merged["_merge"] != "both",
        ["url", "_merge"]
    ].head(10)

    raise ValueError(
        "Metadata and prediction URLs do not match completely.\n"
        f"Examples:\n{unmatched.to_string(index=False)}"
    )

# Confirm labels agree for every URL.
if not (
    merged["label"].astype(int).to_numpy()
    == merged["actual_label"].astype(int).to_numpy()
).all():
    raise ValueError(
        "Actual labels do not match between metadata and predictions."
    )

# Confirm that the saved binary predictions agree with the saved scores.
expected_predictions = (
    merged["phishing_probability"].to_numpy() >= LOCKED_THRESHOLD
).astype(int)

if not np.array_equal(
    expected_predictions,
    merged["predicted_label"].astype(int).to_numpy()
):
    raise ValueError(
        "Saved predictions do not agree with the locked threshold."
    )

print(f"\nMetadata records: {len(metadata):,}")
print(f"Prediction records: {len(predictions):,}")
print(f"Successfully matched URLs: {len(merged):,}")
print("URL alignment: PASSED")
print("Label alignment: PASSED")
print("Threshold verification: PASSED")


# ---------------------------------------------------------
# 7. PREPARE SOURCE MEMBERSHIP
# ---------------------------------------------------------

# Parse source names without changing the original metadata.
merged["source_list"] = merged["sources"].apply(parse_sources)

# Rows without source information cannot be assigned to a source.
missing_source_count = int(
    (merged["source_list"].str.len() == 0).sum()
)

if missing_source_count:
    print(
        f"\nWARNING: {missing_source_count:,} URLs have no source metadata."
    )

# Count how many sources contributed to each URL.
merged["source_count"] = merged["source_list"].str.len()

# Explode the source list. A URL shared by two datasets appears
# once in each of its two source groups.
membership = merged.explode("source_list").copy()

membership = membership.rename(
    columns={"source_list": "source"}
)

membership = membership[
    membership["source"].notna()
    & (membership["source"].astype(str).str.strip() != "")
].copy()

if membership.empty:
    raise ValueError("No usable source memberships were found.")

# Save the detailed source-membership dataset for auditing.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

membership.to_csv(
    OUTPUT_MEMBERSHIP_CSV,
    index=False
)


# ---------------------------------------------------------
# 8. CALCULATE OVERALL AND SOURCE-SPECIFIC METRICS
# ---------------------------------------------------------

overall_result = calculate_metrics(
    merged,
    "OVERALL_TEST_SET"
)

source_results = []

for source_name, source_group in membership.groupby(
    "source",
    sort=True
):
    result = calculate_metrics(
        source_group,
        source_name
    )

    # Count URLs also represented in other source datasets.
    result["urls_shared_with_other_sources"] = int(
        (source_group["source_count"] > 1).sum()
    )

    source_results.append(result)

source_results.sort(
    key=lambda item: item["total_urls"],
    reverse=True
)

# Build a JSON report with the overall test result and source results.
report = {
    "experiment": "Source-specific evaluation of grouped test set",
    "model": "random_forest_40_features",
    "feature_count": 40,
    "decision_threshold": LOCKED_THRESHOLD,

    "evaluation_rules": {
        "model_retrained": False,
        "threshold_changed": False,
        "test_set_modified": False,
        "multiple_source_membership_counted_in_each_source": True,
        "source_totals_may_overlap": True,
        "roc_auc_and_pr_auc_require_both_classes": True,
    },

    "test_set": {
        "unique_test_urls": int(len(merged)),
        "urls_without_source_metadata": missing_source_count,
        "source_membership_records": int(len(membership)),
        "sources_evaluated": int(len(source_results)),
    },

    "overall_test_metrics": overall_result,
    "source_metrics": source_results,
}

# Save the detailed JSON report.
with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4, allow_nan=False)

# Save a convenient source-level CSV.
source_df = pd.DataFrame(source_results)

source_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ---------------------------------------------------------
# 9. DISPLAY THE RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 72)
print("OVERALL TEST SET")
print("=" * 72)

print(f"URLs:       {overall_result['total_urls']:,}")
print(f"Accuracy:   {overall_result['accuracy']:.4f}")
print(f"Precision:  {overall_result['precision']:.4f}")
print(f"Recall:     {overall_result['recall']:.4f}")
print(f"F1-score:   {overall_result['f1_score']:.4f}")
print(
    "FPR:        "
    f"{overall_result['false_positive_rate']:.4f}"
)
print(
    "FNR:        "
    f"{overall_result['false_negative_rate']:.4f}"
)

print("\n" + "=" * 72)
print("SOURCE-SPECIFIC RESULTS")
print("=" * 72)

display_columns = [
    "source",
    "total_urls",
    "legitimate_urls",
    "phishing_urls",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "false_negative_rate",
    "roc_auc",
    "pr_auc",
]

print(
    source_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\n" + "=" * 72)
print("OUTPUT FILES")
print("=" * 72)

print(f"Source metrics CSV: {OUTPUT_CSV}")
print(f"Full report JSON:   {OUTPUT_JSON}")
print(f"Membership CSV:     {OUTPUT_MEMBERSHIP_CSV}")

print("\nSource-specific evaluation completed.")
print("No model retraining or threshold tuning was performed.")
