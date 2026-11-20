
"""Build a domain-disjoint, feature-engineered dataset for the phishing project.

This script:
1. Loads the audited master URL dataset.
2. Groups URLs by registered domain.
3. Creates domain-disjoint training, validation, and test partitions.
4. Extracts the existing 40 URL features once.
5. Saves the datasets separately without overwriting older experiments.

It does not visit URLs or train a model.
"""

from __future__ import annotations

import ipaddress
import json
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------
# 1. PROJECT PATHS AND IMPORT CONFIGURATION
# ---------------------------------------------------------

# Find the project root (the folder containing the ml directory).
ROOT = Path(__file__).resolve().parents[1]

# Add the project root to Python's import path.
# This fixes ModuleNotFoundError when running from PowerShell.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the existing feature extractors.
# These preserve the exact feature definitions from previous experiments.
from ml.prepare_data import features as original_features
from ml.train_static_improved import improved_features
from ml.enhanced_features import enhanced_features


# Input and output locations.
INPUT_FILE = (
    ROOT / "dataset" / "processed_v2_audited" / "master_urls_v2.csv"
)

OUTPUT_DIR = ROOT / "dataset" / "processed_v3_grouped"

# Seven folds approximate a 71/14/14 split.
N_SPLITS = 7

# Fixed random seed makes the experiment reproducible.
RANDOM_STATE = 42

# Process feature extraction in chunks to reduce memory usage.
CHUNK_SIZE = 10_000


# ---------------------------------------------------------
# 2. REGISTERED-DOMAIN EXTRACTION
# ---------------------------------------------------------

# Include private hosting suffixes such as blogspot.com and github.io.
# This prevents unrelated hosted websites from being grouped together.
try:
    DOMAIN_EXTRACTOR = tldextract.TLDExtract(
        suffix_list_urls=(),
        include_psl_private_domains=True,
    )
except TypeError as exc:
    raise RuntimeError(
        "Your tldextract version does not support "
        "include_psl_private_domains. Upgrade it using:\n"
        "python -m pip install --upgrade tldextract"
    ) from exc


@lru_cache(maxsize=250_000)
def registered_domain_from_url(url: str) -> str:
    """Return a registered-domain grouping key without visiting the URL."""

    parsed = urlsplit(str(url))
    hostname = (parsed.hostname or "").strip(".").lower()

    if not hostname:
        raise ValueError(f"URL has no hostname: {url!r}")

    # Treat IP addresses as their own groups.
    try:
        return ipaddress.ip_address(hostname).compressed
    except ValueError:
        pass

    # Extract the registered domain using the public suffix list.
    extracted = DOMAIN_EXTRACTOR(hostname)

    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()

    # Fall back to the complete hostname for unusual suffixes.
    return hostname


# ---------------------------------------------------------
# 3. VERIFY THE THREE FEATURE SETS
# ---------------------------------------------------------

def get_feature_names(sample_url: str) -> dict[str, list[str]]:
    """Verify the 15-, 29-, and 40-feature definitions."""

    feature_sets = {
        "15": list(original_features(sample_url).keys()),
        "29": list(improved_features(sample_url).keys()),
        "40": list(enhanced_features(sample_url).keys()),
    }

    # The original feature sets must be contained in the 40-feature set.
    all_40_features = set(feature_sets["40"])

    for size in ("15", "29"):
        missing = set(feature_sets[size]) - all_40_features

        if missing:
            raise ValueError(
                f"The {size}-feature set is not contained in "
                f"the 40-feature set. Missing: {sorted(missing)}"
            )

    # Confirm that the extractors return the expected number of features.
    expected_counts = {"15": 15, "29": 29, "40": 40}

    for size, expected in expected_counts.items():
        actual = len(feature_sets[size])

        if actual != expected:
            raise ValueError(
                f"Expected {expected} features, but found {actual} "
                f"in the {size}-feature extractor."
            )

    return feature_sets


# ---------------------------------------------------------
# 4. CREATE DOMAIN-DISJOINT DATA PARTITIONS
# ---------------------------------------------------------

def assign_domain_disjoint_splits(data: pd.DataFrame) -> pd.DataFrame:
    """Split data while keeping each registered domain in one partition."""

    print("Deriving registered-domain grouping keys...")

    data["registered_domain"] = [
        registered_domain_from_url(url)
        for url in data["url"].astype(str)
    ]

    unique_domains = data["registered_domain"].nunique()

    if unique_domains < N_SPLITS:
        raise ValueError(
            f"Only {unique_domains} registered domains are available. "
            f"At least {N_SPLITS} are required."
        )

    # Prepare labels and grouping information.
    labels = data["label"].to_numpy(dtype=np.int8)
    groups = data["registered_domain"].to_numpy()

    # StratifiedGroupKFold balances labels while keeping groups together.
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    folds = list(splitter.split(data["url"], labels, groups))

    # Start by assigning every record to training.
    split_names = np.full(len(data), "train", dtype=object)

    # Assign two separate folds to testing and validation.
    split_names[folds[0][1]] = "test"
    split_names[folds[1][1]] = "validation"

    data["split"] = split_names

    # Verify that no registered domain appears in multiple partitions.
    split_counts = (
        data.groupby("registered_domain")["split"].nunique()
    )

    if int(split_counts.max()) != 1:
        raise RuntimeError(
            "Domain leakage detected: a registered domain "
            "appears in multiple partitions."
        )

    # Each partition must contain both phishing and legitimate examples.
    for split_name in ("train", "validation", "test"):

        present_labels = set(
            data.loc[data["split"] == split_name, "label"].unique()
        )

        if present_labels != {0, 1}:
            raise RuntimeError(
                f"The {split_name} partition does not contain both "
                f"labels. Found: {present_labels}"
            )

    return data


# ---------------------------------------------------------
# 5. EXTRACT FEATURES AND SAVE EACH PARTITION
# ---------------------------------------------------------

def extract_and_save_split(
    data: pd.DataFrame,
    split_name: str,
    feature_names: list[str],
) -> dict[str, int]:
    """Extract 40 features in chunks and save the resulting files."""

    # Select the records belonging to this partition.
    subset = (
        data.loc[data["split"] == split_name]
        .reset_index(drop=True)
    )

    feature_path = OUTPUT_DIR / f"{split_name}.csv"
    urls_path = OUTPUT_DIR / f"{split_name}_urls.csv"

    # Save URLs, labels, provenance, and domain grouping separately.
    # These metadata columns must never become model features.
    subset.to_csv(urls_path, index=False)

    # Remove an existing feature file before writing a fresh experiment.
    if feature_path.exists():
        feature_path.unlink()

    print(
        f"Extracting 40 features for {split_name}: "
        f"{len(subset):,} URLs..."
    )

    # Extract features in manageable chunks.
    for start in range(0, len(subset), CHUNK_SIZE):

        chunk = subset.iloc[start:start + CHUNK_SIZE]

        # The enhanced extractor includes the original 15 and improved 29.
        records = [
            enhanced_features(url)
            for url in chunk["url"].astype(str)
        ]

        feature_frame = pd.DataFrame.from_records(
            records,
            columns=feature_names,
        )

        # Ensure that every feature value is numeric and finite.
        numeric_values = feature_frame.to_numpy(dtype=np.float64)

        if not np.isfinite(numeric_values).all():
            raise ValueError(
                f"Non-finite feature value found in {split_name}, "
                f"rows {start} to {start + len(chunk) - 1}."
            )

        # Append the target label after the feature columns.
        feature_frame["label"] = chunk["label"].to_numpy(dtype=np.int8)

        # Write the first chunk with headers and append subsequent chunks.
        feature_frame.to_csv(
            feature_path,
            mode="a",
            header=(start == 0),
            index=False,
        )

        completed = min(start + len(chunk), len(subset))

        print(
            f"  {split_name}: {completed:,}/"
            f"{len(subset):,} rows processed"
        )

    # Return partition statistics for the final summary.
    return {
        "rows": int(len(subset)),
        "domains": int(subset["registered_domain"].nunique()),
        "phishing": int((subset["label"] == 1).sum()),
        "legitimate": int((subset["label"] == 0).sum()),
    }


# ---------------------------------------------------------
# 6. MAIN WORKFLOW
# ---------------------------------------------------------

def main() -> None:
    """Run the complete domain-disjoint feature-engineering workflow."""

    # Confirm that the audited dataset exists.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Audited master file not found: {INPUT_FILE}\n"
            "Confirm that prepare_data_v2.py created master_urls_v2.csv."
        )

    # Create the new output directory without touching older experiments.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading audited data: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE, low_memory=False)

    # Confirm the required columns exist.
    required_columns = {"url", "label"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Master file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Remove rows without URLs or labels.
    data = data.dropna(subset=["url", "label"]).copy()

    data["url"] = data["url"].astype(str)
    data["label"] = pd.to_numeric(
        data["label"],
        errors="raise",
    ).astype(np.int8)

    # Confirm the project label convention.
    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Labels must use 1 for phishing and 0 for legitimate."
        )

    # The audited master file should contain one row per URL.
    if data["url"].duplicated().any():

        duplicate_count = int(data["url"].duplicated().sum())

        raise ValueError(
            f"Found {duplicate_count:,} duplicate URL rows. "
            "Resolve duplicates before splitting."
        )

    print(f"Audited URL rows: {len(data):,}")

    # Verify that all three existing feature extractors are compatible.
    feature_sets = get_feature_names(data.iloc[0]["url"])

    print("Feature definitions verified: 15, 29, and 40 columns.")

    # Assign all records to domain-disjoint partitions.
    data = assign_domain_disjoint_splits(data)

    # Save feature names in their original extractor order.
    with (OUTPUT_DIR / "feature_sets.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(feature_sets, file, indent=2)

    # Prepare the experiment summary.
    summary = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "rows": int(len(data)),
        "unique_registered_domains": int(
            data["registered_domain"].nunique()
        ),
        "split_method": "StratifiedGroupKFold",
        "folds": N_SPLITS,
        "random_state": RANDOM_STATE,
        "approximate_split_ratio": "5/7 train, 1/7 validation, 1/7 test",
        "domain_disjoint": True,
        "label_convention": {
            "1": "phishing",
            "0": "legitimate",
        },
        "partitions": {},
    }

    # Extract features one partition at a time.
    for split_name in ("train", "validation", "test"):

        stats = extract_and_save_split(
            data,
            split_name,
            feature_sets["40"],
        )

        summary["partitions"][split_name] = stats

    # Save the final split statistics.
    with (OUTPUT_DIR / "split_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    # Display a concise completion report.
    print("\nDomain-disjoint feature engineering complete.")
    print(f"Output folder: {OUTPUT_DIR}")

    print("\nGenerated files:")
    print("  train.csv, validation.csv, test.csv")
    print("  train_urls.csv, validation_urls.csv, test_urls.csv")
    print("  feature_sets.json")
    print("  split_summary.json")

    print("\nPartition summary:")

    for split_name, stats in summary["partitions"].items():

        print(
            f"  {split_name}: {stats['rows']:,} rows, "
            f"{stats['domains']:,} domains, "
            f"phishing={stats['phishing']:,}, "
            f"legitimate={stats['legitimate']:,}"
        )


# Run the workflow when this file is executed directly.
if __name__ == "__main__":
    main()
