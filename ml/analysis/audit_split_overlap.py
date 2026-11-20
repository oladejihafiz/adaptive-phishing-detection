
"""
Audit train, validation, and test split integrity.

This script is READ-ONLY:
- It does not modify processed datasets.
- It does not retrain or overwrite any model.
- It checks row counts and label alignment.
- It measures exact URL and registered-domain overlap.

Run from the project root:
    python -m ml.audit_split_overlap
"""

from pathlib import Path
from itertools import combinations
from urllib.parse import urlsplit
import ipaddress
import re

import pandas as pd
import tldextract


# ---------------------------------------------------------
# 1. Define the locations of the processed data files.
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

SPLITS = {
    "train": {
        "features": PROCESSED_DIR / "train.csv",
        "urls": PROCESSED_DIR / "train_urls.csv",
    },
    "validation": {
        "features": PROCESSED_DIR / "validation.csv",
        "urls": PROCESSED_DIR / "validation_urls.csv",
    },
    "test": {
        "features": PROCESSED_DIR / "test.csv",
        "urls": PROCESSED_DIR / "test_urls.csv",
    },
}

# Disable automatic downloads of the public suffix list.
# This uses the suffix data available locally to tldextract.
extract_domain = tldextract.TLDExtract(suffix_list_urls=())


# ---------------------------------------------------------
# 2. Validate hostnames before extracting registered domains.
# ---------------------------------------------------------

def get_registered_domain(raw_url):
    """
    Return a validated registered domain, or None if the URL
    does not contain a structurally valid hostname.

    IP addresses and single-label hosts are not counted as
    registered domains because they have no registrable domain.
    """

    # Ignore missing URLs and URLs containing whitespace.
    if pd.isna(raw_url):
        return None

    url = str(raw_url).strip()

    if not url or any(character.isspace() for character in url):
        return None

    # Add a scheme only when the URL does not provide one.
    if "://" not in url:
        url = "http://" + url

    try:
        # Parse the URL and reject unsupported schemes.
        parsed = urlsplit(url)

        if parsed.scheme.lower() not in {"http", "https"}:
            return None

        # Accessing port checks whether the port is malformed.
        _ = parsed.port

        hostname = parsed.hostname

        if not hostname:
            return None

        # Valid IP addresses are URLs, but not registered domains.
        try:
            ipaddress.ip_address(hostname)
            return None
        except ValueError:
            pass

        # Normalize a trailing root dot and internationalized names.
        hostname = hostname.rstrip(".")
        hostname = hostname.encode("idna").decode("ascii").lower()

        # Reject empty or excessively long hostnames.
        if not hostname or len(hostname) > 253:
            return None

        # Validate the syntax and length of every hostname label.
        label_pattern = re.compile(
            r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
        )

        labels = hostname.split(".")

        if any(
            not label or not label_pattern.fullmatch(label)
            for label in labels
        ):
            return None

        # A single-label hostname is not a registered public domain.
        if len(labels) < 2:
            return None

        # Extract the registered domain using the public suffix rules.
        result = extract_domain(hostname)

        if not result.domain or not result.suffix:
            return None

        return f"{result.domain}.{result.suffix}"

    except (ValueError, UnicodeError):
        # Invalid URLs are excluded from domain-overlap calculations.
        return None


# ---------------------------------------------------------
# 3. Load the feature and URL files for every split.
# ---------------------------------------------------------

loaded_splits = {}
audit_errors = []

for split_name, paths in SPLITS.items():
    print("\n" + "=" * 75)
    print(f"LOADING SPLIT: {split_name.upper()}")
    print("=" * 75)

    # Stop processing a split if either required file is missing.
    missing_files = [
        str(path) for path in paths.values() if not path.exists()
    ]

    if missing_files:
        for missing_file in missing_files:
            print(f"FILE NOT FOUND: {missing_file}")
            audit_errors.append(f"Missing file: {missing_file}")
        continue

    # Load both files without changing them.
    features = pd.read_csv(paths["features"])
    urls = pd.read_csv(paths["urls"])

    # Check that the expected columns exist.
    if "label" not in features.columns:
        audit_errors.append(f"{split_name}: features file has no label column")
        print("ERROR: Features file has no 'label' column.")
        continue

    if not {"url", "label"}.issubset(urls.columns):
        audit_errors.append(f"{split_name}: URL file lacks url or label")
        print("ERROR: URL file must contain 'url' and 'label'.")
        continue

    # Report basic split statistics.
    print(f"Feature rows: {len(features):,}")
    print(f"URL rows:     {len(urls):,}")
    print(f"Feature label counts: {features['label'].value_counts().to_dict()}")
    print(f"URL label counts:     {urls['label'].value_counts().to_dict()}")

    # Check row counts before comparing labels row by row.
    row_counts_match = len(features) == len(urls)
    print(f"Feature/URL row counts match: {row_counts_match}")

    labels_match = False

    if row_counts_match:
        # Reset indexes so the comparison uses row order only.
        feature_labels = features["label"].reset_index(drop=True)
        url_labels = urls["label"].reset_index(drop=True)

        # Convert to numeric labels; invalid values become missing.
        feature_numeric = pd.to_numeric(feature_labels, errors="coerce")
        url_numeric = pd.to_numeric(url_labels, errors="coerce")

        labels_match = (
            feature_numeric.notna().all()
            and url_numeric.notna().all()
            and feature_numeric.equals(url_numeric)
        )

    print(f"Feature/URL row-by-row labels match: {labels_match}")

    if not row_counts_match or not labels_match:
        audit_errors.append(
            f"{split_name}: row counts or row-by-row labels do not match"
        )

    # Keep only the information required for the overlap analysis.
    loaded_splits[split_name] = {
        "features": features,
        "urls": urls,
        "url_values": urls["url"].fillna("").astype(str).str.strip(),
    }


# ---------------------------------------------------------
# 4. Stop overlap analysis if any required split failed to load.
# ---------------------------------------------------------

if len(loaded_splits) != len(SPLITS):
    print("\nERROR: Not all splits loaded successfully.")
    print("Fix the file or column issues before interpreting overlap.")
    raise SystemExit(1)


# ---------------------------------------------------------
# 5. Measure exact URL duplicates between different splits.
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("EXACT URL OVERLAP")
print("=" * 75)

for first_name, second_name in combinations(loaded_splits, 2):
    first_urls = loaded_splits[first_name]["url_values"]
    second_urls = loaded_splits[second_name]["url_values"]

    # Ignore empty strings when counting duplicate URLs.
    first_set = set(first_urls[first_urls != ""])
    second_set = set(second_urls[second_urls != ""])

    shared_urls = first_set & second_set

    print(f"\n{first_name} vs {second_name}")
    print(f"Unique URLs in {first_name}: {len(first_set):,}")
    print(f"Unique URLs in {second_name}: {len(second_set):,}")
    print(f"Shared exact URL strings: {len(shared_urls):,}")

    if shared_urls:
        print("Examples:")
        for url in sorted(shared_urls)[:10]:
            print(f"  {url}")


# ---------------------------------------------------------
# 6. Extract registered domains and measure domain overlap.
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("REGISTERED-DOMAIN OVERLAP")
print("=" * 75)

domain_sets = {}

for split_name, split_data in loaded_splits.items():
    # Extract each URL's registered domain.
    domains = split_data["url_values"].map(get_registered_domain)

    # Keep only successfully extracted registered domains.
    valid_domains = domains.dropna()
    domain_sets[split_name] = set(valid_domains)

    print(f"\n{split_name.upper()}")
    print(f"URL rows: {len(domains):,}")
    print(f"URLs with extractable registered domains: {len(valid_domains):,}")
    print(f"Unique registered domains: {len(domain_sets[split_name]):,}")
    print(f"URLs excluded from domain analysis: {domains.isna().sum():,}")

for first_name, second_name in combinations(domain_sets, 2):
    first_domains = domain_sets[first_name]
    second_domains = domain_sets[second_name]

    shared_domains = first_domains & second_domains

    # Measure the proportion of each split's domains that overlap.
    first_overlap_pct = (
        100 * len(shared_domains) / len(first_domains)
        if first_domains else 0
    )
    second_overlap_pct = (
        100 * len(shared_domains) / len(second_domains)
        if second_domains else 0
    )

    print(f"\n{first_name} vs {second_name}")
    print(f"Shared registered domains: {len(shared_domains):,}")
    print(f"{first_name} domains also in {second_name}: {first_overlap_pct:.2f}%")
    print(f"{second_name} domains also in {first_name}: {second_overlap_pct:.2f}%")

    if shared_domains:
        print("Example shared domains:")
        for domain in sorted(shared_domains)[:15]:
            print(f"  {domain}")


# ---------------------------------------------------------
# 7. Print a final status without modifying any files.
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("AUDIT SUMMARY")
print("=" * 75)

if audit_errors:
    print("Issues requiring investigation:")
    for error in audit_errors:
        print(f"- {error}")
else:
    print("All split files loaded and feature/URL row-label checks passed.")

print("\nAudit complete. No datasets or model files were changed.")
