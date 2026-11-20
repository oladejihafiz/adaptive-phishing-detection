
"""
Audit raw phishing datasets for potential source and URL-feature bias.

This script is READ-ONLY:
- It does not modify raw or processed datasets.
- It does not train or overwrite any model.
- It reports source-level label, URL-scheme, homepage, and domain patterns.
- It compares registered domains across the raw data sources.
"""

from pathlib import Path
from urllib.parse import urlsplit
from collections import Counter
import ipaddress
import re

import pandas as pd
import tldextract


# Resolve the project root from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Point to the correct locations of the raw CSV datasets.
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"

DATASET_1_PATH = RAW_DATA_DIR / "dataset_1.csv"
DATASET_2_PATH = RAW_DATA_DIR / "dataset_2.csv"
DATASET_4_PATH = RAW_DATA_DIR / "dataset_4.csv"

# Configure domain extraction without downloading the public suffix list.
domain_extractor = tldextract.TLDExtract(suffix_list_urls=())

# Keep each dataset's expected column names and label mapping.
SOURCES = [
    {
        "file": "dataset_1.csv",
        "path": DATASET_1_PATH,
        "url_column": "URL",
        "label_column": "Label",
        "labels": {"1": 1, "0": 0},
    },
    {
        "file": "dataset_2.csv",
        "path": DATASET_2_PATH,
        "url_column": "url",
        "label_column": "type",
        "labels": {"phishing": 1, "legitimate": 0},
    },
    {
        "file": "dataset_4.csv",
        "path": DATASET_4_PATH,
        "url_column": "URL",
        "label_column": "Label",
        "labels": {"bad": 1, "good": 0},
    },
]



# Import these modules at the top of audit_source_bias.py if they are not
# already imported.
import ipaddress
import re
from urllib.parse import urlsplit


def normalize_url(raw_url):
    """
    Normalize and validate a URL for the source-bias audit.

    Returns a normalized URL string if the URL is usable.
    Returns None when the URL has an invalid scheme, hostname, or port.

    Note: This is a structural validation check, not proof that a URL
    exists, is reachable, or is safe.
    """

    # Reject missing values and convert other input types to strings.
    if raw_url is None:
        return None

    url = str(raw_url).strip()

    # Empty strings and URLs containing whitespace are not accepted.
    if not url or any(character.isspace() for character in url):
        return None

    # Preserve explicit schemes; otherwise assume HTTP for the audit.
    if "://" not in url:
        url = "http://" + url

    try:
        # Parse the URL into its scheme, hostname, port, and other parts.
        parsed = urlsplit(url)

        # Only HTTP and HTTPS URLs are included in this audit.
        if parsed.scheme.lower() not in {"http", "https"}:
            return None

        # Accessing .port also detects malformed or out-of-range ports.
        port = parsed.port

        # A web URL must contain a hostname.
        hostname = parsed.hostname
        if not hostname:
            return None

        # Accept valid IP addresses, including IPv6 addresses.
        try:
            ipaddress.ip_address(hostname)
            return parsed._replace(
                scheme=parsed.scheme.lower()
            ).geturl()
        except ValueError:
            # Not an IP address; validate it as a hostname below.
            pass

        # Remove a trailing DNS root dot before validating the hostname.
        hostname = hostname.rstrip(".")

        # Convert internationalized hostnames to ASCII using IDNA.
        # This preserves support for many legitimate non-ASCII domain names.
        try:
            ascii_hostname = hostname.encode("idna").decode("ascii").lower()
        except UnicodeError:
            return None

        # Reject empty or excessively long hostnames.
        if not ascii_hostname or len(ascii_hostname) > 253:
            return None

        # Validate every hostname label. Labels cannot start or end with
        # a hyphen, and each label must be no longer than 63 characters.
        labels = ascii_hostname.split(".")
        label_pattern = re.compile(
            r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
        )

        for label in labels:
            if not label or not label_pattern.fullmatch(label):
                return None

        # Replace the original hostname with its validated ASCII form.
        # Keep the URL's path, query string, fragment, and port intact.
        netloc = ascii_hostname
        if port is not None:
            netloc = f"{netloc}:{port}"

        normalized = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=netloc
        ).geturl()

        return normalized

    except (ValueError, UnicodeError):
        # Reject malformed URLs rather than letting them distort the audit.
        return None



def get_registered_domain(url):
    """Extract the registered domain, handling IPs and localhost safely."""

    try:
        hostname = urlsplit(url).hostname

        if not hostname:
            return None

        hostname = hostname.rstrip(".").lower()

        # IP addresses are not ordinary registered domains.
        try:
            ipaddress.ip_address(hostname)
            return hostname
        except ValueError:
            pass

        # Handle localhost and other single-label hostnames.
        if "." not in hostname:
            return hostname

        # Extract the registered domain using the public suffix rules.
        result = domain_extractor(hostname)

        if result.domain and result.suffix:
            return f"{result.domain}.{result.suffix}".lower()

        # Preserve the hostname when no registered domain can be extracted.
        return hostname

    except (ValueError, TypeError):
        return None


def summarize_source(source):
    """Load one raw dataset and report its URL and label patterns."""

    # Use the explicit path configured for this dataset.
    file_path = source["path"]

    print("\n" + "=" * 78)
    print(f"SOURCE: {source['file']}")
    print("=" * 78)

    # Stop safely if the expected raw dataset is missing.
    if not file_path.exists():
        print(f"FILE NOT FOUND: {file_path}")
        return None

    # Load the CSV without intentionally changing the source file.
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as error:
        print(f"Could not read file: {error}")
        return None

    url_col = source["url_column"]
    label_col = source["label_column"]

    # Verify the required columns before starting the audit.
    missing_columns = [
        column
        for column in (url_col, label_col)
        if column not in df.columns
    ]

    if missing_columns:
        print(f"Missing expected columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return None

    print(f"Raw rows: {len(df):,}")

    # Standardize label text for analysis only.
    raw_labels = df[label_col].astype("string").str.strip().str.lower()
    labels = raw_labels.map(source["labels"])

    # Build a temporary working table; the CSV remains untouched.
    working = pd.DataFrame({
        "raw_url": df[url_col],
        "raw_label": raw_labels,
        "label": labels,
    })

    # Record the original scheme before normalization adds a default.
    raw_text = working["raw_url"].astype("string").str.strip()

    working["original_scheme"] = "missing"

    working.loc[
        raw_text.str.match(r"(?i)^https://", na=False),
        "original_scheme",
    ] = "https"

    working.loc[
        raw_text.str.match(r"(?i)^http://", na=False),
        "original_scheme",
    ] = "http"

    working.loc[
        raw_text.str.match(
            r"(?i)^[a-z][a-z0-9+.-]*://",
            na=False,
        )
        & ~raw_text.str.match(r"(?i)^https?://", na=False),
        "original_scheme",
    ] = "other"

    # Normalize URLs and exclude rows with invalid URLs or unknown labels.
    working["normalized_url"] = working["raw_url"].map(normalize_url)

    working = working.dropna(
        subset=["label", "normalized_url"]
    ).copy()

    working["label"] = working["label"].astype(int)

    # Parse normalized URLs to inspect their characteristics.
    parsed_urls = working["normalized_url"].map(urlsplit)

    working["normalized_scheme"] = parsed_urls.map(
        lambda parsed: parsed.scheme.lower()
    )

    working["path"] = parsed_urls.map(lambda parsed: parsed.path)
    working["has_query"] = parsed_urls.map(
        lambda parsed: bool(parsed.query)
    )

    # Define homepage-like URLs as root paths without query parameters.
    working["homepage_like"] = (
        working["path"].isin(["", "/"])
        & ~working["has_query"]
    )

    # Extract registered domains for domain-level comparisons.
    working["registered_domain"] = working["normalized_url"].map(
        get_registered_domain
    )

    print(
        "Rows with recognized labels and valid URLs: "
        f"{len(working):,}"
    )
    print(
        "Rows excluded during this audit: "
        f"{len(df) - len(working):,}"
    )

    if working.empty:
        print("No usable rows were found in this dataset.")
        return working

    # Report class balance to reveal source-specific label skew.
    print("\nLabel distribution (1 = phishing, 0 = legitimate):")
    print(
        working["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nLabel percentages:")
    print(
        (
            working["label"].value_counts(normalize=True)
            .sort_index() * 100
        ).round(2).to_string()
    )

    # Report the original scheme distribution for each class.
    print("\nOriginal URL scheme, by label (%):")
    scheme_table = pd.crosstab(
        working["label"],
        working["original_scheme"],
        normalize="index",
    ).mul(100).round(2)

    print(scheme_table.to_string())

    # Report HTTPS rates after URL normalization.
    print("\nNormalized HTTPS rate, by label (%):")
    https_rates = working.groupby("label")["normalized_scheme"].apply(
        lambda values: (values == "https").mean() * 100
    )

    print(https_rates.round(2).to_string())

    # Compare homepage-like URL rates between classes.
    print("\nHomepage-like URL rate, by label (%):")
    homepage_rates = (
        working.groupby("label")["homepage_like"].mean() * 100
    )

    print(homepage_rates.round(2).to_string())

    # Report how many unique domains occur in each class.
    print("\nUnique registered domains, by label:")
    unique_domains = working.groupby("label")[
        "registered_domain"
    ].nunique()

    print(unique_domains.to_string())

    # Show the most frequent domains for each class.
    print("\nMost frequent registered domains (top 10):")

    for label, group in working.groupby("label"):
        counts = Counter(
            domain
            for domain in group["registered_domain"].dropna()
        )

        print(f"\nLabel {label}:")

        for domain, count in counts.most_common(10):
            print(f"  {domain}: {count:,}")

    # Show examples to help interpret the homepage statistics.
    print("\nHomepage-like URL examples, by label (up to 5 each):")

    for label, group in working.groupby("label"):
        examples = (
            group.loc[
                group["homepage_like"],
                "normalized_url",
            ]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        print(f"\nLabel {label}:")

        if not examples:
            print("  No examples found.")

        for example in examples:
            print(f"  {example}")

    # Return the working data for cross-source domain analysis.
    return working


def main():
    """Audit each source and compare registered domains across sources."""

    source_results = {}

    # Analyze each configured raw dataset independently.
    for source in SOURCES:
        result = summarize_source(source)

        if result is not None and not result.empty:
            source_results[source["file"]] = result

    # Compare registered-domain coverage across readable datasets.
    print("\n" + "=" * 78)
    print("CROSS-SOURCE REGISTERED-DOMAIN OVERLAP")
    print("=" * 78)

    domain_sets = {
        name: set(result["registered_domain"].dropna())
        for name, result in source_results.items()
    }

    names = list(domain_sets)

    if len(names) < 2:
        print(
            "At least two readable, non-empty datasets are needed "
            "for this comparison."
        )
    else:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                first = names[i]
                second = names[j]

                shared = domain_sets[first] & domain_sets[second]

                print(f"\n{first} vs {second}")
                print(f"Shared registered domains: {len(shared):,}")
                print(
                    f"Unique domains in {first}: "
                    f"{len(domain_sets[first]):,}"
                )
                print(
                    f"Unique domains in {second}: "
                    f"{len(domain_sets[second]):,}"
                )
                print(
                    "Examples of shared domains: "
                    + (", ".join(sorted(shared)[:10]) or "None")
                )

    print("\nAudit complete. No datasets or model files were changed.")
    print(
        "Important: cross-source domain overlap does not prove "
        "train/validation/test leakage. That requires a separate "
        "split-specific analysis."
    )


if __name__ == "__main__":
    main()
