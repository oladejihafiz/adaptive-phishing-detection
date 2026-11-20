
# STEP 38: DOMAIN OVERLAP ANALYSIS
#
# Purpose:
# Check whether URLs in our validation dataset belong to
# registered domains that also appear in the training dataset.
#
# Example:
# https://login.example.com and https://shop.example.com
# belong to the same registered domain: example.com.

from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import tldextract


# Find the project's main directory.
ROOT = Path(__file__).resolve().parents[1]

# Locate the processed datasets.
DATA_DIR = ROOT / "dataset" / "processed"

# Locate the folder for saving our analysis results.
RESULTS_DIR = ROOT / "results" / "metrics"

# Create the results folder if it does not exist.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use the packaged Public Suffix List.
# This avoids downloading domain information from the internet.
extractor = tldextract.TLDExtract(suffix_list_urls=())


def get_registered_domain(url):
    """
    Extract the registered domain from a URL.

    Examples:
    https://mail.google.com -> google.com
    https://login.example.co.uk -> example.co.uk

    This function does not visit the website.
    """

    # Convert the URL to text and remove surrounding spaces.
    url = str(url).strip()

    # Add a scheme if the URL does not already have one.
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url

    # Extract the hostname from the URL.
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()

    # Separate the hostname into subdomain, domain, and suffix.
    extracted = extractor(hostname)

    # Get the registered domain, such as google.com.
    registered_domain = extracted.top_domain_under_public_suffix

    # Use the hostname as a fallback for unusual domains.
    return registered_domain or hostname


def main():

    print("Loading training and validation datasets...")

    # Load the URL datasets.
    train = pd.read_csv(DATA_DIR / "train_urls.csv")
    validation = pd.read_csv(DATA_DIR / "validation_urls.csv")

    # Ensure that the required columns exist.
    for name, dataset in [
        ("Training", train),
        ("Validation", validation),
    ]:
        if not {"url", "label"}.issubset(dataset.columns):
            raise ValueError(
                f"{name} dataset must contain 'url' and 'label' columns."
            )

    print("Extracting registered domains from training URLs...")

    # Convert training URLs into registered domains.
    train_domains = train["url"].map(get_registered_domain)

    print("Extracting registered domains from validation URLs...")

    # Convert validation URLs into registered domains.
    validation_domains = validation["url"].map(
        get_registered_domain
    )

    # Build a set of unique training domains.
    train_domain_set = set(train_domains)

    # Mark validation URLs whose domains appeared in training.
    overlap = validation_domains.isin(train_domain_set)

    # Calculate overall statistics.
    total_validation = len(validation)
    overlapping_urls = int(overlap.sum())
    unseen_urls = total_validation - overlapping_urls

    overlap_percentage = (
        overlapping_urls / total_validation * 100
        if total_validation else 0
    )

    unseen_percentage = (
        unseen_urls / total_validation * 100
        if total_validation else 0
    )

    print("\n" + "=" * 55)
    print("DOMAIN OVERLAP ANALYSIS")
    print("=" * 55)

    print(f"Training URLs: {len(train):,}")
    print(f"Validation URLs: {total_validation:,}")

    print(f"\nUnique training domains: {len(train_domain_set):,}")
    print(f"Overlapping-domain validation URLs: {overlapping_urls:,}")
    print(f"Unseen-domain validation URLs: {unseen_urls:,}")

    print(f"\nOverlapping-domain percentage: {overlap_percentage:.2f}%")
    print(f"Unseen-domain percentage: {unseen_percentage:.2f}%")

    # Analyze legitimate and phishing URLs separately.
    print("\nBREAKDOWN BY URL CLASS")
    print("-" * 55)

    for label, class_name in [
        (0, "Legitimate"),
        (1, "Phishing"),
    ]:

        # Select URLs belonging to the current class.
        class_mask = validation["label"] == label

        class_total = int(class_mask.sum())

        # Count how many URLs in this class share a training domain.
        class_overlap = int((overlap & class_mask).sum())

        class_unseen = class_total - class_overlap

        class_overlap_percentage = (
            class_overlap / class_total * 100
            if class_total else 0
        )

        print(f"\n{class_name} URLs:")
        print(f"  Total: {class_total:,}")
        print(f"  Overlapping domains: {class_overlap:,}")
        print(f"  Unseen domains: {class_unseen:,}")
        print(
            f"  Domain overlap: "
            f"{class_overlap_percentage:.2f}%"
        )

    # Save the summary so it can be used in the project report.
    summary = pd.DataFrame([
        {
            "training_urls": len(train),
            "validation_urls": total_validation,
            "unique_training_domains": len(train_domain_set),
            "overlapping_validation_urls": overlapping_urls,
            "unseen_domain_validation_urls": unseen_urls,
            "overlap_percentage": overlap_percentage,
            "unseen_percentage": unseen_percentage,
        }
    ])

    output_file = RESULTS_DIR / "domain_overlap_summary.csv"

    summary.to_csv(output_file, index=False)

    print("\n" + "=" * 55)
    print(f"Results saved to: {output_file}")
    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()

