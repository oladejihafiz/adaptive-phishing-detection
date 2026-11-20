
# Import libraries for URL parsing and statistical calculations.
import math
from urllib.parse import urlsplit

import tldextract

# Reuse the 29 features from our previous experiment.
from ml.train_static_improved import improved_features, normalize_url

# Disable automatic public-suffix-list downloads.
# This helps make feature extraction predictable and avoids
# unexpected network requests during model training or scanning.
extractor = tldextract.TLDExtract(suffix_list_urls=())


def calculate_entropy(text):
    """
    Calculate Shannon entropy, which measures character diversity.

    Higher entropy can indicate more varied characters,
    but does not automatically mean a URL is malicious.
    """
    if not text:
        return 0.0

    # Count the frequency of each unique character.
    length = len(text)

    probabilities = [
        text.count(character) / length
        for character in set(text)
    ]

    # Apply the Shannon entropy formula.
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities
    )


def enhanced_features(url):
    """
    Extract the original 29 experimental features
    plus additional domain and statistical features.

    The result is a dictionary of numerical values.
    No website is visited during extraction.
    """

    # Normalize the URL using the same procedure as training.
    url = normalize_url(url)

    # Begin with our existing experimental feature set.
    result = improved_features(url)

    # Parse the URL into components.
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()

    # Extract registered-domain components using the suffix list.
    extracted = extractor(hostname)

    registered_domain = extracted.domain or ""
    suffix = extracted.suffix or ""
    subdomain = extracted.subdomain or ""

    # Add statistical character-distribution features.
    result["url_entropy"] = calculate_entropy(url.lower())
    result["hostname_entropy"] = calculate_entropy(hostname)
    result["registered_domain_entropy"] = calculate_entropy(
        registered_domain
    )

    # Add registered-domain structure features.
    result["registered_domain_length"] = len(registered_domain)
    result["suffix_length"] = len(suffix)

    result["subdomain_length"] = len(subdomain)

    result["subdomain_label_count"] = (
        len(subdomain.split(".")) if subdomain else 0
    )

    # Identify URL-like strings embedded inside paths or queries.
    # These are potential structural signals, not proof of phishing.
    result["embedded_http_in_path"] = int(
        "http://" in path or "https://" in path
    )

    result["embedded_http_in_query"] = int(
        "http://" in query or "https://" in query
    )

    # Measure the concentration of digits in the registered domain.
    domain_digits = sum(
        character.isdigit() for character in registered_domain
    )

    result["registered_domain_digit_ratio"] = (
        domain_digits / max(len(registered_domain), 1)
    )

    # Measure the concentration of hyphens in the registered domain.
    result["registered_domain_hyphen_ratio"] = (
        registered_domain.count("-")
        / max(len(registered_domain), 1)
    )

    return result
