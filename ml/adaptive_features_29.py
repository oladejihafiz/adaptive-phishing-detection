
"""
Extract the exact 29 URL features used by the v3 adaptive model.

The feature definitions and calculations must remain consistent
with the training implementation in ml/train_adaptive_29.py.

This module is separate from the 40-feature Random Forest extractor.
"""

# Standard libraries for URL parsing and IP address detection.
import ipaddress
import re
from urllib.parse import urlsplit


def adaptive_features_29(url):
    """
    Extract the 29 features expected by the adaptive classifier.

    The first 15 features retain the original feature definitions.
    The remaining 14 features provide additional URL structure
    and hostname/path information.

    The backend validates URLs before calling this function.
    """

    # Convert the input to a string and remove surrounding spaces.
    url = str(url).strip()

    # Parse the URL into its components.
    parsed = urlsplit(url)

    # Extract the hostname, path, and query string.
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    # Determine whether the hostname is an IP address.
    try:
        is_ip = int(
            ipaddress.ip_address(host) is not None
        )
    except ValueError:
        is_ip = 0

    # --------------------------------------------------------
    # 1. ORIGINAL 15 FEATURES
    # --------------------------------------------------------

    result = {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),

        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(
            character.isdigit()
            for character in url
        ),

        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equals": url.count("="),
        "num_slashes": url.count("/"),

        "is_https": int(
            parsed.scheme.lower() == "https"
        ),

        "has_ip_host": is_ip,
        "num_host_parts": len(host.split(".")),
        "has_punycode": int("xn--" in host),

        "suspicious_words": sum(
            word in url.lower()
            for word in (
                "verify",
                "login",
                "account",
                "secure",
                "update",
                "password",
                "bank"
            )
        )
    }

    # --------------------------------------------------------
    # 2. ADDITIONAL 14 FEATURES
    # --------------------------------------------------------

    # Count digits separately in the hostname and URL path.
    result["hostname_digits"] = sum(
        character.isdigit()
        for character in host
    )

    result["path_digits"] = sum(
        character.isdigit()
        for character in path
    )

    # Measure the proportion of digits in the hostname and URL.
    result["hostname_digit_ratio"] = (
        result["hostname_digits"] / max(len(host), 1)
    )

    result["url_digit_ratio"] = (
        result["num_digits"] / max(len(url), 1)
    )

    # Count hyphens separately in the hostname and path.
    result["hostname_hyphens"] = host.count("-")
    result["path_hyphens"] = path.count("-")

    # Measure the length of the URL query string.
    result["query_length"] = len(query)

    # Identify whether the URL contains a fragment.
    result["has_fragment"] = int(
        bool(parsed.fragment)
    )

    # Identify percent-encoded characters such as %20.
    result["has_percent_encoding"] = int(
        bool(re.search(r"%[0-9a-fA-F]{2}", url))
    )

    # Identify an explicitly specified numeric port.
    result["has_explicit_port"] = int(
        bool(re.search(r":\d+$", parsed.netloc))
    )

    # Estimate subdomain depth using hostname components.
    host_parts = [
        part
        for part in host.split(".")
        if part
    ]

    result["approx_subdomain_depth"] = max(
        len(host_parts) - 2,
        0
    )

    # Count non-empty URL path segments.
    result["path_segments"] = len([
        segment
        for segment in path.split("/")
        if segment
    ])

    # Count suspicious words appearing in the hostname.
    suspicious_words = (
        "verify",
        "login",
        "account",
        "secure",
        "update",
        "password",
        "bank"
    )

    result["hostname_suspicious_words"] = sum(
        word in host
        for word in suspicious_words
    )

    # Count suspicious words appearing in the URL path.
    result["path_suspicious_words"] = sum(
        word in path.lower()
        for word in suspicious_words
    )

    # Return all 29 features in their training-compatible order.
    return result
