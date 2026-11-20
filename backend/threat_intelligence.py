
"""
Google Safe Browsing threat-intelligence integration.

This module checks URLs against Google's Safe Browsing API.
Only explicit SOCIAL_ENGINEERING matches are eligible to become
positive phishing feedback for the adaptive learning model.

Important:
- A no_match result does NOT prove that a URL is legitimate.
- MALWARE and UNWANTED_SOFTWARE are not automatically phishing labels.
- This module does not train or modify either machine-learning model.
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Resolve the project directory and load API credentials from .env.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Read the API key without exposing it in logs or responses.
API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")

# Google Safe Browsing API endpoint.
API_URL = (
    "https://safebrowsing.googleapis.com/"
    "v4/threatMatches:find"
)

# These are the threat categories requested from Google.
THREAT_TYPES = [
    "MALWARE",
    "SOCIAL_ENGINEERING",
    "UNWANTED_SOFTWARE",
]


def unavailable_result():
    """Return a consistent response when the API cannot be reached."""

    return {
        "status": "unavailable",
        "matched": None,
        "threat_types": [],
        "eligible_for_phishing_feedback": False,
    }


def check_url(url):
    """
    Check a URL against Google Safe Browsing.

    A URL is eligible for positive phishing feedback only when Google
    explicitly reports a SOCIAL_ENGINEERING match.

    This function only checks threat intelligence. It does not train
    the adaptive model.
    """

    # Without an API key, the threat-intelligence check is unavailable.
    if not API_KEY:
        return unavailable_result()

    # Construct Google's threat lookup request.
    payload = {
        "client": {
            "clientId": "adaptive-phishing-project",
            "clientVersion": "1.0.0",
        },
        "threatInfo": {
            "threatTypes": THREAT_TYPES,
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }

    try:
        # Submit the URL to Google Safe Browsing.
        response = requests.post(
            API_URL,
            params={"key": API_KEY},
            json=payload,
            timeout=10,
        )

        # Raise an exception for HTTP errors.
        response.raise_for_status()

        # Extract threat matches returned by Google.
        matches = response.json().get("matches", [])

        # Collect unique threat categories from the response.
        matched_types = sorted({
            match.get("threatType", "UNKNOWN")
            for match in matches
            if isinstance(match, dict)
        })

        # Only SOCIAL_ENGINEERING is eligible for phishing feedback.
        eligible_for_feedback = (
            "SOCIAL_ENGINEERING" in matched_types
        )

        # Return the original fields plus the new eligibility field.
        return {
            "status": "matched" if matches else "no_match",
            "matched": bool(matches),
            "threat_types": matched_types,
            "eligible_for_phishing_feedback": eligible_for_feedback,
        }

    except (requests.RequestException, ValueError, AttributeError):
        # An API failure must never be interpreted as a safe URL.
        return unavailable_result()
