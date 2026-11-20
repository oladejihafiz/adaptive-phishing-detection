
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Google Safe Browsing API key is missing from .env"
    )

API_URL = (
    "https://safebrowsing.googleapis.com/"
    "v4/threatMatches:find"
)

payload = {
    "client": {
        "clientId": "adaptive-phishing-project",
        "clientVersion": "1.0.0"
    },
    "threatInfo": {
        "threatTypes": [
            "MALWARE",
            "SOCIAL_ENGINEERING",
            "UNWANTED_SOFTWARE"
        ],
        "platformTypes": ["ANY_PLATFORM"],
        "threatEntryTypes": ["URL"],
        "threatEntries": [
            {"url": "https://www.google.com/"}
        ]
    }
}

print("Checking URL against Google Safe Browsing...")

try:
    response = requests.post(
        API_URL,
        params={"key": API_KEY},
        json=payload,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()
    matches = data.get("matches", [])

    if matches:
        print("Known threat match found!")
        for match in matches:
            print("Threat type:", match.get("threatType"))
    else:
        print("No known threat match found.")

    print("API request completed successfully.")

except requests.RequestException as error:
    print("Threat intelligence request failed.")
    print("Error:", error)
