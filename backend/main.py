
# ============================================================
# ADAPTIVE PHISHING DETECTION SYSTEM - FASTAPI BACKEND
#
# Project architecture:
# 1. Domain-disjoint v3 Random Forest (40 features).
# 2. Passive-Aggressive adaptive classifier (29 features).
# 3. Google Safe Browsing threat intelligence.
# 4. Token-authenticated manual feedback.
# 5. Automatic positive feedback from confirmed phishing matches.
# 6. Persistent SQLite duplicate and conflict protection.
# 7. Narrow trusted-domain exception for verified domains.
#
# IMPORTANT:
# - The Random Forest is never modified by adaptive feedback.
# - Normal URL scans do not directly train either model.
# - Only eligible SOCIAL_ENGINEERING matches qualify for
#   automatic positive phishing feedback.
# - Google Safe Browsing positive matches always take priority.
# - Trusted-domain exceptions do not modify raw RF predictions.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

import os
import secrets
import ipaddress
import sqlite3
import hashlib
import logging

from pathlib import Path
from contextlib import closing
from urllib.parse import urlsplit
from datetime import datetime, timezone

import joblib
import pandas as pd
import numpy as np

from fastapi import (
    FastAPI,
    HTTPException,
    Header,
    BackgroundTasks
)

from pydantic import BaseModel, Field
from dotenv import load_dotenv


# ------------------------------------------------------------
# 2. CONFIGURE PROJECT PATHS AND ENVIRONMENT VARIABLES
# ------------------------------------------------------------

# Locate the root directory of the project.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables before importing project modules.
load_dotenv(BASE_DIR / ".env")

# Retrieve the secret used to authenticate manual feedback.
FEEDBACK_TOKEN = os.getenv("ADAPTIVE_FEEDBACK_TOKEN")

# Prevent the API from starting without its feedback token.
if not FEEDBACK_TOKEN:
    raise RuntimeError(
        "ADAPTIVE_FEEDBACK_TOKEN is missing from .env"
    )


# ------------------------------------------------------------
# 3. IMPORT PROJECT MODULES
# ------------------------------------------------------------

# Google Safe Browsing integration.
from backend.threat_intelligence import check_url

# Existing 29-feature adaptive learning service.
from backend.adaptive_service import adaptive_service

# Extract the 29 features expected by the adaptive learner.
from ml.adaptive_features_29 import adaptive_features_29

# Existing 40-feature URL extractor used during RF training.
from ml.enhanced_features import enhanced_features


# ------------------------------------------------------------
# 4. CONFIGURE APPLICATION LOGGING
# ------------------------------------------------------------

# Use standard Python logging instead of printing sensitive data.
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 5. CONFIGURE VERIFIED TRUSTED DOMAINS
# ------------------------------------------------------------

# Only explicitly verified domains should be included.
# This initial entry addresses the confirmed example.com false positive.
#
# Do not add domains automatically based only on RF predictions.
TRUSTED_DOMAINS = {
    "example.com",
}


def get_trusted_domain(url):
    """
    Return a configured trusted domain when the URL hostname
    exactly matches that domain or is one of its subdomains.

    Hostname-boundary matching prevents lookalike domains such as
    example.com.evil-site.com from receiving trusted treatment.
    """

    try:
        # Extract and normalize the hostname.
        hostname = (
            urlsplit(url).hostname or ""
        ).lower().rstrip(".")

    except ValueError:
        # Invalid hostnames must never receive trusted treatment.
        return None

    # Compare the hostname against each configured trusted domain.
    for domain in TRUSTED_DOMAINS:

        # Normalize the configured domain.
        domain = domain.lower().strip(".")

        # Accept exact matches and genuine subdomains only.
        if (
            hostname == domain
            or hostname.endswith("." + domain)
        ):
            return domain

    return None


# ------------------------------------------------------------
# 6. LOAD THE V3 RANDOM FOREST MODEL
# ------------------------------------------------------------

# Locate the domain-disjoint v3 Random Forest.
MODEL_PATH = (
    BASE_DIR
    / "models_v3_grouped"
    / "random_forest_40_features.joblib"
)

# Locate the saved feature order from training.
FEATURE_ORDER_PATH = (
    BASE_DIR
    / "models_v3_grouped"
    / "random_forest_40_features_features.joblib"
)

# Keep the threshold selected using validation data.
# This is a model score threshold, not necessarily a calibrated probability.
PHISHING_THRESHOLD = 0.747170

# Confirm that both model files exist.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"V3 Random Forest model not found: {MODEL_PATH}"
    )

if not FEATURE_ORDER_PATH.exists():
    raise FileNotFoundError(
        f"V3 feature-order file not found: {FEATURE_ORDER_PATH}"
    )

# Load the trained Random Forest and saved feature order.
model = joblib.load(MODEL_PATH)
EXPECTED_FEATURES = list(joblib.load(FEATURE_ORDER_PATH))

# Use one inference worker in the Random Forest.
if hasattr(model, "n_jobs"):
    model.n_jobs = 1


# ------------------------------------------------------------
# 7. VERIFY THE RANDOM FOREST AND FEATURE ORDER
# ------------------------------------------------------------

# Confirm that exactly 40 features are expected.
if len(EXPECTED_FEATURES) != 40:
    raise RuntimeError(
        f"Expected 40 features, found {len(EXPECTED_FEATURES)}."
    )

# Confirm that feature names are valid and unique.
if (
    not all(
        isinstance(name, str) and name
        for name in EXPECTED_FEATURES
    )
    or len(set(EXPECTED_FEATURES)) != len(EXPECTED_FEATURES)
):
    raise RuntimeError(
        "The saved feature order contains invalid or duplicate names."
    )

# Confirm that the model expects exactly 40 features.
if getattr(model, "n_features_in_", None) != len(EXPECTED_FEATURES):
    raise RuntimeError(
        "The Random Forest feature count does not match "
        "the saved feature order."
    )

# Verify feature names if the model contains them.
if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != EXPECTED_FEATURES:
        raise RuntimeError(
            "The Random Forest feature names do not match "
            "the saved feature-order file."
        )

# Confirm that probability prediction is supported.
if not hasattr(model, "predict_proba"):
    raise RuntimeError(
        "The Random Forest does not support predict_proba()."
    )

# Confirm that phishing class 1 is available.
if 1 not in list(model.classes_):
    raise RuntimeError(
        "The Random Forest does not contain phishing class 1."
    )


# ------------------------------------------------------------
# 8. VERIFY THE 40-FEATURE EXTRACTOR
# ------------------------------------------------------------

# Generate a sample using the same extractor used during training.
sample_features = enhanced_features(
    "https://example.com/"
)

# Confirm that the extractor produces the expected feature names.
if (
    len(sample_features) != 40
    or set(sample_features.keys()) != set(EXPECTED_FEATURES)
):
    raise RuntimeError(
        "The enhanced feature extractor does not match "
        "the v3 Random Forest feature definitions."
    )

# Convert the sample features into the saved training order.
sample_frame = pd.DataFrame(
    [
        [
            sample_features[name]
            for name in EXPECTED_FEATURES
        ]
    ],
    columns=EXPECTED_FEATURES
).to_numpy(dtype=float)

# Test that the Random Forest can process the sample.
sample_scores = model.predict_proba(sample_frame)

# Verify the prediction output shape.
if sample_scores.shape != (1, len(model.classes_)):
    raise RuntimeError(
        "The Random Forest returned an unexpected prediction shape."
    )

# Confirm that the generated scores are valid numbers.
if not np.isfinite(sample_scores).all():
    raise RuntimeError(
        "The Random Forest generated invalid prediction scores."
    )

logger.info(
    "Loaded v3 domain-disjoint Random Forest successfully."
)

logger.info(
    "Random Forest features: %s",
    len(EXPECTED_FEATURES)
)

logger.info(
    "Random Forest threshold: %s",
    PHISHING_THRESHOLD
)


# ------------------------------------------------------------
# 9. INITIALIZE THE FASTAPI APPLICATION
# ------------------------------------------------------------

app = FastAPI(
    title="Adaptive Phishing Detection API"
)


# ------------------------------------------------------------
# 10. CONFIGURE THE AUTOMATIC FEEDBACK DATABASE
# ------------------------------------------------------------

# Keep the feedback database separate from trained model files.
FEEDBACK_DB_DIR = (
    BASE_DIR
    / "results"
    / "adaptive_state"
)

# Create the directory if it does not already exist.
FEEDBACK_DB_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# SQLite database containing automatic feedback history.
FEEDBACK_DB = (
    FEEDBACK_DB_DIR
    / "google_feedback.sqlite3"
)


def initialize_feedback_database():
    """
    Create or safely upgrade the persistent feedback ledger.

    URL hashes prevent duplicate learning.
    The label column allows the system to detect conflicting
    feedback instead of silently accepting it.
    """

    with sqlite3.connect(
        FEEDBACK_DB,
        timeout=10
    ) as connection:

        # Create the table for new installations.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS processed_feedback (
                url_hash TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                label INTEGER
            )
        """)

        # Read existing columns to support older databases.
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(processed_feedback)"
            ).fetchall()
        }

        # Add the label column without deleting existing records.
        if "label" not in columns:
            connection.execute("""
                ALTER TABLE processed_feedback
                ADD COLUMN label INTEGER
            """)

        # Existing Google SOCIAL_ENGINEERING records
        # represent confirmed phishing matches.
        connection.execute("""
            UPDATE processed_feedback
            SET label = 1
            WHERE source = ?
              AND label IS NULL
        """, (
            "google_safe_browsing_social_engineering",
        ))

        # Commit the schema upgrade and data migration.
        connection.commit()


# Initialize the feedback database during backend startup.
initialize_feedback_database()


# ------------------------------------------------------------
# 11. DEFINE API REQUEST DATA STRUCTURES
# ------------------------------------------------------------

class URLRequest(BaseModel):
    # URL submitted for phishing detection.
    url: str = Field(
        min_length=1,
        max_length=8192
    )


class VerifiedFeedback(BaseModel):
    # URL associated with manually submitted feedback.
    url: str = Field(
        min_length=1,
        max_length=8192
    )

    # 0 = legitimate, 1 = phishing.
    label: int = Field(
        ge=0,
        le=1
    )

    # Describes the source of the submitted verification.
    # This field is metadata, not independent proof.
    source: str = Field(
        min_length=3,
        max_length=200
    )


# ------------------------------------------------------------
# 12. VALIDATE SUBMITTED URLS
# ------------------------------------------------------------

def validate_url(url):
    """
    Validate the basic structure of an HTTP/HTTPS URL.

    This function does not visit the website.
    """

    # Remove surrounding whitespace.
    url = url.strip()

    try:
        # Parse the URL into its components.
        parsed = urlsplit(url)

        # Accept only HTTP and HTTPS URLs.
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(
                "Only HTTP and HTTPS URLs are supported."
            )

        # Require a hostname.
        if not parsed.hostname:
            raise ValueError(
                "URL must contain a hostname."
            )

        # Accessing .port also checks malformed port values.
        _ = parsed.port

        # Validate hostname encoding.
        _ = parsed.hostname.encode("idna")

        return url

    except ValueError as error:
        raise ValueError(str(error))


# ------------------------------------------------------------
# 13. LEGACY 15-FEATURE EXTRACTOR
# ------------------------------------------------------------

def features(url):
    """
    Retain the original 15-feature extractor for compatibility.

    The active adaptive model uses adaptive_features_29().
    This legacy function is not used by the current prediction
    and feedback endpoints.
    """

    # Parse the submitted URL.
    parsed = urlsplit(url)

    # Extract hostname and path.
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""

    # Determine whether the hostname is an IP address.
    try:
        is_ip = int(
            ipaddress.ip_address(host) is not None
        )
    except ValueError:
        is_ip = 0

    # Return the original 15 features.
    return {
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
        ),
    }


# ------------------------------------------------------------
# 14. AUTOMATIC GOOGLE SAFE BROWSING FEEDBACK
# ------------------------------------------------------------

def claim_feedback(url, label, source):
    """
    Atomically register a feedback submission before model learning.

    Each exact URL can be used for only one learning update.
    Existing submissions are checked for duplicate or conflicting labels.
    """

    # Hash the URL so the database does not need to store the full URL.
    url_hash = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()

    # Attempt to claim this URL in the persistent feedback ledger.
    with closing(
        sqlite3.connect(FEEDBACK_DB, timeout=10)
    ) as connection:

        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO processed_feedback
            (
                url_hash,
                source,
                status,
                created_at,
                label
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                url_hash,
                source,
                "processing",
                datetime.now(timezone.utc).isoformat(),
                label
            )
        )

        connection.commit()

        # A newly inserted record means this request owns the claim.
        if cursor.rowcount == 1:
            return {
                "status": "claimed",
                "url_hash": url_hash
            }

        # The URL already exists. Retrieve its recorded label and status.
        existing = connection.execute(
            """
            SELECT label, status
            FROM processed_feedback
            WHERE url_hash = ?
            """,
            (url_hash,)
        ).fetchone()

        # Treat an unexpected missing record as a database error.
        if existing is None:
            raise sqlite3.DatabaseError(
                "Existing feedback record could not be retrieved."
            )

        existing_label, existing_status = existing

        # Never allow a conflicting label to overwrite an existing record.
        if existing_label != label:
            return {
                "status": "conflict",
                "existing_label": existing_label,
                "existing_status": existing_status
            }

        # Same label means the URL has already been claimed or processed.
        return {
            "status": "duplicate",
            "existing_status": existing_status
        }


def update_feedback_status(url_hash, status):
    """
    Update the processing status of a previously claimed URL.
    """

    # Record the completion time only after successful learning.
    completed_at = (
        datetime.now(timezone.utc).isoformat()
        if status == "processed"
        else None
    )

    with closing(
        sqlite3.connect(FEEDBACK_DB, timeout=10)
    ) as connection:

        connection.execute(
            """
            UPDATE processed_feedback
            SET status = ?, completed_at = ?
            WHERE url_hash = ?
            """,
            (
                status,
                completed_at,
                url_hash
            )
        )

        connection.commit()


def process_google_feedback(url, threat_result):
    """
    Learn automatically from eligible Google Safe Browsing
    SOCIAL_ENGINEERING matches.

    Google feedback receives label 1 (phishing).
    Malware-only and unwanted-software-only matches are excluded.
    """

    # Read the threat categories returned by Google.
    threat_types = threat_result.get("threat_types", [])

    # Only an explicitly eligible SOCIAL_ENGINEERING match qualifies.
    eligible = (
        threat_result.get("eligible_for_phishing_feedback") is True
        and "SOCIAL_ENGINEERING" in threat_types
    )

    # Ignore results that are not verified phishing matches.
    if not eligible:
        return

    source = "google_safe_browsing_social_engineering"

    try:
        # Claim the URL and save its phishing label before learning.
        claim = claim_feedback(
            url,
            1,
            source
        )

    except sqlite3.Error:
        # Never train if the persistent feedback ledger is unavailable.
        logger.exception(
            "Could not claim Google Safe Browsing feedback."
        )
        return

    # Do not train the same URL twice.
    if claim["status"] == "duplicate":
        logger.info(
            "Google feedback skipped: duplicate URL."
        )
        return

    # Do not override an existing conflicting label.
    if claim["status"] == "conflict":
        logger.warning(
            "Google feedback rejected: conflicting existing label."
        )
        return

    url_hash = claim["url_hash"]

    try:
        # Extract the 29 features expected by the adaptive model.
        adaptive_feature_values = adaptive_features_29(url)

        # Google SOCIAL_ENGINEERING is treated as phishing (label 1).
        learning_result = adaptive_service.learn(
            adaptive_feature_values,
            1
        )

        # Mark the record as successfully processed.
        update_feedback_status(
            url_hash,
            "processed"
        )

        logger.info(
            "Google Safe Browsing feedback processed."
        )

        logger.info(
            "Adaptive learning result: %s",
            learning_result
        )

    except Exception:
        # Preserve the claim if learning or finalization fails.
        # This prevents uncertain updates from being repeated.
        logger.exception(
            "Automatic adaptive feedback failed or was interrupted."
        )

        try:
            update_feedback_status(
                url_hash,
                "failed_or_interrupted"
            )
        except sqlite3.Error:
            logger.exception(
                "Could not update Google feedback failure status."
            )

        return


# ------------------------------------------------------------
# 15. BASIC API ENDPOINTS
# ------------------------------------------------------------

@app.get("/")
def home():
    # Confirm that the API is accessible and identify its models.
    return {
        "message": "Phishing Detection API is running",
        "primary_model": "V3 Random Forest (40 features)",
        "adaptive_model": "Passive-Aggressive (29 features)"
    }


@app.get("/health")
def health():
    # Return a basic health status.
    return {
        "status": "healthy",
        "primary_model": "V3 Random Forest"
    }


# ------------------------------------------------------------
# 16. MAIN PHISHING DETECTION ENDPOINT
# ------------------------------------------------------------

@app.post("/predict")
def predict(
    request: URLRequest,
    background_tasks: BackgroundTasks
):
    """
    Combine the v3 Random Forest, 29-feature adaptive learner,
    and Google Safe Browsing threat intelligence.

    Eligible Google Safe Browsing feedback is processed
    in the background after the response is prepared.

    The v3 Random Forest remains the primary classifier.
    """

    try:
        # ----------------------------------------------------
        # A. VALIDATE THE SUBMITTED URL
        # ----------------------------------------------------

        url = validate_url(request.url)

        # ----------------------------------------------------
        # B. EXTRACT FEATURES FOR BOTH MODELS
        # ----------------------------------------------------

        # Extract the 29 features expected by the adaptive learner.
        adaptive_feature_values = adaptive_features_29(url)

        # Extract the 40 features expected by the Random Forest.
        rf_features = enhanced_features(url)

        # Build the feature matrix in the exact training order.
        X = pd.DataFrame(
            [
                [
                    rf_features[name]
                    for name in EXPECTED_FEATURES
                ]
            ],
            columns=EXPECTED_FEATURES
        ).to_numpy(dtype=float)

        # ----------------------------------------------------
        # C. OBTAIN THE RANDOM FOREST PREDICTION
        # ----------------------------------------------------

        # Locate the probability column for phishing class 1.
        phishing_index = list(model.classes_).index(1)

        # Obtain the Random Forest phishing score.
        phishing_probability = float(
            model.predict_proba(X)[0][phishing_index]
        )

        # Apply the existing validation-selected threshold.
        prediction = int(
            phishing_probability >= PHISHING_THRESHOLD
        )

        # ----------------------------------------------------
        # D. OBTAIN THE ADAPTIVE MODEL PREDICTION
        # ----------------------------------------------------

        # This is a prediction only; it does not train the model.
        adaptive_result = adaptive_service.predict(
            adaptive_feature_values
        )

        adaptive_prediction = adaptive_result["prediction"]

        # Compare both model predictions.
        models_agree = (
            prediction == adaptive_prediction
        )

        # ----------------------------------------------------
        # E. CHECK GOOGLE SAFE BROWSING
        # ----------------------------------------------------

        # Check whether Google identifies the submitted URL as a threat.
        threat_result = check_url(url)

        # ----------------------------------------------------
        # F. DETERMINE THE OVERALL ASSESSMENT
        # ----------------------------------------------------

        # Identify whether the hostname belongs to a configured
        # trusted domain.
        trusted_domain = get_trusted_domain(url)

        # Apply the trusted-domain exception only when Google
        # completed its check and returned no threat match.
        trusted_domain_override = (
            trusted_domain is not None
            and threat_result.get("matched") is False
        )

        # A positive Google threat match always takes priority.
        if threat_result.get("matched") is True:
            combined_status = "known_threat"

        # Apply the narrow trusted-domain exception.
        # The original Random Forest result remains unchanged.
        elif trusted_domain_override:
            combined_status = "no_threat_detected"

        # Otherwise, retain the Random Forest's positive assessment.
        elif prediction == 1:
            combined_status = "suspected_phishing"

        # A negative RF result and completed Google check mean
        # neither source detected a threat.
        elif threat_result.get("matched") is False:
            combined_status = "no_threat_detected"

        # An unavailable or inconclusive Google check is not
        # evidence that the submitted URL is safe.
        else:
            combined_status = "threat_check_unavailable"

        # ----------------------------------------------------
        # G. QUEUE ELIGIBLE AUTOMATIC ADAPTIVE FEEDBACK
        # ----------------------------------------------------

        # Only confirmed SOCIAL_ENGINEERING matches qualify
        # for automatic positive phishing feedback.
        if (
            threat_result.get(
                "eligible_for_phishing_feedback"
            ) is True
            and "SOCIAL_ENGINEERING"
            in threat_result.get("threat_types", [])
        ):

            # Schedule the update after the response is prepared.
            background_tasks.add_task(
                process_google_feedback,
                url,
                threat_result
            )

            feedback_status = {
                "status": "queued",
                "source": "google_safe_browsing",
                "label": 1
            }

        else:
            # No adaptive update for non-eligible results.
            feedback_status = {
                "status": "not_eligible",
                "source": "google_safe_browsing"
            }

        # ----------------------------------------------------
        # H. RETURN THE COMPLETE DETECTION RESULT
        # ----------------------------------------------------

        return {
            "url": url,

            # Preserve the original Random Forest prediction.
            "prediction": prediction,

            # This field describes the raw RF decision.
            "status": (
                "phishing"
                if prediction == 1
                else "legitimate"
            ),

            # Return the original RF score and selected threshold.
            "phishing_probability": round(
                phishing_probability,
                4
            ),

            "phishing_threshold": PHISHING_THRESHOLD,

            # Return both model results and their agreement.
            "model": "V3 Random Forest (40 features)",
            "adaptive_model": adaptive_result,
            "models_agree": models_agree,

            # Return the Google Safe Browsing result.
            "threat_intelligence": threat_result,

            # Report automatic feedback scheduling information.
            "adaptive_feedback": feedback_status,

            # Return the overall assessment used by the extension.
            "combined_status": combined_status,

            # Explicitly disclose whether the trusted-domain
            # exception changed the overall assessment.
            "trusted_domain_override": {
                "applied": trusted_domain_override,
                "domain": trusted_domain,
                "reason": (
                    "Configured trusted domain with no "
                    "Google Safe Browsing threat match."
                    if trusted_domain_override
                    else None
                )
            }
        }

    except ValueError as error:
        # Return HTTP 400 for invalid URL requests.
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ------------------------------------------------------------
# 17. ADAPTIVE MODEL STATUS ENDPOINT
# ------------------------------------------------------------

@app.get("/adaptive/status")
def adaptive_status():
    # Return adaptive learning and ADWIN statistics.
    return adaptive_service.status()


# ------------------------------------------------------------
# 18. ADAPTIVE MODEL PREDICTION ENDPOINT
# ------------------------------------------------------------

@app.post("/adaptive/predict")
def adaptive_predict(request: URLRequest):

    try:
        # Validate the submitted URL.
        url = validate_url(request.url)

        # Extract the 29 adaptive-model features.
        adaptive_feature_values = adaptive_features_29(url)

        # Return a prediction without training the model.
        return {
            "url": url,
            **adaptive_service.predict(adaptive_feature_values)
        }

    except ValueError as error:
        # Report invalid URL requests.
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ------------------------------------------------------------
# 19. AUTHENTICATED MANUAL FEEDBACK AND INCREMENTAL LEARNING
# ------------------------------------------------------------

@app.post("/adaptive/feedback")
def adaptive_feedback(
    request: VerifiedFeedback,
    x_feedback_token: str | None = Header(default=None)
):
    """
    Accept authenticated manual feedback and safely update
    the adaptive model.

    Authentication confirms that the caller knows the feedback
    token. It does not independently prove that the submitted
    phishing or legitimate label is correct.

    The shared SQLite ledger prevents repeated or conflicting
    feedback from training the same exact URL more than once.
    """

    # Require the configured private feedback token.
    if (
        x_feedback_token is None
        or not secrets.compare_digest(
            x_feedback_token,
            FEEDBACK_TOKEN
        )
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid feedback token."
        )

    try:
        # Validate the submitted URL before processing feedback.
        url = validate_url(request.url)

        # Extract the 29 features expected by the adaptive model.
        adaptive_feature_values = adaptive_features_29(url)

        # Claim the URL in the shared feedback ledger.
        try:
            claim = claim_feedback(
                url,
                request.label,
                request.source
            )

        except sqlite3.Error:
            logger.exception(
                "Could not register manual feedback."
            )

            raise HTTPException(
                status_code=503,
                detail="Feedback storage is temporarily unavailable."
            )

        # Reject a different label for a URL already in the ledger.
        if claim["status"] == "conflict":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Conflicting feedback already exists for this URL. "
                    "The existing label has been preserved."
                )
            )

        # Do not train again when this URL has already been submitted.
        if claim["status"] == "duplicate":
            return {
                "url": url,
                "verification_source": request.source,
                "feedback_status": "duplicate",
                "learning_performed": False,
                "existing_status": claim["existing_status"],
                "message": (
                    "This URL has already been submitted. "
                    "No additional learning update was performed."
                )
            }

        # Retrieve the hash of the newly claimed URL.
        url_hash = claim["url_hash"]

        try:
            # Apply the submitted label to the adaptive model.
            result = adaptive_service.learn(
                adaptive_feature_values,
                request.label
            )

            # Mark the feedback as successfully processed.
            update_feedback_status(
                url_hash,
                "processed"
            )

            # Return the learning result and feedback metadata.
            return {
                "url": url,
                "verification_source": request.source,
                "feedback_status": "processed",
                "learning_performed": True,
                **result
            }

        except Exception:
            # Preserve the claim if learning or finalization fails.
            # This avoids uncertain repeated model updates.
            logger.exception(
                "Manual adaptive feedback failed or was interrupted."
            )

            try:
                update_feedback_status(
                    url_hash,
                    "failed_or_interrupted"
                )
            except sqlite3.Error:
                logger.exception(
                    "Could not update manual feedback failure status."
                )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Feedback processing failed or was interrupted. "
                    "The submission was retained to prevent duplicate learning."
                )
            )

    except HTTPException:
        # Preserve intentional HTTP status codes such as 409 and 503.
        raise

    except ValueError as error:
        # Report invalid URL or feedback data.
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


@app.post("/adaptive/feedback-status")
def adaptive_feedback_status(request: URLRequest):
    """
    Retrieve the processing status of a URL's feedback.

    This endpoint is read-only:
    - It does not submit new feedback.
    - It does not modify the SQLite database.
    - It does not train or update the adaptive model.
    """

    try:
        # Validate the URL using the same validation used during submission.
        url = validate_url(request.url)

        # Generate the same SHA-256 hash used by the feedback ledger.
        url_hash = hashlib.sha256(
            url.encode("utf-8")
        ).hexdigest()

        # Retrieve the existing feedback record without modifying it.
        with sqlite3.connect(FEEDBACK_DB) as connection:
            connection.row_factory = sqlite3.Row

            record = connection.execute(
                """
                SELECT source, status, label, created_at, completed_at
                FROM processed_feedback
                WHERE url_hash = ?
                """,
                (url_hash,)
            ).fetchone()

        # Return a clear response when the URL has no feedback record.
        if record is None:
            return {
                "url": url,
                "feedback_status": "not_found",
                "message": "No feedback record exists for this URL."
            }

        # Return the recorded processing information.
        return {
            "url": url,
            "feedback_status": record["status"],
            "verification_source": record["source"],
            "label": record["label"],
            "created_at": record["created_at"],
            "completed_at": record["completed_at"]
        }

    except ValueError as error:
        # Return a client error for invalid URLs.
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except sqlite3.Error:
        # Report database failures without exposing internal details.
        logger.exception(
            "Could not retrieve feedback processing status."
        )

        raise HTTPException(
            status_code=503,
            detail="Feedback status is temporarily unavailable."
        )
