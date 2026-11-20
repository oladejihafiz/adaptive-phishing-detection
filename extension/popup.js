
/*
============================================================
ADAPTIVE PHISHING DETECTOR - BROWSER EXTENSION
POPUP JAVASCRIPT
============================================================

This script connects the Chrome extension popup to the
FastAPI phishing detection backend.

Detection components:
1. Enhanced Random Forest - primary phishing classifier.
2. Passive-Aggressive - adaptive supporting classifier.
3. Google Safe Browsing - known-threat intelligence.
4. Backend combined_status - authoritative overall result.

IMPORTANT:
The backend determines the final assessment.
The popup must not independently override that decision
using the adaptive model's prediction.
============================================================
*/


// ============================================================
// 1. CONFIGURATION
// ============================================================

// FastAPI prediction endpoint.
const API_URL = "http://127.0.0.1:8000/predict";


// ============================================================
// 2. GET HTML ELEMENTS
// ============================================================

// Main URL and scanning controls.
const urlElement = document.getElementById("url");
const checkBtn = document.getElementById("checkBtn");

// Enhanced Random Forest elements.
const mlBadge = document.getElementById("mlBadge");
const mlDescription = document.getElementById("mlDescription");
const probabilityElement = document.getElementById("probability");
const progressFill = document.getElementById("progressFill");

// Adaptive model elements.
const adaptiveBadge = document.getElementById("adaptiveBadge");
const adaptiveDescription = document.getElementById("adaptiveDescription");
const learningStatus = document.getElementById("learningStatus");
const driftStatus = document.getElementById("driftStatus");

// Model agreement elements.
const agreementBadge = document.getElementById("agreementBadge");
const agreementDescription = document.getElementById("agreementDescription");

// Google Safe Browsing elements.
const threatBadge = document.getElementById("threatBadge");
const threatDescription = document.getElementById("threatDescription");

// Overall assessment elements.
const overall = document.getElementById("overall");
const overallTitle = document.getElementById("overallTitle");
const overallDescription = document.getElementById("overallDescription");


// ============================================================
// 3. HELPER FUNCTIONS
// ============================================================

/*
Safely display text in an HTML element.

Using textContent prevents API-provided text from being
interpreted as HTML.
*/
function setText(element, value, fallback = "Unavailable") {
    if (element) {
        element.textContent =
            value !== undefined && value !== null
                ? String(value)
                : fallback;
    }
}


/*
Return a readable prediction label.

The backend uses:
0 = legitimate
1 = phishing
*/
function getPredictionLabel(prediction) {

    if (prediction === 1) {
        return "Phishing";
    }

    if (prediction === 0) {
        return "Legitimate";
    }

    return "Unavailable";
}


/*
Validate that a value is a usable non-negative integer.

This prevents invalid statistics from being displayed.
*/
function getValidCount(value) {

    const count = Number(value);

    if (!Number.isInteger(count) || count < 0) {
        return "Unavailable";
    }

    return count;
}


/*
Display a neutral loading message before a scan begins.
*/
function showLoading() {

    setText(mlBadge, "Scanning...");
    setText(
        mlDescription,
        "Analyzing the URL using the Enhanced Random Forest."
    );

    setText(probabilityElement, "--");

    if (progressFill) {
        progressFill.style.width = "0%";
    }

    setText(adaptiveBadge, "Checking...");
    setText(
        adaptiveDescription,
        "Obtaining adaptive model prediction."
    );

    setText(learningStatus, "Learning updates: --");
    setText(driftStatus, "ADWIN drift events: --");

    setText(agreementBadge, "Checking...");
    setText(
        agreementDescription,
        "Comparing model predictions."
    );

    setText(threatBadge, "Checking...");
    setText(
        threatDescription,
        "Checking Google Safe Browsing."
    );

    setText(overallTitle, "Analysis in Progress");
    setText(
        overallDescription,
        "Please wait while the URL is analyzed."
    );

    if (overall) {
        overall.dataset.status = "loading";
    }
}


/*
Display an error when the API cannot complete a scan.

An API error must never be interpreted as a safe result.
*/
function showError(message) {

    setText(mlBadge, "Unavailable");
    setText(mlDescription, message);

    setText(probabilityElement, "--");

    if (progressFill) {
        progressFill.style.width = "0%";
    }

    setText(adaptiveBadge, "Unavailable");
    setText(
        adaptiveDescription,
        "Adaptive model results could not be retrieved."
    );

    setText(learningStatus, "Learning updates: Unavailable");
    setText(driftStatus, "ADWIN drift events: Unavailable");

    setText(agreementBadge, "Unavailable");
    setText(
        agreementDescription,
        "Model agreement could not be determined."
    );

    setText(threatBadge, "Unavailable");
    setText(
        threatDescription,
        "Threat intelligence results could not be retrieved."
    );

    setText(overallTitle, "Assessment Unavailable");

    setText(
        overallDescription,
        "The phishing detection service could not complete the assessment. The website's safety could not be confirmed."
    );

    if (overall) {
        overall.dataset.status = "error";
    }
}


// ============================================================
// 4. GET THE ACTIVE CHROME TAB
// ============================================================

/*
Retrieve the currently active browser tab.

The extension needs the active tab URL to determine
which website the user wants to inspect.
*/
function getActiveTab() {

    return new Promise((resolve, reject) => {

        chrome.tabs.query(
            {
                active: true,
                currentWindow: true
            },
            (tabs) => {

                // Check whether Chrome reported an error.
                if (chrome.runtime.lastError) {
                    reject(
                        new Error(chrome.runtime.lastError.message)
                    );
                    return;
                }

                // Ensure an active tab was returned.
                if (!tabs || tabs.length === 0) {
                    reject(
                        new Error("No active browser tab was found.")
                    );
                    return;
                }

                resolve(tabs[0]);
            }
        );

    });
}


// ============================================================
// 5. VALIDATE THE URL
// ============================================================

/*
Only HTTP and HTTPS addresses are supported.

Chrome internal pages, extension pages, and other unsupported
protocols must not be submitted to the backend.
*/
function validateURL(rawURL) {

    if (!rawURL) {
        throw new Error(
            "No URL was found in the active browser tab."
        );
    }

    let parsedURL;

    try {
        parsedURL = new URL(rawURL);
    } catch {
        throw new Error(
            "The active tab does not contain a valid URL."
        );
    }

    if (
        parsedURL.protocol !== "http:" &&
        parsedURL.protocol !== "https:"
    ) {
        throw new Error(
            "Only HTTP and HTTPS websites can be analyzed."
        );
    }

    if (!parsedURL.hostname) {
        throw new Error(
            "The URL does not contain a valid hostname."
        );
    }

    return parsedURL.href;
}


// ============================================================
// 6. DISPLAY ENHANCED RANDOM FOREST RESULTS
// ============================================================

/*
The Enhanced Random Forest is the primary static detector.

The backend returns:
- prediction: 0 or 1
- phishing_probability: score between 0 and 1
*/
function displayRandomForestResults(data) {

    const prediction = data.prediction;
    const probability = Number(data.phishing_probability);

    // Display the individual Random Forest classification.
    const predictionLabel = getPredictionLabel(prediction);

    setText(mlBadge, predictionLabel);

    if (prediction === 1) {

        setText(
            mlDescription,
            "The Enhanced Random Forest classified this URL as potentially phishing."
        );

    } else if (prediction === 0) {

        setText(
            mlDescription,
            "The Enhanced Random Forest did not classify this URL as phishing."
        );

    } else {

        setText(
            mlDescription,
            "The Random Forest returned an invalid prediction."
        );
    }

    // Display phishing probability as a percentage.
    if (
        Number.isFinite(probability) &&
        probability >= 0 &&
        probability <= 1
    ) {

        const percentage = probability * 100;

        setText(
            probabilityElement,
            `${percentage.toFixed(2)}%`
        );

        // Update the probability progress bar.
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }

    } else {

        setText(probabilityElement, "Unavailable");

        if (progressFill) {
            progressFill.style.width = "0%";
        }
    }
}


// ============================================================
// 7. DISPLAY ADAPTIVE MODEL RESULTS
// ============================================================

/*
The Passive-Aggressive classifier provides a supporting
prediction.

It does not independently determine the final assessment
displayed by the extension.
*/
function displayAdaptiveResults(data) {

    const adaptive = data.adaptive_model;

    // Check that the backend returned adaptive model information.
    if (!adaptive || typeof adaptive !== "object") {

        setText(adaptiveBadge, "Unavailable");

        setText(
            adaptiveDescription,
            "Adaptive model information was not returned by the backend."
        );

        setText(learningStatus, "Learning updates: Unavailable");
        setText(driftStatus, "ADWIN drift events: Unavailable");

        return;
    }

    // Display the adaptive model's prediction.
    const adaptivePrediction = adaptive.prediction;

    setText(
        adaptiveBadge,
        getPredictionLabel(adaptivePrediction)
    );

    if (adaptivePrediction === 1) {

        setText(
            adaptiveDescription,
            "The adaptive model identified this URL as potentially phishing."
        );

    } else if (adaptivePrediction === 0) {

        setText(
            adaptiveDescription,
            "The adaptive model classified this URL as legitimate."
        );

    } else {

        setText(
            adaptiveDescription,
            "The adaptive model did not return a valid prediction."
        );
    }

    // Preserve the descriptive labels when displaying statistics.
    setText(
        learningStatus,
        `Learning updates: ${getValidCount(adaptive.learning_updates)}`
    );

    setText(
        driftStatus,
        `ADWIN drift events: ${getValidCount(adaptive.drift_events)}`
    );
}


// ============================================================
// 8. DISPLAY MODEL AGREEMENT
// ============================================================

/*
Compare the Enhanced Random Forest and adaptive model.

Agreement is supporting information only.
It does not override the backend's combined_status.
*/
function displayAgreement(data) {

    if (data.models_agree === true) {

        setText(agreementBadge, "Agree");

        setText(
            agreementDescription,
            "Both models returned the same classification."
        );

    } else if (data.models_agree === false) {

        setText(agreementBadge, "Disagree");

        setText(
            agreementDescription,
            "The two models returned different classifications. The backend's overall assessment remains authoritative."
        );

    } else {

        setText(agreementBadge, "Unavailable");

        setText(
            agreementDescription,
            "Model agreement could not be determined."
        );
    }
}


// ============================================================
// 9. DISPLAY GOOGLE SAFE BROWSING RESULTS
// ============================================================

/*
Google Safe Browsing provides known-threat intelligence.

Possible results:
- matched = true: known threat detected.
- matched = false: no known threat match returned.
- null/missing: result is inconclusive or unavailable.

A no-match result does not guarantee that a website is safe.
*/
function displayThreatIntelligence(data) {

    const threat = data.threat_intelligence;

    if (!threat || typeof threat !== "object") {

        setText(threatBadge, "Unavailable");

        setText(
            threatDescription,
            "Google Safe Browsing information was not returned."
        );

        return;
    }

    if (threat.matched === true) {

        setText(threatBadge, "Threat Found");

        setText(
            threatDescription,
            "Google Safe Browsing identified this URL as a known threat."
        );

    } else if (threat.matched === false) {

        setText(threatBadge, "No Match");

        setText(
            threatDescription,
            "Google Safe Browsing did not find a known threat match. This does not guarantee that the website is safe."
        );

    } else {

        setText(threatBadge, "Unavailable");

        setText(
            threatDescription,
            "Google Safe Browsing could not provide a conclusive result."
        );
    }
}


// ============================================================
// 10. DISPLAY THE BACKEND'S OVERALL ASSESSMENT
// ============================================================

/*
IMPORTANT DECISION RULE:

The backend's combined_status is the single source of truth
for the overall assessment.

The backend returns one of four possible statuses:

1. known_threat
2. suspected_phishing
3. no_threat_detected
4. threat_check_unavailable
*/

/*
Display the backend's authoritative overall assessment.

The trusted-domain exception is explained explicitly,
without hiding the Random Forest's original prediction.
*/
function displayOverallAssessment(data) {

    const combinedStatus = data.combined_status;
    const trustedOverride = data.trusted_domain_override;

    // Store the returned status for status-specific styling.
    if (overall) {
        overall.dataset.status = combinedStatus || "unknown";
    }

    switch (combinedStatus) {

        // CASE 1: Google Safe Browsing confirmed a known threat.
        case "known_threat":

            setText(overallTitle, "Known Threat Detected");

            setText(
                overallDescription,
                "Google Safe Browsing identified this URL as a known threat. Avoid visiting this website or providing sensitive information."
            );

            break;


        // CASE 2: The backend suspects phishing.
        case "suspected_phishing":

            setText(overallTitle, "Suspected Phishing");

            setText(
                overallDescription,
                "The Enhanced Random Forest classified this URL as potentially malicious. Exercise caution and avoid entering passwords or sensitive information."
            );

            break;


        // CASE 3: No threat detected.
        case "no_threat_detected":

            setText(overallTitle, "No Threat Detected");

            // Explain the trusted-domain exception explicitly.
            if (
                trustedOverride &&
                trustedOverride.applied === true
            ) {

                const originalPrediction =
                    getPredictionLabel(data.prediction).toLowerCase();

                setText(
                    overallDescription,
                    `This domain (${trustedOverride.domain}) is on the configured trusted-domain list, and Google Safe Browsing reported no known threat. The Random Forest independently classified this URL as ${originalPrediction}. The trusted-domain exception does not change the underlying model prediction. This result does not guarantee complete website safety.`
                );

            } else {

                setText(
                    overallDescription,
                    "The backend did not identify a threat based on its combined assessment, and Google Safe Browsing reported no known threat. This does not guarantee that the website is completely safe."
                );
            }

            break;


        // CASE 4: Threat intelligence could not provide a conclusive result.
        case "threat_check_unavailable":

            setText(overallTitle, "Threat Check Incomplete");

            setText(
                overallDescription,
                "Google Safe Browsing could not provide a conclusive result. The website's safety could not be fully confirmed."
            );

            break;


        // Unknown or missing backend status.
        default:

            setText(overallTitle, "Assessment Unavailable");

            setText(
                overallDescription,
                "The backend returned an unknown assessment. The website's safety could not be confirmed."
            );

            if (overall) {
                overall.dataset.status = "unknown";
            }

            break;
    }
}


// ============================================================
// 11. PROCESS THE API RESPONSE
// ============================================================

/*
Check that the backend returned the expected information
before displaying the results.
*/
function processResponse(data) {

    // Ensure that the API returned a valid object.
    if (!data || typeof data !== "object" || Array.isArray(data)) {
        throw new Error(
            "The backend returned an invalid response."
        );
    }

    // Validate the primary Random Forest prediction.
    if (data.prediction !== 0 && data.prediction !== 1) {
        throw new Error(
            "The backend returned an invalid prediction."
        );
    }

    // Validate the primary phishing score.
    const probability = Number(data.phishing_probability);

    if (
        !Number.isFinite(probability) ||
        probability < 0 ||
        probability > 1
    ) {
        throw new Error(
            "The backend returned an invalid phishing probability."
        );
    }

    // Require the authoritative overall assessment.
    const validStatuses = [
        "known_threat",
        "suspected_phishing",
        "no_threat_detected",
        "threat_check_unavailable"
    ];

    if (
        typeof data.combined_status !== "string" ||
        !validStatuses.includes(data.combined_status)
    ) {
        throw new Error(
            "The backend did not return a valid combined assessment."
        );
    }

    // Display the URL being analyzed.
    setText(urlElement, data.url || "Unknown URL");

    // Display each component's results.
    displayRandomForestResults(data);

    displayAdaptiveResults(data);

    displayAgreement(data);

    displayThreatIntelligence(data);

    // Always display the backend's final assessment last.
    displayOverallAssessment(data);
}


// ============================================================
// 12. MAIN URL SCANNING FUNCTION
// ============================================================

/*
This function:
1. Retrieves the active browser tab.
2. Validates its URL.
3. Sends the URL to FastAPI.
4. Receives the detection results.
5. Displays the backend's authoritative assessment.
*/
async function scanCurrentTab() {

    // Prevent multiple simultaneous scans.
    if (checkBtn) {
        checkBtn.disabled = true;
        checkBtn.textContent = "Scanning...";
    }

    showLoading();

    try {

        // Retrieve the currently active browser tab.
        const tab = await getActiveTab();

        // Validate the active tab URL.
        const targetURL = validateURL(tab.url);

        // Display the URL being checked.
        setText(urlElement, targetURL);

        // Send the URL to the FastAPI backend.
        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                url: targetURL
            })

        });

        // Handle unsuccessful HTTP responses.
        if (!response.ok) {

            let errorMessage =
                `Backend error: HTTP ${response.status}`;

            try {

                const errorData = await response.json();

                if (
                    errorData &&
                    typeof errorData.detail === "string"
                ) {
                    errorMessage = errorData.detail;
                }

            } catch {
                // Keep the original HTTP error message.
            }

            throw new Error(errorMessage);
        }

        // Convert the backend response into JavaScript data.
        const data = await response.json();

        // Log the response for local development and debugging.
        console.log(
            "BACKEND RESPONSE RECEIVED BY EXTENSION:",
            data
        );

        // Display the detection results.
        processResponse(data);

    } catch (error) {

        // Log the technical error for debugging.
        console.error("Phishing scan failed:", error);

        // Display an error instead of incorrectly showing safe.
        showError(
            error.message ||
            "An unexpected error occurred during scanning."
        );

    } finally {

        // Re-enable the scan button.
        if (checkBtn) {
            checkBtn.disabled = false;
            checkBtn.textContent = "Scan Again";
        }
    }
}


// ============================================================
// 13. MANUAL SCAN BUTTON
// ============================================================

/*
Allow the user to manually scan the active browser tab again.
*/
if (checkBtn) {

    checkBtn.addEventListener("click", () => {
        scanCurrentTab();
    });

}


// ============================================================
// 14. AUTOMATIC SCANNING WHEN POPUP OPENS
// ============================================================

/*
Automatically analyze the active browser tab whenever
the extension popup is opened.

popup.js is loaded at the bottom of popup.html, so the
DOMContentLoaded event listener is registered before
the document finishes loading.
*/
document.addEventListener("DOMContentLoaded", () => {

    scanCurrentTab();

});
