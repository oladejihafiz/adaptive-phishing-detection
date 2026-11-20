# Adaptive Phishing Detection

A machine-learning-based phishing detection system combining static URL analysis, online learning, concept-drift detection, and Google Safe Browsing threat intelligence.

## Project Overview

This project implements a Chrome browser extension supported by a FastAPI backend. It analyses the URL of the active browser tab and assesses its likelihood of being a phishing website.

The system combines:

- **Random Forest:** Primary phishing classifier using 40 URL-derived features.
- **Passive-Aggressive Classifier:** Adaptive online learner using 29 features and verified feedback.
- **ADWIN:** Monitors prediction errors during feedback-based learning to detect potential concept drift.
- **Google Safe Browsing:** Checks URLs against Google's threat intelligence service.

The project explores how static machine learning and adaptive online learning can work together to improve phishing detection as threat patterns evolve.

## System Architecture

1. The Chrome extension obtains the active tab's URL.
2. The extension sends the URL to the FastAPI backend.
3. The backend extracts URL features and obtains predictions from the Random Forest and adaptive classifier.
4. Google Safe Browsing checks the URL for known threats.
5. The backend combines the available results and returns a detection status.
6. Eligible verified feedback can update the adaptive learner.
7. ADWIN monitors prediction errors during adaptive learning.

## Technologies Used

- Python
- FastAPI
- Scikit-learn
- River
- Pandas and NumPy
- Google Safe Browsing API
- JavaScript, HTML and CSS
- Chrome Extensions Manifest V3
- Git LFS

## Project Structure

```text
adaptive-phishing-detection/
├── backend/
│   ├── main.py
│   ├── adaptive_service.py
│   └── threat_intelligence.py
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   └── popup.js
├── ml/
│   ├── enhanced_features.py
│   ├── adaptive_features_29.py
│   ├── train_models_v3.py
│   └── evaluation/
├── models/
│   ├── adaptive_model_29.joblib
│   └── scaler_29.joblib
├── models_v3_grouped/
│   ├── random_forest_40_features.joblib
│   └── random_forest_40_features_features.joblib
├── results/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Requirements

- Python installed on your computer
- Google Chrome
- A Google Safe Browsing API key
- Git and Git LFS

## Installation

Clone the repository:

```bash
git clone https://github.com/oladejihafiz/adaptive-phishing-detection.git
cd adaptive-phishing-detection
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Configuration

Create your local `.env` file using `.env.example` as a template.

Configure the following variables:

```env
GOOGLE_SAFE_BROWSING_API_KEY=your_google_safe_browsing_api_key
ADAPTIVE_FEEDBACK_TOKEN=your_secure_feedback_token
```

Use your own credentials. Never publish your actual `.env` file or expose API keys and authentication tokens.

## Running the Backend

From the project root, start the FastAPI application:

```powershell
python -m uvicorn backend.main:app --reload
```

The local API will be available at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

## Installing the Chrome Extension

1. Start the backend.
2. Open `chrome://extensions/` in Google Chrome.
3. Enable **Developer mode**.
4. Select **Load unpacked**.
5. Select the project's `extension` folder.
6. Open the extension from Chrome's toolbar.
7. Scan the active tab or select **Scan Again**.

The extension currently communicates with the local backend at `http://127.0.0.1:8000`.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | API information |
| `GET /health` | Backend health check |
| `POST /predict` | Analyse a URL |
| `GET /adaptive/status` | Adaptive learner status |
| `POST /adaptive/predict` | Adaptive model prediction |
| `POST /adaptive/feedback` | Submit authenticated feedback |
| `GET /adaptive/feedback-status` | Check feedback processing status |

The feedback endpoint requires the configured authentication token.

## Adaptive Learning and Concept Drift

The adaptive component uses a Passive-Aggressive classifier that can learn incrementally from verified feedback.

ADWIN monitors prediction errors during feedback-based updates. When drift is detected, the system can replay recent verified examples to help the adaptive learner respond to changing patterns.

The adaptive learner is separate from the Random Forest model. Feedback updates the adaptive component; it does not automatically retrain the Random Forest.

## Evaluation

The repository includes evaluation scripts and selected aggregate reports covering:

- Classification performance
- Precision, recall and F1-score
- False-positive and false-negative analysis
- Threshold sensitivity
- Performance across data sources
- Adaptive learning and drift-replay experiments

Refer to the reports under `results/` for the corresponding experimental results. Reported metrics should be interpreted in the context of their evaluation datasets and thresholds.

## Important Limitations

- The extension currently scans the active tab when opened or when the user requests another scan. It does not continuously monitor browsing activity.
- It does not automatically block websites.
- Detection relies primarily on URL-derived features and available threat-intelligence results.
- A Google Safe Browsing no-match result does not guarantee that a website is safe.
- Threat-intelligence checks may be unavailable due to connectivity or API issues.
- Adaptive learning depends on verified feedback; ordinary URL scans do not automatically train the models.
- The current extension is configured for a locally running backend and is not a hosted production service.

## Security Notes

- Keep API keys and feedback tokens private.
- Do not commit `.env` files.
- Do not publish private datasets, individual URL-level predictions, or sensitive feedback records.
- Use authenticated and trusted sources when submitting adaptive feedback.

## Project Status

Developed as a final-year academic project exploring machine-learning-based phishing detection, adaptive online learning, and browser-based threat assessment.

---

**Repository:** [https://github.com/oladejihafiz/adaptive-phishing-detection](https://github.com/oladejihafiz/adaptive-phishing-detection)