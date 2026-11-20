from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

print("Loading datasets...")

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train.drop(columns=["label"])
y_train = train["label"]

X_validation = validation.drop(columns=["label"])
y_validation = validation["label"]

print("Training records:", len(X_train))
print("Validation records:", len(X_validation))

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nEvaluating model...")

predictions = model.predict(X_validation)

print("\nAccuracy:", accuracy_score(y_validation, predictions))
print("Precision:", precision_score(y_validation, predictions))
print("Recall:", recall_score(y_validation, predictions))
print("F1-score:", f1_score(y_validation, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, predictions))

print("\nClassification Report:")
print(classification_report(y_validation, predictions))

model_path = MODEL_DIR / "static_model.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved successfully to: {model_path}")