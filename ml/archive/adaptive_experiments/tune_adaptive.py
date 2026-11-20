from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train.drop(columns=["label"])
y_train = train["label"].to_numpy()

X_val = validation[X_train.columns]
y_val = validation["label"].to_numpy()

print("Scaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Reproducible random generator
rng = np.random.default_rng(42)

# Different Passive-Aggressive configurations
experiments = [
    {"C": 0.01, "epochs": 3, "balanced": False},
    {"C": 0.1, "epochs": 3, "balanced": False},
    {"C": 1.0, "epochs": 3, "balanced": False},
    {"C": 0.1, "epochs": 5, "balanced": True},
    {"C": 1.0, "epochs": 5, "balanced": True},
]

results = []

for exp in experiments:

    print("\n" + "=" * 50)
    print("Testing configuration:", exp)

    model = SGDClassifier(
        loss="hinge",
        penalty=None,
        learning_rate="pa1",
        eta0=exp["C"],
        random_state=42
    )

    for epoch in range(exp["epochs"]):

        indices = rng.permutation(len(y_train))

        if exp["balanced"]:
            positive = indices[y_train[indices] == 1]
            negative = indices[y_train[indices] == 0]

            # Balanced exposure for this experiment
            count = min(len(positive), len(negative))

            indices = np.concatenate([
                positive[:count],
                negative[:count]
            ])

            rng.shuffle(indices)

        batch_size = 5000

        for start in range(0, len(indices), batch_size):

            batch_indices = indices[start:start + batch_size]

            model.partial_fit(
                X_train[batch_indices],
                y_train[batch_indices],
                classes=np.array([0, 1])
            )

        print(
            f"Completed epoch {epoch + 1}/{exp['epochs']}"
        )

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"False-positive rate: {fpr:.4f}")

    results.append({
        **exp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_DIR / "adaptive_tuning_results.csv",
    index=False
)

print("\n" + "=" * 50)
print("ALL EXPERIMENT RESULTS")
print(results_df.to_string(index=False))

print("\nResults saved successfully!")