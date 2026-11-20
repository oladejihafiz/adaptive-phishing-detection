from pathlib import Path
import joblib
import pandas as pd

# Load the enhanced Random Forest used by the backend.
root = Path.cwd()
model_path = root / "models" / "static_model_enhanced.joblib"
model = joblib.load(model_path)

# Load the feature names in the exact order used during training.
feature_path = root / "models" / "static_enhanced_features.joblib"
feature_order = list(joblib.load(feature_path))

# Display the features with the greatest global importance.
print("\nTOP 20 ENHANCED MODEL FEATURES")
importance = pd.DataFrame({
    "feature": feature_order,
    "importance": model.feature_importances_,
})
print(importance.sort_values("importance", ascending=False).head(20).to_string(index=False))

# Inspect existing validation threshold results, if available.
comparison_path = root / "results" / "metrics" / "enhanced_threshold_comparison.csv"
print("\nVALIDATION THRESHOLD COMPARISON")
if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)
    print(comparison.to_string(index=False))
else:
    print(f"File not found: {comparison_path}")
