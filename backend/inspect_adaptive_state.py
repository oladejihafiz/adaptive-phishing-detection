
"""Inspect saved adaptive-learning statistics without modifying state."""

from pathlib import Path
import joblib

# Locate the saved state relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = (
    PROJECT_ROOT
    / "results"
    / "adaptive_state"
    / "learning_state.joblib"
)

if not STATE_FILE.exists():
    raise FileNotFoundError(
        f"Adaptive state file not found: {STATE_FILE}"
    )

# Load the trusted state file for read-only inspection.
state = joblib.load(STATE_FILE)

# Print counters without displaying the model or training data.
print("Saved adaptive-learning state")
print("-" * 35)
print("Learning updates:", state.get("updates", "missing"))
print("Prediction errors:", state.get("errors", "missing"))
print("Drift events:", len(state.get("drift_events", [])))
print("Replay updates:", state.get("replay_updates", 0))
print(
    "Recent verified examples:",
    len(state.get("recent_examples", []))
)
print("Saved keys:", sorted(state.keys()))
