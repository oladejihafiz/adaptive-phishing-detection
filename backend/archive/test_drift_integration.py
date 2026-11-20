
# Import tools for temporary files and paths.
import sys
import tempfile
from pathlib import Path

# Add the project root to Python's import path.
# This allows the script to import the backend package when
# executed directly from the project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import NumPy to check whether the classifier changed.
import numpy as np

# Import the existing adaptive learning module.
import backend.adaptive_service as service_module


# This test detector simulates an ADWIN drift signal.
# It allows us to verify the response without waiting
# for thousands of real verified examples.
class ControlledDriftDetector:
    def __init__(self):
        # The drift signal begins as False.
        self.drift_detected = False

        # Track how many prediction errors were received.
        self.count = 0

        # Provide a window-width value for compatibility.
        self.width = 0.0

    def update(self, error):
        # Count the prediction-error observation.
        self.count += 1
        self.width = float(self.count)

        # Trigger a controlled drift event on the 20th update.
        self.drift_detected = self.count == 20


# Use a temporary folder to protect the real saved state.
with tempfile.TemporaryDirectory() as temporary_folder:

    # Preserve both production paths before redirecting test writes.
    original_state_dir = service_module.STATE_DIR
    original_state_file = service_module.STATE_FILE

    # Route both the saved state and temporary save files into the
    # temporary directory so the integration test cannot write to
    # the real adaptive-state folder.
    service_module.STATE_DIR = Path(temporary_folder)
    service_module.STATE_FILE = (
       service_module.STATE_DIR / "integration_state.joblib"
)

    try:
        # Create an isolated service from the original model.
        test_service = service_module.AdaptiveService()

        # Replace ADWIN only in this isolated test instance.
        test_service.detector = ControlledDriftDetector()

        # Supply 20 synthetic, labelled feature examples.
        for index in range(20):

            # Create all required feature names.
            features = {
                name: 0
                for name in test_service.feature_names
            }

            # Vary selected feature values.
            features["url_length"] = 25 + index
            features["hostname_length"] = 12 + (index % 5)
            features["num_digits"] = index % 6

            # Alternate the two class labels.
            verified_label = index % 2

            # Run the real learning method.
            result = test_service.learn(
                features,
                verified_label
            )

        # The final update should trigger replay.
        print("Drift detected:", result["adwin_drift_detected"])
        print("New feedback updates:", result["total_updates"])
        print("Examples replayed:", result["replayed_examples"])
        print("Total replay updates:", result["total_replay_updates"])
        print("Drift events:", result["total_drift_events"])

        # Confirm the integrated behavior.
        assert result["adwin_drift_detected"] is True
        assert result["total_updates"] == 20
        assert result["replayed_examples"] == 20
        assert result["total_replay_updates"] == 20
        assert result["total_drift_events"] == 1

        # Load the state saved by the real learning method.
        saved = service_module.joblib.load(
            service_module.STATE_FILE
        )

        # Confirm the counters survived serialization.
        assert saved["updates"] == 20
        assert saved["replay_updates"] == 20
        assert len(saved["recent_examples"]) == 20
        assert len(saved["drift_events"]) == 1

        # Check that the saved classifier matches the
        # classifier currently held by the test service.
        assert np.array_equal(
            saved["model"].coef_,
            test_service.model.coef_
        )

        print("PASS: Drift triggered replay and state was saved.")

    finally:
        # Restore both original paths even if the test fails.
        service_module.STATE_DIR = original_state_dir
        service_module.STATE_FILE = original_state_file
