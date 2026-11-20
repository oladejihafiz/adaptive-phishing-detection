
"""
Safely smoke-test adaptive learning without changing production state.

This test uses the original model and scaler but redirects all learning-state
writes to a temporary directory. It checks learning, saving, and restoration.
The synthetic feature row is for plumbing tests only, not model evaluation.
"""
import sys
import tempfile
from pathlib import Path



# Ensure Python can find the project packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Import the active service module and its class.
import backend.adaptive_service as adaptive_module


def main():
    # Preserve the real paths so they can be restored after the test.
    original_state_dir = adaptive_module.STATE_DIR
    original_state_file = adaptive_module.STATE_FILE

    try:
        # Keep all test state in a temporary directory, separate from the
        # project's real results/adaptive_state/learning_state.joblib file.
        with tempfile.TemporaryDirectory(prefix="adaptive_smoke_test_") as temp:
            temporary_dir = Path(temp)
            adaptive_module.STATE_DIR = temporary_dir
            adaptive_module.STATE_FILE = temporary_dir / "test_learning_state.joblib"

            # A fresh service loads the original model/scaler because no
            # saved state exists in the temporary directory.
            service = adaptive_module.AdaptiveService()

            # Create a numeric test row with exactly the expected feature names.
            # Zeros are synthetic and must not be treated as a real URL sample.
            test_features = {
                name: 0 for name in service.feature_names
            }

            # Confirm invalid labels are rejected before any learning occurs.
            try:
                service.learn(test_features, 2)
                raise AssertionError("An invalid label was unexpectedly accepted.")
            except ValueError:
                print("PASS: Invalid label rejected.")

            assert service.updates == 0, "Invalid label changed the update count."

            # Perform one synthetic update with a valid binary label.
            result = service.learn(test_features, 0)

            assert result["learning_completed"] is True
            assert result["total_updates"] == 1
            assert service.replay_updates == 0

            # Verify that the state was saved only inside the temporary folder.
            test_state_file = adaptive_module.STATE_FILE
            assert test_state_file.exists(), "Test state file was not created."

            saved_state = adaptive_module.joblib.load(test_state_file)

            required_keys = {
                "model",
                "scaler",
                "detector",
                "updates",
                "errors",
                "drift_events",
                "replay_updates",
                "recent_examples",
            }
            missing_keys = required_keys - set(saved_state.keys())
            assert not missing_keys, f"Missing saved-state keys: {missing_keys}"

            assert saved_state["updates"] == 1
            assert len(saved_state["recent_examples"]) == 1

            print("PASS: Synthetic learning update completed.")
            print("PASS: State saved with replay and recent-example fields.")

            # Create another service instance to verify state restoration.
            restored_service = adaptive_module.AdaptiveService()
            restored_status = restored_service.status()

            assert restored_status["learning_updates"] == 1
            assert restored_status["recent_verified_examples"] == 1
            assert restored_status["replay_updates"] == 0

            print("PASS: Saved test state restored successfully.")
            print("\nISOLATED SMOKE TEST PASSED.")
            print("Production adaptive state was not used for test writes.")

    finally:
        # Restore the real module paths even if an assertion fails.
        adaptive_module.STATE_DIR = original_state_dir
        adaptive_module.STATE_FILE = original_state_file


if __name__ == "__main__":
    main()
