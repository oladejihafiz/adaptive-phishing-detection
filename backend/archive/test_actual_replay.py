
"""
Test whether replay learning changes the adaptive classifier.

SAFETY:
- Uses the original trained model and scaler.
- Writes no test state to the production adaptive-state directory.
- Uses synthetic examples, not real verified URL feedback.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Add the project root so Python can import the backend package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the active adaptive service module.
import backend.adaptive_service as service_module


def main():
    # Remember the production paths so we can restore them afterwards.
    original_state_dir = service_module.STATE_DIR
    original_state_file = service_module.STATE_FILE

    try:
        # Keep all test state inside a disposable temporary directory.
        with tempfile.TemporaryDirectory() as temporary_folder:
            test_dir = Path(temporary_folder)

            # Redirect both state paths to avoid writing test files
            # into the real results/adaptive_state directory.
            service_module.STATE_DIR = test_dir
            service_module.STATE_FILE = test_dir / "test_learning_state.joblib"

            # Create a separate service using the original model and scaler.
            # No test state exists yet, so it loads the initial model.
            test_service = service_module.AdaptiveService()

            # Create 20 synthetic feature examples using the exact
            # feature names expected by the adaptive model.
            for index in range(20):
                example_features = {
                    name: 0 for name in test_service.feature_names
                }

                # Vary a few numerical features between examples.
                # These values are artificial and are not real URLs.
                if "url_length" in example_features:
                    example_features["url_length"] = 20 + index

                if "hostname_length" in example_features:
                    example_features["hostname_length"] = 10 + (index % 5)

                if "num_digits" in example_features:
                    example_features["num_digits"] = index % 4

                # Alternate labels to exercise both classes.
                # These are synthetic labels for testing only.
                label = index % 2

                # Add each example to the recent-example buffer.
                test_service.recent_examples.append({
                    "features": example_features,
                    "label": label,
                })

            # Save the classifier parameters before replay.
            coefficients_before = test_service.model.coef_.copy()
            intercept_before = test_service.model.intercept_.copy()

            # Call the actual replay method implemented in the service.
            replayed_count = test_service._replay_recent_examples()

            # Capture the classifier parameters after replay.
            coefficients_after = test_service.model.coef_.copy()
            intercept_after = test_service.model.intercept_.copy()

            # Determine whether replay changed the classifier.
            parameters_changed = (
                not np.array_equal(
                    coefficients_before, coefficients_after
                )
                or not np.array_equal(
                    intercept_before, intercept_after
                )
            )

            # Display the results so we can inspect the test.
            print("Actual replayed examples:", replayed_count)
            print("Replay updates counter:", test_service.replay_updates)
            print("Classifier parameters changed:", parameters_changed)

            # Verify the expected replay behaviour.
            assert replayed_count == 20, (
                f"Expected 20 replayed examples, got {replayed_count}"
            )
            assert test_service.replay_updates == 20, (
                "Replay counter did not increase as expected."
            )
            assert parameters_changed, (
                "Classifier parameters did not change after replay."
            )

            print("PASS: Actual classifier replay updated the model.")

    finally:
        # Restore the original paths even if the test fails.
        service_module.STATE_DIR = original_state_dir
        service_module.STATE_FILE = original_state_file


if __name__ == "__main__":
    main()
