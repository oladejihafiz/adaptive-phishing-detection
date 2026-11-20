
"""
Safely test the adaptive feedback API without updating the production model.

The real adaptive learner is temporarily replaced with a fake function.
This allows us to test API authentication, URL validation, and the success
response without changing the saved adaptive learning state.
"""

import sys
from pathlib import Path

# Add the project root so imports work when this script is run from
# the project's root directory in Windows PowerShell.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
import backend.main as main


def run_feedback_api_tests():
    """Run API tests while preventing real adaptive model updates."""

    # Confirm that the application's private token was loaded.
    assert main.FEEDBACK_TOKEN, (
        "Feedback token is missing. Check ADAPTIVE_FEEDBACK_TOKEN in .env."
    )

    # Save the real learner method so it can be restored after testing.
    original_learn = main.adaptive_service.learn

    # Record calls made by the API to our fake learner.
    fake_calls = []

    def fake_learn(feature_values, label):
        """
        Simulate a successful learning update without training or saving
        the real model. The test can inspect what the API passed here.
        """
        fake_calls.append({
            "feature_values": feature_values,
            "label": label,
        })

        # Return a small simulated result for the API response.
        return {
            "test_update": True,
            "adwin_drift_detected": False,
        }

    # Temporarily replace the real learner with the safe fake function.
    main.adaptive_service.learn = fake_learn

    try:
        # Test the application directly; a running Uvicorn server is
        # not required.
        client = TestClient(main.app)

        valid_payload = {
            "url": "https://example.com",
            "label": 0,
            "source": "manual verification",
        }

        # ---------------------------------------------------------
        # TEST 1: Reject a request with no feedback token.
        # ---------------------------------------------------------
        response = client.post(
            "/adaptive/feedback",
            json=valid_payload,
        )

        assert response.status_code == 403, (
            f"Missing-token test failed: {response.status_code}"
        )
        assert len(fake_calls) == 0
        print("PASS: Missing token was rejected.")

        # ---------------------------------------------------------
        # TEST 2: Reject a request with an incorrect token.
        # ---------------------------------------------------------
        response = client.post(
            "/adaptive/feedback",
            json=valid_payload,
            headers={"x-feedback-token": "incorrect-test-token"},
        )

        assert response.status_code == 403, (
            f"Incorrect-token test failed: {response.status_code}"
        )
        assert len(fake_calls) == 0
        print("PASS: Incorrect token was rejected.")

        # ---------------------------------------------------------
        # TEST 3: Reject an unsupported URL scheme.
        # ---------------------------------------------------------
        invalid_scheme_payload = {
            "url": "ftp://example.com",
            "label": 0,
            "source": "manual verification",
        }

        response = client.post(
            "/adaptive/feedback",
            json=invalid_scheme_payload,
            headers={"x-feedback-token": main.FEEDBACK_TOKEN},
        )

        assert response.status_code == 400, (
            f"Invalid-scheme test failed: {response.status_code}"
        )
        assert len(fake_calls) == 0
        print("PASS: Unsupported URL scheme was rejected.")

        # ---------------------------------------------------------
        # TEST 4: Reject a URL without a hostname.
        # ---------------------------------------------------------
        invalid_host_payload = {
            "url": "http:///missing-host",
            "label": 0,
            "source": "manual verification",
        }

        response = client.post(
            "/adaptive/feedback",
            json=invalid_host_payload,
            headers={"x-feedback-token": main.FEEDBACK_TOKEN},
        )

        assert response.status_code == 400, (
            f"Missing-host test failed: {response.status_code}"
        )
        assert len(fake_calls) == 0
        print("PASS: URL without a hostname was rejected.")

        # ---------------------------------------------------------
        # TEST 5: Reject an out-of-range label at schema validation.
        # ---------------------------------------------------------
        invalid_label_payload = {
            "url": "https://example.com",
            "label": 2,
            "source": "manual verification",
        }

        response = client.post(
            "/adaptive/feedback",
            json=invalid_label_payload,
            headers={"x-feedback-token": main.FEEDBACK_TOKEN},
        )

        assert response.status_code == 422, (
            f"Invalid-label test failed: {response.status_code}"
        )
        assert len(fake_calls) == 0
        print("PASS: Invalid label was rejected.")

        # ---------------------------------------------------------
        # TEST 6: Valid token and URL reach only the fake learner.
        # ---------------------------------------------------------
        response = client.post(
            "/adaptive/feedback",
            json=valid_payload,
            headers={"x-feedback-token": main.FEEDBACK_TOKEN},
        )

        assert response.status_code == 200, (
            f"Valid-request test failed: "
            f"{response.status_code} - {response.text}"
        )

        result = response.json()

        assert result["url"] == valid_payload["url"]
        assert result["verification_source"] == valid_payload["source"]
        assert result["test_update"] is True

        # Confirm that exactly one request reached the fake learner
        # and that the expected label was passed to it.
        assert len(fake_calls) == 1
        assert fake_calls[0]["label"] == 0
        assert len(fake_calls[0]["feature_values"]) > 0

        print("PASS: Valid request returned success.")
        print("PASS: URL features and label reached the fake learner.")
        print("PASS: The production learner was not called.")

        print("\nALL FEEDBACK API TESTS PASSED.")

    finally:
        # Restore the real method even if a test fails.
        main.adaptive_service.learn = original_learn


if __name__ == "__main__":
    run_feedback_api_tests()
