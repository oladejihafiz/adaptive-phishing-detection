
# Import the ADWIN drift detector.
from river.drift import ADWIN

# Import deque to simulate a recent-example buffer.
from collections import deque


# Create a fresh detector for this test only.
# It does not affect the running API's detector.
detector = ADWIN(delta=0.002)

# Create a buffer that holds at most 128 recent examples.
recent_examples = deque(maxlen=128)

# Count the number of simulated replay operations.
replay_updates = 0

# Record drift detections.
drift_events = []


# Simulate 3,000 mostly correct predictions followed by
# 3,000 mostly incorrect predictions.
#
# Phase 1: one error in every 20 predictions (5%).
# Phase 2: eight errors in every 20 predictions (40%).
errors = (
    [1 if i % 20 == 0 else 0 for i in range(3000)]
    + [1 if i % 20 < 8 else 0 for i in range(3000)]
)


# Process the simulated prediction-error stream.
for index, error in enumerate(errors, start=1):

    # Feed each simulated error to ADWIN.
    detector.update(error)

    # Simulate storing a verified example.
    recent_examples.append({
        "example_number": index,
        "error": error
    })

    # Respond if ADWIN detects a change.
    if detector.drift_detected:

        # Select at most 32 recent examples.
        examples_to_replay = list(recent_examples)[-32:]

        # Count the examples selected for replay.
        # This is a simulation, not actual model training.
        replayed_count = len(examples_to_replay)
        replay_updates += replayed_count

        # Record the detected drift.
        drift_events.append({
            "sample": index,
            "replayed_examples": replayed_count
        })


# Display the test results.
print("Total simulated samples:", len(errors))
print("Drift events detected:", len(drift_events))
print("Total simulated replay updates:", replay_updates)
print("Drift event details:", drift_events)
