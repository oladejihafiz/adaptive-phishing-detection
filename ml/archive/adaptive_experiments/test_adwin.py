import numpy as np
from river.drift import ADWIN

rng = np.random.default_rng(42)

# First phase: low prediction-error rate (5%)
phase_1 = rng.binomial(1, 0.05, 3000)

# Second phase: higher prediction-error rate (40%)
phase_2 = rng.binomial(1, 0.40, 3000)

error_stream = np.concatenate([phase_1, phase_2])

detector = ADWIN(delta=0.002)

detections = []

for i, error in enumerate(error_stream, start=1):

    detector.update(int(error))

    if detector.drift_detected:
        detections.append(i)
        print(f"Drift detected at sample {i}")

print("\nTotal samples:", len(error_stream))
print("Actual change introduced after sample: 3000")
print("Detected drift positions:", detections)