# VitalDB labeling specification — audit freeze v0.1

## Label source

- Source: `Solar8000/ART_MBP` only.
- ART waveform is not a P1 model input.
- Hypotension event: MAP < 65 mmHg sustained for at least 60 seconds.
- MAP values outside [20, 250] mmHg are treated as invalid for this audit.
- A gap > 5 seconds between valid MAP measurements breaks event continuity.
- Event duration is conservatively measured from first to last below-threshold timestamp.

## Prediction framing for Gate 0

- Observation window: 80 seconds.
- First 60 seconds: hypotension-free baseline check.
- Final 20 seconds: ECG/PLETH model input segment.
- Prediction window: following 5 minutes.
- Slack window: following 1 minute.
- Positive: a qualifying hypotension event begins in the prediction window.
- Skip: a qualifying event is active in the observation window.
- Negative: no qualifying event occurs in observation, prediction, or slack windows.
- Skip rather than label negative when an event begins in the slack window.

## Sampling

- Do not enumerate every overlapping sample for training.
- Window-sampling seed will be saved.
- Minimum separation between retained windows from the same case: 60 seconds.
- Cap retained windows per case before Gate 0 training; exact cap will be selected after the natural candidate-window prevalence audit.
- Final test evaluation retains natural class prevalence; balancing/weighting is training-only.

## Status

This file freezes the event and temporal framing. Exact per-case window cap is intentionally deferred until candidate-window prevalence is measured; models must not be compared across different framing rules.
