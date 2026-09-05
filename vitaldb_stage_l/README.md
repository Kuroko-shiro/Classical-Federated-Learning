# VitalDB Stage L — conditional homogeneous FedAvg

This stage is gated by the fixed P1 60-second sensitivity result from workflow run `33937156959`.

Continuation rule, frozen before observing that result:

- mean ECG+PLETH AUPRC gain over the best unimodal model >= 0.02;
- positive fusion gain in at least 2 of 3 fixed seeds;
- signal-QC cohort >= 2,500 cases.

Only when all conditions are met does the downstream workflow run the minimal homogeneous K=8 full-parameter FedAvg plumbing test. Client construction is subject-disjoint and approximately IID; non-IID and missing-modality stress are intentionally deferred.

The preregistered 20-second centralized Gate 0 remains YELLOW and is not retroactively overwritten by this sensitivity analysis.
