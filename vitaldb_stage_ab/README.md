# VitalDB Stage A/B audit

Execution bundle for the VitalDB benchmark Go/No-Go preflight.

Run locally from the repository root:

```bash
bash vitaldb_stage_ab/run_stage_ab.sh
```

Outputs are written to `artifacts/vitaldb_stage_ab/`.

This stage only audits access, track availability, a small raw-track smoke download, and synchronized signal loading. It does not run Gate 0 training or federated learning.
