# Revised Phase 0: classical audit and pathway diagnosis

Updated: 2026-07-16

Phase 0 now has three tracks. It does not introduce a new final method.

- **P0-A**: reproducible evaluation and communication audit
- **P0-M**: diagnose why MIN did not improve HeteroFL
- **P0-N**: diagnose the α=0.1 client-drift collapse

The outcome is a frozen handoff for backbone-wide low-rank/quantized MMFL in
Phase 1 and a shared-update-subspace design requirement for Phase 2.

## Frozen benchmark

| item | canonical value |
|---|---|
| dataset | IU X-ray, manifest 3,337 |
| split | train pool 2,510 / public 200 / test 627 |
| clients | 4 |
| split seed | 0 |
| local validation | 10%, validation seed 0 |
| train seeds | 0, 1, 2 for the specified multi-seed comparisons |
| checkpoint selection | mean-client validation macro-AUROC |
| test policy | one pass after checkpoint selection |
| primary metric | macro-AUROC |
| important secondary metrics | macro-AUPRC, macro-F1 |
| rare labels | bottom quartile by train positive count, frozen before test |

Public data is reserved for public-anchor methods such as FedMD/LOOT and is not
used as validation data.

## Canonical metric policy

For each disease label, an F1 threshold is fitted on validation only and then
held fixed on test:

\[
\tau_k^*=\arg\max_\tau F1_k^{val}(\tau).
\]

Every `test.json` stores both:

- `macro_f1_val_optimized`
- `macro_f1_threshold_0.5`

It also stores macro/micro AUROC, AUPRC and F1, rare-label macro-F1,
bottom-three/worst-label F1, per-label precision/recall/F1/AUPRC, thresholds and
the number of evaluable labels. A label without both positive and negative
examples is NaN and is excluded from a macro average; it is never replaced by
0.5 or 0.

## P0-A: audit commands

Install from the repository root:

```bash
python -m pip install -e '.[torch]'
```

Create environment and data/split audit artifacts:

```bash
python scripts/iu_phase0_audit.py \
  --reports data/indiana_reports.csv \
  --projections data/indiana_projections.csv \
  --images data/images/images_normalized \
  --split splits/iu_split.json \
  --img-cache data/img_cache_224.pt
```

This writes `environment/` and `results/phase0/data_audit/`. Dataset, split and
cache hashes, overlap checks, client counts and label prevalence are recorded.

Canonical runners write:

```text
results/iu/<run>/
├── config.json
├── validation.csv
├── communication.jsonl
├── best_validation.npz
├── test.json
└── diagnostics/
```

Communication records contain logical tensor bytes and canonical tensor-serialized
serialized payload bytes, separated by upload/download, client, round, dtype and
payload type. `qkd_otp` converts serialized encrypted bytes to required OTP key
bits and key-generation time at 10/50/100 kbps and 1 Mbps.

After runs finish, build the registry and summary:

```bash
python scripts/iu_build_registry.py \
  --results-root results/iu --output results/phase0

python scripts/iu_summarize_runs.py \
  --results-root results/iu --output results/phase0/canonical_summary.csv
```

The 627-study legacy reevaluation set is:

- Scenario 1: FedAvg α=100/0.1; homogeneous FedMD α=100
- Scenario 2: HeteroFL α=100/0.1; FedMD/FedProto α=100
- Scenario 3: FedAvg 1:3 α=100
- Scenario 4: HeteroFL 1:3 α=100/0.1; FedMD and FedMD+LOOT 1:3 α=100
- centralized lr=3e-5

Legacy trajectory peak/conv is diagnostic only. Canonical scores always come
from validation-selected checkpoints.

## P0-M: MIN pathway audit

Use Scenario 4 HeteroFL with the following controls:

| ID | embed dims | MIN |
|---|---|---|
| M0 | 128 256 192 320 | off |
| M1 | 320 256 192 128 | off |
| M2 | 320 256 192 128 | on |
| M3 | 320 320 320 320 | on |

Example commands (append the common dataset/split/batch arguments):

```bash
# M0
python scripts/iu_federated_s4.py --method heterofl --mm-ratio 1:3 \
  --alpha 100 --embed-dims 128 256 192 320 --diagnostics ...

# M1
python scripts/iu_federated_s4.py --method heterofl --mm-ratio 1:3 \
  --alpha 100 --embed-dims 320 256 192 128 --diagnostics ...

# M2
python scripts/iu_federated_s4.py --method heterofl --mm-ratio 1:3 \
  --alpha 100 --embed-dims 320 256 192 128 --use-min --diagnostics ...

# M3
python scripts/iu_federated_s4.py --method heterofl --mm-ratio 1:3 \
  --alpha 100 --embed-dims 320 320 320 320 --use-min --diagnostics ...
```

`M2-M1` is the pure MIN effect. The maximum-width client must be multimodal so
the learned MIN can be sliced into narrower clients. MIN is local and excluded
from HeteroFL aggregation, but its state is saved at the validation-selected
checkpoint.

MIN diagnostics record:

- pretraining and per-round gradient/update norm
- zero-gradient rate and activation norm
- true/generated embedding cosine, normalized MSE and norm ratio
- image-only, text-only, true-text, MIN-text and zero-text ablation

Full FedRecon is not a Phase 0 primary method. Two-stage MIN and frozen-global
MIN are follow-up diagnostics only if M0-M3 leave the cause unresolved.

## P0-N: non-IID drift audit

Run the same HeteroFL configuration at α=100 and α=0.1 with `--diagnostics`.
Each round stores update norm, cosine to global update, pairwise update cosine
and aggregation cancellation ratio.

Minimum diagnostic ablations are:

| ID | local epochs | correction |
|---|---:|---|
| N0 | 2 | none |
| N1 | 1 | none |
| N2 | 2 | FedProx, one fixed μ (where model shape permits) |
| N3 | 2 | validation-selected early stopping |

No large hyperparameter sweep is part of Phase 0. Partition reports separate
quantity skew, label skew and positive scarcity. Rare-label and per-client
trajectories determine where the collapse begins.

## Multi-seed scope

Minimum three-seed comparisons in the revised plan are:

- Scenario 2 HeteroFL α=100
- Scenario 2 FedMD α=100
- Scenario 4 HeteroFL 1:3 α=100
- Scenario 4 FedMD 1:3 α=100

Scenario 2/4 HeteroFL α=0.1 and Local α=0.1 should be added when time permits.
This replaces the former interpretation that every diagnostic cell must be run
at all three seeds. Diagnostic M/N ablations may first be run at seed 0; confirm
the selected conclusion at additional seeds if it changes the Phase 1 choice.

## Definition of Done

- package import and environment reconstruction succeed
- dataset/split/cache hashes and split integrity are frozen
- all important runs are registered as canonical, legacy or invalid
- all stated legacy comparisons use the same 627-study test set
- validation-selected evaluation and validation-fitted F1 thresholds work
- macro-AUROC/AUPRC/F1 and rare/per-label metrics are present
- serialized upload/download bytes and QKD accounting are present
- centralized C and canonical HeteroFL/FedMD/Local scores are fixed
- M0-M3 separate MIN from width reassignment and slicing
- MIN update path and modality ablation identify at least one failure hypothesis
- α=100/0.1 update drift and local-epoch-1 diagnostic are compared
- the rare-label collapse location is identified
- `docs/phase0_handoff.md` records the Phase 1 multimodal and non-IID decision

Only after these checks is Phase 0 tagged and frozen.
