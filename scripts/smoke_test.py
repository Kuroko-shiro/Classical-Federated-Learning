"""Smoke test: run scenarios 1 and 2 end-to-end on synthetic data (no torch).

This proves the harness works:
  - scenario 1: FedAvg + FedProx with a homogeneous model on full modalities,
    non-IID (Dirichlet) partition, with Centralized/Local baselines.
  - scenario 2: FedProto with heterogeneous embed widths on full modalities.

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qflbench.engine.simulator import run_experiment  # noqa: E402


BASE = {
    "seed": 0,
    "num_clients": 6,
    "rounds": 15,
    "client_fraction": 1.0,
    # harder task: more classes, weaker per-modality signal, more noise -> FL has
    # a real gap to close vs local/centralized, so the heterogeneity tax is visible.
    "dataset": {"name": "synthetic", "num_classes": 6, "n_per_class": 300,
                "image_dim": 40, "text_dim": 30,
                "image_signal": 0.45, "text_signal": 0.45, "noise": 2.5,
                "subclusters": 5, "cluster_spread": 2.2, "seed": 0},
    "partition": {"name": "dirichlet", "alpha": 0.2},   # stronger non-IID
    "channel": {"name": "classical"},
    "local": {"local_epochs": 2, "lr": 0.05},
}


def scenario_1():
    print("\n===== Scenario 1: same model, same modality (FedAvg) =====")
    cfg = dict(BASE)
    cfg.update({
        "modality_mode": "full",
        "model_hetero": False,
        "model": {"name": "mock", "embed_choices": [16]},
        "strategy": {"name": "fedavg"},
        "with_baselines": True,
        "run_dir": "results/s1_fedavg",
    })
    s = run_experiment(cfg)
    _report(s)

    print("\n===== Scenario 1: same model, same modality (FedProx mu=0.1) =====")
    cfg2 = dict(cfg)
    cfg2["strategy"] = {"name": "fedprox", "mu": 0.1}
    cfg2["with_baselines"] = False
    cfg2["run_dir"] = "results/s1_fedprox"
    _report(run_experiment(cfg2))


def scenario_2():
    # Scenario 2 = same modality, DIFFERENT models. Parameter averaging is invalid,
    # so we use model-heterogeneity methods. We try distillation (FedMD) as the
    # primary method and FedProto as a contrast.
    common = dict(BASE)
    common.update({
        "modality_mode": "full",
        # NOTE: model heterogeneity. In the mock, clients differ by init/seed; the
        # torch backend will instantiate genuinely different encoders. Distillation
        # (FedMD) needs only a shared output space (classes), so unlike FedProto it
        # does NOT require a common embedding dimension.
        "model_hetero": True,
        "model": {"name": "mock", "embed_choices": [16], "share_encoders": False},
    })

    print("\n===== Scenario 2: different model, same modality (FedMD / distillation) =====")
    cfg = dict(common)
    cfg.update({
        "strategy": {"name": "fedmd", "public_size": 300, "distill_epochs": 2,
                     "temperature": 2.0},
        "local": {"local_epochs": 2, "lr": 0.05, "distill_lr": 0.05},
        "with_baselines": True,
        "run_dir": "results/s2_fedmd",
    })
    _report(run_experiment(cfg))

    print("\n===== Scenario 2: contrast — FedProto (prototype) =====")
    cfg2 = dict(common)
    cfg2.update({
        "strategy": {"name": "fedproto", "proto_mu": 1.0},
        "with_baselines": False,
        "run_dir": "results/s2_fedproto",
    })
    _report(run_experiment(cfg2))


def _report(summary):
    hist = summary["history"]
    last = hist[-1]
    g = last.get("global_metrics")
    extra = last.get("extra", {})
    if g:
        print(f"  final GLOBAL: acc={g.get('accuracy'):.3f} "
              f"macro_f1={g.get('macro_f1'):.3f}")
    print(f"  final per-client mean acc={extra.get('mean_accuracy', float('nan')):.3f} "
          f"worst acc={extra.get('worst_accuracy', float('nan')):.3f} "
          f"std={extra.get('std_accuracy', float('nan')):.3f}")
    tot = summary["totals"]
    print(f"  comms total: up={tot['total_uplink_bytes']/1e3:.1f}KB "
          f"down={tot['total_downlink_bytes']/1e3:.1f}KB over {tot['rounds']} rounds")
    if "baselines" in summary:
        b = summary["baselines"]
        cen = b["centralized"]
        loc = b["local_only"]
        print(f"  baseline CENTRALIZED: acc={cen.get('accuracy'):.3f} "
              f"macro_f1={cen.get('macro_f1'):.3f}")
        print(f"  baseline LOCAL-only:  mean acc={loc.get('mean_accuracy', float('nan')):.3f}")
        # FL score: use global accuracy if a global model exists, else per-client mean.
        g_now = last.get("global_metrics")
        fl = g_now.get("accuracy") if g_now else extra.get("mean_accuracy")
        cen_acc = cen.get("accuracy")
        loc_acc = loc.get("mean_accuracy")
        if fl is not None and cen_acc is not None and loc_acc is not None:
            denom = cen_acc - loc_acc
            if abs(denom) > 1e-6:
                frac = (fl - loc_acc) / denom
                print(f"  >> FL gap-filled fraction (FL-Local)/(Central-Local) = {frac:.2f}")


if __name__ == "__main__":
    scenario_1()
    scenario_2()
    print("\nOK: scenarios 1 and 2 ran end-to-end.")
