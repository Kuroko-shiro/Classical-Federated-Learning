"""Minimal sanity tests for the harness. Run: pytest tests/ -q

These check that the protocol wiring works and that the consistent metrics
(analysis.py) compute without error on each strategy, including the ones with no
single global model (FedProto/FedMD). They are smoke-level, not accuracy gates.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qflbench.engine.simulator import run_experiment
from qflbench.metrics.analysis import summarize_run, rounds_to_target


def _base(seed=0, rounds=5):
    return {
        "seed": seed, "num_clients": 4, "rounds": rounds, "client_fraction": 1.0,
        "dataset": {"name": "synthetic", "num_classes": 4, "n_per_class": 120,
                    "image_dim": 16, "text_dim": 12, "subclusters": 3,
                    "cluster_spread": 1.5, "noise": 1.5, "seed": seed},
        "partition": {"name": "dirichlet", "alpha": 0.5},
        "channel": {"name": "classical"},
        "local": {"local_epochs": 1, "lr": 0.05},
    }


def test_fedavg_runs_and_has_global_metrics():
    cfg = _base()
    cfg.update({"modality_mode": "full", "model_hetero": False,
                "model": {"name": "mock", "embed_choices": [12], "share_encoders": True},
                "strategy": {"name": "fedavg"}, "with_baselines": True,
                "run_dir": "results/test_fedavg"})
    s = run_experiment(cfg)
    assert len(s["history"]) == 5
    assert s["history"][-1]["global_metrics"] is not None
    row = summarize_run(s, target=0.5)
    assert row["final_accuracy"] is not None
    assert "baselines" in s


def test_fedmd_runs_without_global_model():
    cfg = _base()
    cfg.update({"modality_mode": "full", "model_hetero": True,
                "model": {"name": "mock", "embed_choices": [12], "share_encoders": False},
                "strategy": {"name": "fedmd", "public_size": 60, "distill_epochs": 1},
                "local": {"local_epochs": 1, "lr": 0.05, "distill_lr": 0.05},
                "with_baselines": False, "run_dir": "results/test_fedmd"})
    s = run_experiment(cfg)
    # no single global model -> per-client metrics drive the analysis
    assert s["history"][-1]["global_metrics"] is None
    row = summarize_run(s, target=0.5)
    assert row["final_accuracy"] is not None  # falls back to per-client mean


def test_fedproto_runs():
    cfg = _base()
    cfg.update({"modality_mode": "full", "model_hetero": True,
                "model": {"name": "mock", "embed_choices": [12], "share_encoders": False},
                "strategy": {"name": "fedproto", "proto_mu": 1.0},
                "with_baselines": False, "run_dir": "results/test_fedproto"})
    s = run_experiment(cfg)
    assert s["history"][-1]["per_client_metrics"]


def test_communication_is_accounted():
    cfg = _base()
    cfg.update({"modality_mode": "full", "model_hetero": False,
                "model": {"name": "mock", "embed_choices": [12], "share_encoders": True},
                "strategy": {"name": "fedavg"}, "with_baselines": False,
                "run_dir": "results/test_comms"})
    s = run_experiment(cfg)
    assert s["totals"]["total_uplink_bytes"] > 0
    assert s["totals"]["total_downlink_bytes"] > 0


def test_modality_subset_scenario3_runs():
    # scenario 3 wiring: modality subsets + head-only sharing
    cfg = _base()
    cfg.update({"modality_mode": "random", "model_hetero": False,
                "model": {"name": "mock", "embed_choices": [12], "share_encoders": False},
                "strategy": {"name": "fedavg"}, "with_baselines": False,
                "run_dir": "results/test_s3"})
    s = run_experiment(cfg)
    # clients should hold differing modality counts
    mods = s["client_modalities"]
    sizes = {len(v) for v in mods.values()}
    assert len(s["history"]) == 5
    assert sizes  # at least ran


if __name__ == "__main__":
    test_fedavg_runs_and_has_global_metrics()
    test_fedmd_runs_without_global_model()
    test_fedproto_runs()
    test_communication_is_accounted()
    test_modality_subset_scenario3_runs()
    print("all tests passed")


def test_scenario3_loot_and_min_run():
    # scenario 3 (Saha-faithful): mixed modalities, full model, LOOT + MIN-lite
    base = _base(rounds=4)
    # LOOT
    c = dict(base)
    c.update({"modality_mode": "mixed:2", "model_hetero": False,
              "model": {"name": "mock", "embed_choices": [12],
                        "share_encoders": True, "all_modalities": True},
              "strategy": {"name": "loot", "public_size": 60,
                           "align_epochs": 1, "align_lr": 0.01},
              "with_baselines": False, "run_dir": "results/test_s3_loot"})
    s = run_experiment(c)
    assert s["history"][-1]["global_metrics"] is not None
    # MIN-lite imputation path
    c2 = dict(base)
    c2.update({"modality_mode": "mixed:2", "model_hetero": False,
               "model": {"name": "mock", "embed_choices": [12],
                         "share_encoders": True, "all_modalities": True},
               "strategy": {"name": "fedavg"}, "min_imputation": True,
               "with_baselines": False, "run_dir": "results/test_s3_min"})
    s2 = run_experiment(c2)
    assert len(s2["history"]) == 4
