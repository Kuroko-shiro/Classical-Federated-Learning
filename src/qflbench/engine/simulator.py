"""Scenario builder + simulator + centralized/local baselines.

Turns a plain config dict into a running experiment. This is what scripts/run.py
and the smoke test call. It also computes the two reference lines from the design
doc: Centralized (skyline) and Local-only (floor), so FL can be scored as
    (FL - Local) / (Centralized - Local).

Scenario wiring (the 2x2):
  - model heterogeneity  : per-client model_name / embed widths (here via ctx.extra)
  - modality heterogeneity: modality assignment mode ('full' vs 'random'/'disjoint')

  scenario 1: model=homo,   modality=full
  scenario 2: model=hetero, modality=full
  scenario 3: model=homo,   modality=subset
  scenario 4: model=hetero, modality=subset
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core.registry import CHANNELS, DATASETS, MODEL_FACTORIES, PARTITIONERS, STRATEGIES
from ..core.types import ClientContext
from ..data.base import assign_modalities
from ..metrics.classification import classification_metrics
from ..metrics.logger import RunLogger
from ..utils.seed import seed_everything
from .client import Client
from .server import FederatedEngine

# Import side-effect: register implementations.
from ..data import synthetic as _synthetic          # noqa: F401
from ..data import partition as _partition           # noqa: F401
from ..models import mock_models as _mock            # noqa: F401
from ..comm import classical as _classical           # noqa: F401
from ..strategies import fedavg as _fedavg           # noqa: F401
from ..strategies import fedproto as _fedproto       # noqa: F401
from ..strategies import fedmd as _fedmd             # noqa: F401
from ..strategies import loot as _loot               # noqa: F401


def _build_clients(cfg, dataset, rng) -> List[Client]:
    num_clients = cfg["num_clients"]

    # statistical heterogeneity: partition the TRAIN pool
    part_cls = PARTITIONERS.get(cfg["partition"]["name"])
    part_kwargs = {k: v for k, v in cfg["partition"].items() if k != "name"}
    partitioner = part_cls(**part_kwargs)
    train_labels = dataset.labels_for_partition()
    local_parts = partitioner.partition(train_labels, num_clients, rng)
    # map back to global indices
    train_pool = dataset.train_pool_index
    client_train = {c: train_pool[local_parts[c]] for c in range(num_clients)}
    dataset.attach_partition(client_train)

    # modality heterogeneity: which modalities each client holds
    mod_assign = assign_modalities(
        num_clients, dataset.modalities, cfg["modality_mode"], rng
    )

    # model heterogeneity: vary embed widths across clients if requested
    factory_cls = MODEL_FACTORIES.get(cfg["model"]["name"])
    factory = factory_cls(**{k: v for k, v in cfg["model"].items()
                             if k not in ("name", "embed_choices")})

    hetero_models = cfg.get("model_hetero", False)
    embed_choices = cfg["model"].get("embed_choices", [16])

    clients: List[Client] = []
    for c in range(num_clients):
        extra = {}
        # NOTE: embed_dim must stay equal across clients for head aggregation in
        # FedAvg/FedProx to be valid; for FedProto/FedMD it may differ. We vary the
        # *encoder hidden* conceptually; embed_dim varies only for proto/distill.
        if hetero_models:
            extra["embed_dim"] = int(embed_choices[c % len(embed_choices)])
        else:
            extra["embed_dim"] = int(embed_choices[0])
        ctx = ClientContext(
            client_id=c,
            model_name=f"{cfg['model']['name']}_{c}" if hetero_models else cfg["model"]["name"],
            modalities=mod_assign[c],
            num_train=len(client_train[c]),
            num_classes=dataset.num_classes,
            extra=extra,
        )
        model = factory.build(ctx, dataset)
        clients.append(Client(ctx, model, dataset))
    return clients


def run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rng = seed_everything(cfg.get("seed", 0))

    # dataset
    ds_cls = DATASETS.get(cfg["dataset"]["name"])
    dataset = ds_cls(**{k: v for k, v in cfg["dataset"].items() if k != "name"})

    clients = _build_clients(cfg, dataset, rng)

    # ---- MIN-lite (Saha Sec. 6): pre-FL modality imputation -------------------
    # Train a linear translator src->tgt on the FIRST multimodal client's local
    # data (paper: MIN is trained in one multimodal client, frozen, then used in
    # unimodal clients). Clients missing tgt get the imputer; data access then
    # generates x[tgt] = x[src] @ W + b, converting incongruent -> pseudo-congruent
    # MMFL before any federated round (no per-round overhead, as in the paper).
    if cfg.get("min_imputation", False):
        mods = list(dataset.modalities)
        if len(mods) >= 2:
            src, tgt = mods[0], mods[1]
            donor = next((c for c in clients if set(mods) <= set(c.ctx.modalities)), None)
            if donor is not None:
                d = dataset.get_split(donor.ctx.client_id, "train", mods)
                Xs = d["x"][src].astype(np.float64)
                Xt = d["x"][tgt].astype(np.float64)
                Xs1 = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
                lam = float(cfg.get("min_ridge", 1e-2))
                A = Xs1.T @ Xs1 + lam * np.eye(Xs1.shape[1])
                Wb = np.linalg.solve(A, Xs1.T @ Xt)
                W, b = Wb[:-1].astype(np.float32), Wb[-1].astype(np.float32)
                for c in clients:
                    if tgt not in c.ctx.modalities:
                        c.imputer = (src, tgt, W, b)

    # strategy
    strat_cls = STRATEGIES.get(cfg["strategy"]["name"])
    factory_cls = MODEL_FACTORIES.get(cfg["model"]["name"])
    factory = factory_cls(**{k: v for k, v in cfg["model"].items()
                             if k not in ("name", "embed_choices")})
    strat_kwargs = {k: v for k, v in cfg["strategy"].items() if k != "name"}
    strategy = strat_cls(factory=factory, dataset=dataset, **strat_kwargs)
    strategy.initialize([c.ctx for c in clients])

    # channel
    chan_cls = CHANNELS.get(cfg["channel"]["name"])
    channel = chan_cls(**{k: v for k, v in cfg["channel"].items() if k != "name"})

    engine = FederatedEngine(
        strategy=strategy,
        channel=channel,
        clients=clients,
        rng=rng,
        client_fraction=cfg.get("client_fraction", 1.0),
        local_cfg=cfg.get("local", {}),
    )

    logger = RunLogger(cfg.get("run_dir", "results/run"), cfg)
    for r in range(cfg["rounds"]):
        rm = engine.run_round(r)
        logger.log_round(rm)
    summary = logger.finalize()

    # record which modalities each client held (for modality-breakdown analysis;
    # meaningful in scenarios 3/4, degenerate in 1/2).
    summary["client_modalities"] = {c.ctx.client_id: list(c.ctx.modalities)
                                    for c in clients}

    # baselines. NOTE: local_only_baseline retrains client models in place, so it
    # MUST run after the federated history is finalized (it is). It does not affect
    # the saved federated results.
    if cfg.get("with_baselines", False):
        summary["baselines"] = {
            "centralized": centralized_baseline(cfg, dataset, rng),
            "local_only": local_only_baseline(clients),
        }
    return summary


# --------------------------------------------------------------------------- #
# Baselines (reference lines)
# --------------------------------------------------------------------------- #
def centralized_baseline(cfg, dataset, rng) -> Dict[str, float]:
    """Skyline: train one model on the pooled TRAIN data over all modalities."""
    full_mods = dataset.modalities
    ctx = ClientContext(
        client_id=-100, model_name="central", modalities=full_mods,
        num_train=0, num_classes=dataset.num_classes,
        extra={"embed_dim": cfg["model"].get("embed_choices", [16])[0]},
    )
    factory_cls = MODEL_FACTORIES.get(cfg["model"]["name"])
    factory = factory_cls(**{k: v for k, v in cfg["model"].items()
                             if k not in ("name", "embed_choices")})
    model = factory.build(ctx, dataset)

    # pooled train = all client train indices = the whole train pool
    pool = dataset.train_pool_index
    train = {"x": {m: _gather(dataset, m, pool) for m in full_mods},
             "y": _labels(dataset, pool)}
    model.local_train(train, epochs=cfg.get("central_epochs", 5),
                       lr=cfg.get("local", {}).get("lr", 0.05))
    test = dataset.get_split(0, "test", full_mods)
    return model.evaluate(test)


def local_only_baseline(clients: List[Client]) -> Dict[str, float]:
    """Floor: each client trains only on its own data; report macro-averaged test."""
    per = {}
    for client in clients:
        client.model.local_train(client.train_data(), epochs=5, lr=0.05)
        per[client.ctx.client_id] = client.model.evaluate(client.test_data())
    from ..metrics.classification import aggregate_client_metrics
    agg = aggregate_client_metrics(per)
    return agg


def _gather(dataset, modality, idx):
    # reach into the synthetic dataset's arrays (works for the mock; torch loaders
    # would implement an equivalent pooled accessor).
    return dataset._x[modality][idx]


def _labels(dataset, idx):
    return dataset._y[idx]
