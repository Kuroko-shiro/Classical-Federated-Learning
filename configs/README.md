# Configs

現状、実験設定は **Python dict** で記述する（`scripts/run_benchmark.py` の
`base_cfg()` / `make_configs()` を参照）。これは依存を最小化し、ローカルで即動く
ようにするための選択。

将来 Hydra に移行する場合は、このディレクトリに以下の構成で YAML を置き、
`registry` のキー（`fedavg`, `fedmd`, `dirichlet`, `synthetic`, `classical` など）を
`name:` フィールドで参照する形にする。dict のキー構造をそのまま YAML に写せる。

```
configs/
  config.yaml            # defaults を束ねる
  scenario/              # s1/s2/s3/s4（modality_mode, model_hetero, share_encoders の組合せ）
  dataset/               # synthetic, iu_xray, rsna
  partition/             # iid, dirichlet, quantity_skew
  strategy/              # fedavg, fedprox, fedproto, fedmd
  model/                 # mock, torch
  channel/               # classical, quantum_stub
  privacy/               # none, dp（Phase 2）
```

## dict config のスキーマ（参考）

```python
cfg = {
    "seed": 0,
    "num_clients": 6,
    "rounds": 15,
    "client_fraction": 1.0,                 # 各ラウンドで選ぶクライアント割合
    "modality_mode": "full",                # full | disjoint | random  (③④で random/disjoint)
    "model_hetero": False,                  # True で各クライアント別アーキ相当
    "dataset":   {"name": "synthetic", ...},
    "partition": {"name": "dirichlet", "alpha": 0.2},
    "model":     {"name": "mock", "embed_choices": [16], "share_encoders": True},
    "strategy":  {"name": "fedavg"},        # fedprox: {"name":"fedprox","mu":0.1}
                                            # fedmd:   {"name":"fedmd","public_size":300,...}
                                            # fedproto:{"name":"fedproto","proto_mu":1.0}
    "channel":   {"name": "classical"},
    "local":     {"local_epochs": 2, "lr": 0.05},
    "with_baselines": True,                 # Centralized/Local 基準線を計算
    "run_dir": "results/myrun",
}
```

### シナリオの作り方（2×2）

| シナリオ | modality_mode | share_encoders | model_hetero | 推奨 strategy |
|---|---|---|---|---|
| ① 同モデル・同モダリティ | full | True | False | fedavg / fedprox |
| ② 異モデル・同モダリティ | full | False | True | fedmd（主）/ fedproto |
| ③ 同モデル・異モダリティ | random/disjoint | False | False | fedavg（head のみ集約） |
| ④ 異モデル・異モダリティ | random/disjoint | False | True | fedmd |
