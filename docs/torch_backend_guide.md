# Torch backend 実装ガイド

`models/torch_models.py` の stub を、実環境（torch あり）で埋めるための手引き。
**`MockMultimodalModel`（`models/mock_models.py`）と完全に同じインターフェース**を
実装すれば、ハーネス・戦略・評価・通信計測はそのまま再利用できる。config の
`model.name` を `"mock"` → `"torch"` に変えるだけで差し替わる。

## 実装すべきメソッド（ModelBackend 契約）

`core/interfaces.py` の `ModelBackend` を参照。必須は以下。

| メソッド | 役割 | torch 実装のヒント |
|---|---|---|
| `get_parameters(only_shared)` | パラメータ取得（numpy） | `state_dict()` → `.detach().cpu().numpy()`。`only_shared` のとき `shared_parameter_keys()` で絞る |
| `set_parameters(params, only_shared)` | パラメータ設定 | numpy → tensor、`load_state_dict(strict=False)` |
| `shared_parameter_keys()` | 共有するキー集合 | ①は全キー、③④は head（+共有 fusion）のみ。`share_encoders` で分岐 |
| `local_train(data, epochs, lr, proximal_mu, global_params, extra)` | ローカル学習 | 標準学習ループ。FedProx 項 = `mu/2 * Σ‖w-w_g‖²`（shared params のみ）。FedProto 項 = `extra["global_prototypes"]` への埋め込み引き寄せ |
| `class_prototypes(data)` | クラス平均埋め込み | エンコーダ+fusion を forward し、クラスごとに平均 |
| `predict_logits(x)` | logit 出力（no grad） | FedMD の公開データ予測 |
| `distill(x_public, soft_targets, epochs, lr, temperature)` | ソフトターゲット蒸留 | KD: `KL(softmax(student/T), softmax(teacher/T))`。FedMD のアライメント |
| `evaluate(data)` | 指標計算 | `metrics.classification.classification_metrics` をそのまま使える |
| `embedding_dim()` | 共有埋め込み次元 | プロトタイプ整合のため全クライアントで一致させる |

## IU X-ray 向け推奨アーキテクチャ

```
image encoder: DenseNet121 / 医用 CNN → global pool → Linear(embed_dim)
text encoder : ClinicalBERT / CXR-BERT の CLS → Linear(embed_dim)
fusion       : mean（部分集合フレンドリ） or 小さな attention（共有・集約可能）
head         : Linear(embed_dim → num_classes)  ← 共有
```

## 重要な制約（モックで実証済みの罠）

1. **プロトタイプ/蒸留法でも `embed_dim` は全クライアントで一致**させる
   （FedProto はプロトタイプ平均に、評価の最近傍に必要）。違ってよいのは
   エンコーダの深さ・幅と、集約されない head。
2. **③で「同じモデル」= head のみ共有**（`share_encoders=False`）。
   モダリティ別エンコーダはローカル保持。これを誤ると global モデルが
   未学習エンコーダで壊れる（モックで確認済み）。
3. **シード**：factory で `torch.manual_seed(seed)`。

## 検証方法

torch backend 実装後、まず合成データの `feature` モードで
`scripts/run_benchmark.py` を走らせ、モックと同等の傾向（①が②を上回る、
gap-filled が①>②）が出ることを確認してから IU X-ray に進む。
