# QFL Multimodal Benchmark — Classical Baseline Harness

マルチモーダル医療データ（IU X-ray, 14ラベル multilabel）向け量子連合学習（QFL）の
**古典ベンチマーク基盤**。4シナリオ（モデル同種/異種 × モダリティ同種/異種）を
古典FLで実装し、**異種性のコストを一貫した指標で定量化**する。

長期目標：古典で「穴」になる箇所を特定し、そこに QFL の優位（情報理論的安全性）を
導く。本リポジトリはその**比較基準線**と、**量子コンポーネントの差し替え点**を提供する。

> 本フェーズで量子は入れない。量子の効果を測る「物差し」を先に作るための設計。

**現状（2026-07-15）：Phase 0（評価基盤の固定）を実装済み。**
新規ランは、client-local validation による checkpoint 選択、共通627件 test の一回評価、
upload/download・client別の実byte計測、全args・split hash・git hashの保存を行う。
旧ランの数値は評価プロトコルが異なるため historical 扱いとし、Phase 0規約で再取得する。

詳細な実験順序と完了条件は [`docs/phase0_protocol.md`](docs/phase0_protocol.md) を参照。

---

## 環境

```bash
conda activate qfl-torch
export PYTORCH_ENABLE_MPS_FALLBACK=1     # Apple Silicon (MPS) 必須
cd qfl-benchmark
```

長時間ランは必ず `caffeinate -dimsu` を前置する（スリープでランが死ぬ）。

### 画像キャッシュ（初回のみ）

全ランは事前計算した画像テンソルキャッシュを前提とする。

```bash
python scripts/iu_cache_images.py \
    --projections data/indiana_projections.csv \
    --images data/images/images_normalized \
    --img-size 224 --out data/img_cache_224.pt
```

以降 `--img-cache data/img_cache_224.pt` を全コマンドに渡す。

---

## Phase 0 固定プロトコル

| 項目 | 固定値 |
|---|---|
| frozen split | `splits/iu_split.json` |
| train pool | 2,510件 |
| validation | client partitionごとに10%を固定抽出（既定seed `0`） |
| test | 627件すべて |
| checkpoint選択 | validation macro-AUROC 最大 |
| test利用 | 選択済みcheckpointに対して最後に1回のみ |
| 通信量 | logical/serialized payload bytesをupload/download・client別に記録 |

旧実装の「test trajectoryからpeakを選ぶ」「③だけ627件、他は300件」問題は解消済み。

---

## ランナーと手法

| ランナー | シナリオ | `--method` の選択肢 |
|---|---|---|
| `iu_federated.py` | ①（統計的異種）②（モデル異種） | `fedavg` `fedprox` `fedmd` `heterofl` `fedproto` |
| `iu_federated_s3.py` | ③（モダリティ不一致・モデル同種） | `uniml` `fedavg` `min` `loot` |
| `iu_federated_s4.py` | ④（複合異種） | `heterofl` `fedmd` `fedmd_loot` |
| `iu_baselines.py` | 基準線 | `--mode centralized` / `--mode local` |

補助スクリプト：`iu_make_split.py`（凍結split生成）、`iu_cache_images.py`（画像キャッシュ）、
`iu_comm_estimate.py`（学習せずに FedAvg 系の通信量を解析的に算出）、
`iu_smoke_test.py`、`iu_plot_s1.py`。

`scripts/run_benchmark.py` と `scripts/smoke_test.py` は**合成データ時代の遺物**で、
現在の実データ実験には使わない（ハーネス検証用に残置）。

### モデル異種（`vary_embed`）の実体

クライアント毎に `embed_dim ∈ [128, 256, 192, 320]`（cid 順、昇順ではない）。
幅が変わるのは **`img.proj` / `txt.proj` / `head` の3層のみ**で、
バックボーン（ResNet-50 23.5M + BERT-base 109.5M）は全員同一形状。
幅可変部は全パラメタの **0.27〜0.68%**。したがって本ベンチの「モデル異種」は
**幅異種（same-family width heterogeneity）**であり、真のアーキ異種ではない。

---

## 実行例

### ① 統計的異種（FedAvg, 強非IID）

```bash
caffeinate -dimsu python scripts/iu_federated.py \
    --reports data/indiana_reports.csv \
    --projections data/indiana_projections.csv \
    --images data/images/images_normalized \
    --img-cache data/img_cache_224.pt --split splits/iu_split.json \
    --scenario 1 --method fedavg --clients 4 --alpha 0.1 \
    --rounds 40 --train-subset 2510 --test-subset 627 \
    --val-fraction 0.1 --seed 0 \
    --local-epochs 2 --batch 8 --num-workers 2 \
    2>&1 | tee logs/s1_fedavg_a0.1.log
```

### ② モデル異種（HeteroFL）

```bash
caffeinate -dimsu python scripts/iu_federated.py \
    --reports data/indiana_reports.csv \
    --projections data/indiana_projections.csv \
    --images data/images/images_normalized \
    --img-cache data/img_cache_224.pt --split splits/iu_split.json \
    --scenario 2 --method heterofl --clients 4 --alpha 100.0 \
    --rounds 40 --train-subset 2510 --test-subset 627 \
    --val-fraction 0.1 --seed 0 \
    --local-epochs 2 --batch 8 --num-workers 2 \
    2>&1 | tee logs/s2_heterofl_a100.log
```

FedProto は `--method fedproto --proto-dim 128 --proto-mu 0.1 --proto-warmup 3`。

### ③ モダリティ不一致

```bash
caffeinate -dimsu python scripts/iu_federated_s3.py \
    --reports data/indiana_reports.csv \
    --projections data/indiana_projections.csv \
    --images data/images/images_normalized \
    --img-cache data/img_cache_224.pt --split splits/iu_split.json \
    --method loot --mm-ratio 1:3 --alpha 0.1 \
    --clients 4 --rounds 40 \
    --train-subset 2510 --test-subset 627 --val-fraction 0.1 --seed 0 \
    --local-epochs 2 --batch 8 --num-workers 2 \
    2>&1 | tee logs/s3_loot_1-3_a0.1.log
```

### ④ 複合異種

`scripts/iu_federated_s4.py --method heterofl|fedmd|fedmd_loot --mm-ratio 1:3 --alpha 100.0`（他は同様）

### 基準線

```bash
# Local 下限（クライアント毎に単独学習。gap_filled の分母）
caffeinate -dimsu python scripts/iu_baselines.py \
    --reports data/indiana_reports.csv \
    --projections data/indiana_projections.csv \
    --images data/images/images_normalized \
    --img-cache data/img_cache_224.pt --split splits/iu_split.json \
    --mode local --clients 4 --alpha 0.1 --vary-embed \
    --epochs 80 --eval-every 2 --train-subset 2510 --test-subset 627 \
    --val-fraction 0.1 --seed 0 --batch 8
```

**Local は α ごとに取り直す必要がある**（Dirichlet 分割が α で変わるため）。
α の効果は +0.117、幅異種の効果は +0.027 と実測されており、**使い回し不可**。

Centralized 上限は `iu_baselines.py --mode centralized --lr 3e-5` で取得し、
validation選択済みcheckpointのtest macro-AUROCを `C` とする。

---

## 評価プロトコル（全シナリオ共通・変更禁止）

| 項目 | 値 |
|---|---|
| 分割 | `splits/iu_split.json`（凍結。train_pool / public / test / by_alpha） |
| ラウンド | 40 |
| ローカル | `--local-epochs 2`, batch 8, AdamW lr 1e-4（毎ラウンド再初期化） |
| Local基準線 | client-local validationで各clientのcheckpointを選び、test値を平均 |
| 主指標 | macro-AUROC |
| 主報告 | validation-selected checkpointのtest値 |
| 補助報告 | validation trajectoryのpeak / conv（R14–39平均） |
| durable判定 | Δ = peak − conv。Δ≈0.01 は堅牢、Δ=0.05〜0.15 は一過性 |
| 異種モデルの評価 | 集約後に各自の幅で再参加 → **per-client 平均** |

**上限の定義**：centralized、lr=`3e-5`、40 epochsを完走し、validation macro-AUROCで
選択したcheckpointを共通testで一度評価する。旧 `C=0.9748` は撤回済み。

```
gap_filled = (FL_peak − Local) / (C − Local)
```

---

## Historical results（旧評価プロトコル、Phase 0再取得前）

以下は研究仮説を作った既存値だが、test subset不一致とtest-peak選択を含む。
最終報告値には使用せず、Phase 0規約で再取得した値に置き換える。

| セル | 手法 | 経路 | peak | conv(R14–39) | Local | gap_filled | 通信/R |
|---|---|---|---|---|---|---|---|
| ① α=0.1 | FedAvg | param | 0.8005 | 0.7016 | 0.7433 | +24.7% | — |
| ① α=0.1 | FedProx | param | 0.7924 | 0.7394 | 0.7433 | +21.2% | — |
| ① α=100 | FedAvg | param | **0.9748** | 0.9608 | — | （経験上限） | — |
| ① α=100 | FedMD-homo | logit | 0.8636 | 0.7433 | — | （対照） | 0.09 MB |
| ② α=100 | FedMD | logit | 0.8929 | 0.7412 | 0.8880 | +5.7% | 0.09 MB |
| ② α=100 | **HeteroFL** | param入れ子 | **0.9675** | **0.9570** | 0.8880 | **+91.6%** | 4,278 MB |
| ② α=100 | FedProto | 表現 | 0.8574 | 0.7027 | 0.8880 | −35.2% | 0.086 MB |
| ③ α=100 | FedAvg (1:3) | param | 0.9350 | 0.9215 | 0.7609 | +81.4%※ | 4,281 MB |
| ④ α=100 | FedMD (1:3) | logit | 0.7742 | 0.7168 | 0.7882 | −7.5% | 0.09 MB |
| ④ α=100 | FedMD+LOOT (1:3) | logit | 0.7239 | 0.7061 | 0.7882 | −34.5% | 0.09 MB |

※③は分子627件・分母300件の混合評価（上記の既定値問題）。要注記。

### 2×2：帰属の分離（α=100, peak AUROC）

| | パラメタ経路 | ロジット経路 | 経路差 |
|---|---|---|---|
| **モデル同種** | FedAvg 0.9748 | FedMD-homo 0.8636 | **+0.1111** |
| **モデル異種（幅）** | HeteroFL 0.9675 | FedMD 0.8929 | +0.0746 |
| **異種化の影響** | −0.0073 | −0.0293 | |

**縦（経路効果 −0.111）が主効果、横（異種化 −0.007〜−0.029）は二次。**
決定的なのは FedMD-homo (0.864) < FedMD-異種 (0.893) が成立している点で、
「崖は異種性そのもの」という解釈は棄却される。
**崖は、異種性が密なパラメタ経路を封じ、ロジット経路への退避を強制することの帰結。**

### 共有物の粒度スペクトラム

```
密 ◄──────────────────────────────────────────► 疎
全パラメタ      入れ子パラメタ      ロジット        プロトタイプ
(FedAvg)       (HeteroFL)        (FedMD)        (FedProto)
0.9748         0.9675            0.8929          0.8574→0.703
—              +91.6%            +5.7%           −35.2%
—              4,278 MB/R        0.090 MB/R      0.086 MB/R
```

精度は密ほど高い。ただし FedProto が FedMD より悪いのは、疎さに加えて
プロトタイプの pull が**能動的に有害**だから（**疎さと有害性は別軸**）。
payload は密ほど重い（約 48,000 倍）。この緊張が QFL-② の設計問題そのもの。

---

## 旧ランに残る既知の問題

1. **`--test-subset` の既定値差**（上記）。③のみ627件評価。
2. **③のCSVファイル名から mm_ratio が読めない**：
   `ratio_tag = mm_ratio if method=="fedavg" else "uni"` のため、
   `s3_loot_uni_*.csv` は「uniモダリティ」ではなく単に fedavg 以外という意味。
   実値は CSV の `mm_ratio` 列を見ること。
3. **③の logit 系（min/loot/uniml）の `comm_mb` はバグ**：パラメタサイズを記録している。
   `iu_comm_estimate.py` か解析式で差し替える。
   ③FedAvg（4,281 MB/R）と HeteroFL 実測（4,278 MB/R）は相互検証済みで正常。
4. **`iu_baselines.py --lr-decay` の既定は `none`**。cosine を使うなら明示すること。
   過去の Local ランがどちらだったかは記録が残っていない。
5. **min/loot に重複ランがある**（canonical 未確定）。git 履歴が無い時期のもので、
   序盤カーブでコード版を特定し、同一版で統一採用する必要がある。
6. **FedProto は multilabel で構造的に機能しない**（−35.2%）。μ→0 の極限が
   「純ローカル − 通信」なので便益領域が存在しない。調整で救えないため、対照として残す。

### Phase 0で実装済みの再現性規律

- 全ランで**全パラメタを明示**（既定値に依存しない）。
- 全runnerがrun directoryに `config.json`、`validation.csv`、`communication.jsonl`、
  `best_validation.npz`、`test.json` を保存する。
- 決定性は検証済み（同一幅・同一シード・同一分割で AUROC が小数点4桁まで一致）。
- 学習は `--test-subset` に依存しない（評価のみ）。

---

## ディレクトリ構成

```
src/qflbench/
  core/          types / interfaces / registry（差し替え可能な継ぎ目）
  data/
    base.py          ModalitySpec, モダリティ割当
    partition.py     IID / Dirichlet / 数量歪み
    iu_xray*.py      IU X-ray ローダ・ラベル抽出・前処理・torch dataset
    synthetic.py     合成データ（ハーネス検証用）
  models/
    base.py                    weighted_average / hetero_aggregate / slice_to
    iu_xray_torch_model.py     MultimodalNet + TorchMultimodalBackend（本番）
    mock_models.py             numpy モック（ハーネス検証用）
  strategies/    fedavg / fedmd / fedproto / loot（合成データ経路）
  comm/
    classical.py     通信量計測
    quantum_stub.py  ★ QFL 差し替え点（qubits / ebits フィールド用意済み）
  engine/        client / server / simulator
  metrics/       classification / analysis / logger / results_store

scripts/
  iu_federated.py      ①② 本番ランナー
  iu_federated_s3.py   ③ 本番ランナー
  iu_federated_s4.py   ④ 本番ランナー
  iu_baselines.py      Centralized / Local 基準線
  iu_make_split.py     凍結split生成
  iu_cache_images.py   画像テンソルキャッシュ
  iu_comm_estimate.py  通信量の解析的算出（学習不要）
  run_benchmark.py     ※合成データ時代の遺物
  smoke_test.py        ※同上

splits/iu_split.json   凍結split（train_pool / public / test 627件 / by_alpha）
```

### 主要アーキテクチャ

```
画像 (B,3,224,224) → ResNet-50 → avgpool → (B,2048) → ★img.proj Linear(2048→d) ┐
                                                                                │
                                              fusion = 平均（パラメタなし）→ ★head Linear(d→14)
                                                                                │
テキスト (B,L≤256) → BERT-base → [CLS] → (B,768) → ★txt.proj Linear(768→d) ─────┘
```

★ = 幅可変（`vary_embed` 時のみクライアント間で形が違う）。
`img.proj` は weight (d,2048) の**行**、`head` は weight (14,d) の**列**をスライスする。
FedProto 使用時のみ、fusion と head の間に `proto: Linear(d→128)` が入り、
**分類経路が proto 空間を通る**（BCE 勾配が流れるため、埋め込みの一点崩壊を防ぐ）。

---

## 拡張の道筋

### 直近

- **② の α スイープ**：HeteroFL α=0.1 / α=1.0（対になる Local hetero も要取得）。
- **④ への HeteroFL 移植**：`hetero_aggregate` / `slice_to` は共通なので、
  s4 に heterofl ループを移すだけ。既存④FedMD と同一 dim 割当のため交絡なし。
- **凍結バックボーン HeteroFL**：`--freeze-image --freeze-text` で共有を
  projs+head（推定 ~29 MB/R）に圧縮し、精度税を実測する。QFL-② の OTP 実現可能性に直結。

### HeteroFL + MIN（④・提案段階、未検証）

HeteroFL は入れ子により**モデル異種下で共通の埋め込み空間を復活させる**
（クライアント k の埋め込み = グローバル埋め込みの先頭 d_k 座標）。
これにより、埋め込み空間で動くため④で使えなかった MIN が原理的には適用可能になる。
ただし2つの障害がある：

1. `min_net = Linear(d,d) → ReLU → Linear(d,d)` は入力側と出力側の**両方**をスライスする
   2層非線形写像。単層スライスより入れ子近似が弱い。
2. min_net を訓練できるのはマルチモーダルクライアントのみ。
   `client_mods = [multimodal]*q + [image-only]*n`、`embed_dims = [128,256,192,320]` なので、
   **1:3 では唯一のマルチモーダルが最も狭い d=128** となり、min_net の外殻が未訓練のまま配布される。
   → **制約：HeteroFL+MIN ではマルチモーダルクライアントが最大幅を持たねばならない。**

### 深さ異種（DepthFL / ScaleFL 方式）

early-exit head により、浅いクライアントは共有プレフィックスのみ学習。
集約は「そのキーを持つ人だけで平均」＝幅のカバレッジ平均と同型
（`hetero_aggregate` の missing-key 対応で ~15行）。
ただし現アーキには fusion 後の「深さ」が存在しないため、トランク追加＝**新アーキ**になり、
専用の Local / FedAvg-deep 基準線が別途必要。独立トラック。

### QFL への差し替え

1. `comm/quantum_stub.py` に qubit/ebit 会計・エンタングルメント失敗・減衰・再送を実装。
2. 同一の評価プロトコル（40評価点・peak採用・凍結split）で古典 vs 量子を比較。
3. 通信量の桁差（param 4,278 MB/R vs logit 0.09 MB/R）が OTP 実現可能性を左右する。

**QFL の優位性軸は情報理論的安全性**であり、学習能力ではない
（dequantization / barren plateau 批判は学習能力の主張を狙う）。
重要な整理：**QKD が守るのは経路であって共有物ではない。**
honest-but-curious なサーバはロジット平文を受け取るため、
チャネルの情報理論的安全性と共有内容の推論耐性は**直交**する。

---

## limitation（卒論に明記すること）

- 本ベンチの「モデル異種」は**幅異種**であり、真のアーキ異種（ResNet vs ViT）は未検証。
  入れ子集約は原理的に適用不能で、FML 系の相互蒸留が代替候補。
- FedMD / LOOT は**共有アンカー入力**を要求する。電子カルテ・検査値・時系列バイタルには
  分布整合な公開集合がほぼ存在せず、構造的にスケールしない。
  一方 FedAvg / HeteroFL / FedProto / MIN は**アンカー不要**。
- 単一シード。ヘッドライン4セル × 3seed は将来のオプション。
