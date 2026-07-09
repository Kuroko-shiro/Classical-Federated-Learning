# QFL Multimodal Benchmark — Classical Baseline Harness

マルチモーダル医療データ向け量子連合学習（QFL）の **古典ベンチマーク基盤**。
4シナリオ（モデル同種/異種 × モダリティ同種/異種）を古典FL/MLで実装し、
**異種性のコスト（heterogeneity tax）を一貫した指標で定量化**する。

長期目標：古典で「穴」になる箇所（精度が伸びない／通信が増える／収束が遅い領域）を
特定し、そこに対して QFL が優位を示すことを導く。本リポジトリはそのための
**比較基準線**と、**量子コンポーネントを後から差し込める差し替え点**を提供する。

> 本フェーズで量子は入れない。量子の効果を測る「物差し」を先に作るための設計。

---

## クイックスタート

```bash
# 依存（numpy/scipy/scikit-learn のみで CPU 完結。torch 不要で動く）
pip install -r requirements.txt

# ハーネス検証（合成データで scenario 1 & 2 を end-to-end 実行）
python scripts/smoke_test.py

# 本番ベンチマーク（複数 seed で比較表を出力）
python scripts/run_benchmark.py --seeds 3 --target 0.8
```

`results/<run>/history.json` に全ラウンドの指標と通信量が保存される。

---

## 設計の核心（なぜこの構造か）

### 1. Strategy は「パラメータ集約」ではなく汎用プロトコル
連合学習を `broadcast → ローカル更新 → aggregate` という抽象プロトコルで表現する。
これにより、**パラメータ平均が使えない②③④を最初から構造に乗せられる**。

| 手法 | broadcast するもの | aggregate するもの | 対象シナリオ |
|---|---|---|---|
| FedAvg / FedProx | グローバル共有パラメータ | 重み付き平均 | ① |
| FedProto | グローバルクラスプロトタイプ | プロトタイプ平均 | ②（対照） |
| FedMD | 公開データ上の合意 logit | logit 平均 | ②（主） |

すべて同じ `Payload` 型・同じ `FederatedStrategy` インターフェースに乗る。

### 2. 通信層が量子の差し替え点
`CommunicationChannel` が全送受信を仲介し、通信量（scalar数・bytes・方向）を記録する。
`TransmissionRecord` には `qubits` / `ebits` フィールドが用意済み。
量子チャネル（`comm/quantum_stub.py`）が同じインターフェースで
エンタングルメント失敗・減衰・再送を実装すれば、**Strategy/Engine を一切変えずに
QFL へ移行できる**（チームのロードマップ #3/#4 への接続点）。

### 3. numpy モックで全部動く → torch は差し替えるだけ
`models/mock_models.py` が numpy 製マルチモーダルモデル（モダリティ別エンコーダ +
mean fusion + 共有 head）を実装し、torch 無しでハーネスを完全検証できる。
実環境では `models/torch_models.py`（stub）に本物のエンコーダを実装して差し替える。

---

## ディレクトリ構成

```
src/qflbench/
  core/
    types.py         Payload / ClientUpdate / RoundMetrics 等の抽象型
    interfaces.py    全 ABC（FederatedDataset, ModelBackend, FederatedStrategy,
                     CommunicationChannel, Partitioner）= 差し替え可能な継ぎ目
    registry.py      名前→実装クラスの解決（config 駆動）
  data/
    base.py          ModalitySpec, モダリティ割当（full/disjoint/random）
    partition.py     IID / Dirichlet / 数量歪み（統計的異種性）
    synthetic.py     検証用 合成マルチモーダルデータ（相補的2モダリティ）
    iu_xray.py       IU X-ray ローダ（STUB: 実環境で前処理を実装）
  models/
    base.py          パラメータ平均ヘルパ
    mock_models.py   numpy 製マルチモーダルモデル（動作する基準実装）
    torch_models.py  torch backend（STUB: 同インターフェースで実装）
  strategies/
    fedavg.py        FedAvg + FedProx（①）
    fedproto.py      FedProto（②対照、プロトタイプ正則化つき）
    fedmd.py         FedMD（②主、知識蒸留つき）
  comm/
    classical.py     古典通信チャネル（通信量計測）
    quantum_stub.py  量子チャネル（STUB: QFL 差し替え点）
  engine/
    client.py        クライアントラッパ
    server.py        ラウンドのオーケストレーション + 評価
    simulator.py     シナリオ構築 + Centralized/Local 基準線
  metrics/
    classification.py  accuracy / macro-F1 / AUROC / AUPRC + 公平性集計
    analysis.py        収束効率・heterogeneity tax・モダリティ分解（①②③④共通）
    logger.py          per-round 指標 + 通信量を JSON 保存
  utils/seed.py      再現性

scripts/
  smoke_test.py      ①② を end-to-end 実行（ハーネス検証）
  run_benchmark.py   複数 seed で比較表を出力（本番）
```

---

## 評価指標（①②③④で一貫）

`metrics/analysis.py` が以下を **シナリオ非依存**に計算する。①②では一部が自明値に
縮退するが、**同じコードパスが③④でそのまま意味を持つ**。

| カテゴリ | 指標 | QFL 比較での役割 |
|---|---|---|
| ユーティリティ | accuracy, macro-F1, AUROC, AUPRC | 精度の絶対水準（医療は不均衡 → F1/AUPRC 重視） |
| 公平性 | per-client mean / worst / std | 施設差・バイアスへの頑健性 |
| 通信(量) | 累積 up/down bytes・scalars | **量子通信コストの比較軸** |
| 通信(効率) | 目標精度到達ラウンド数・到達時累積通信量 | **量子の収束効率の比較軸** |
| 基準線 | Centralized 上限・Local 下限・gap-filled | 異種性のコスト（①→④の低下が tax） |
| 学習曲線 | AUC（全ラウンド平均） | 軌跡全体の良さ |
| プライバシ | DP ε vs 精度（Phase 2 で値が入る枠） | **情報理論的安全性 vs DP** |
| モダリティ | モダリティ数別の精度分解（③④で有効） | ③④の穴の特定 |

---

## 現状の結果（合成データ, 6 clients, Dirichlet α=0.2, 3 seeds）

| 手法 | final acc | macro-F1 | worst | rnds→0.8 | total comms | gap-filled |
|---|---|---|---|---|---|---|
| ① FedAvg | 0.869±0.016 | 0.867 | 0.742 | 6.7 | 903KB | 0.784 |
| ① FedProx | 0.871±0.015 | 0.869 | 0.749 | 6.7 | 903KB | 0.807 |
| ② FedMD | 0.734±0.036 | 0.714 | 0.688 | never | 1253KB | 0.193 |
| ② FedProto | 0.187±0.005 | 0.053 | 0.187 | never | 57KB | 0.000 |

**読み方：**
- **model-heterogeneity tax**：gap-filled が ①0.78 → ②0.19 に低下。
  モデルが揃えば 0.87 届くが、バラバラだと蒸留を使っても 0.73 止まり（目標未達）。
- **通信コスト**：② FedMD は公開データ logit を毎ラウンド往復するため通信増（1253KB）。
- **FedProto** は埋め込み空間の整列が不安定で機能せず（対照）。蒸留法の優位を示す。

> これらは合成データの数値。実環境では IU X-ray に差し替えて取り直す。

---

## 拡張の道筋

### ③④（モダリティ異種）への拡張
- `modality_mode` を `"random"`/`"disjoint"` にすればモダリティ部分集合が割り当たる。
- ③の「同じモデル」= `share_encoders: False`（モダリティ別エンコーダはローカル、
  head のみ共有）。④はそれ + モデル異種。
- **未対応の論点**：FedMD の公開データはクライアントごとに保有モダリティが異なると
  同一サンプルでも異なるモダリティ部分集合で予測することになる（要対処）。

### IU X-ray への差し替え
`data/iu_xray.py` の stub を実装（実環境で torch + ダウンロード必要）。
`FederatedDataset` インターフェースは固定なので、合成 → IU X-ray は config 変更で済む。
ラベルリーク対策（Impression マスク等）は実装時に確定（O1）。

### QFL への差し替え（Phase 2）
1. `comm/quantum_stub.py` に qubit/ebit 会計・エンタングルメント失敗・減衰・再送を実装。
2. `models/torch_models.py` を VQC ベースのハイブリッドモデルに置換（任意）。
3. 同じ `run_benchmark.py` で古典 vs 量子を同一指標で比較。

---

## 注意点

- ブラウザストレージや外部ネットワークは使わない（CPU・ローカル完結）。
- 合成モデルは **ベンチマーク基盤の検証用**であり、実データで高精度を狙うものではない。
- torch backend / IU X-ray / 量子チャネルは stub。インターフェースは確定済み。
