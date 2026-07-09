# 実装手法と設計方針

本ドキュメントは、QFL マルチモーダルベンチマーク（古典基盤）の実装に際して
用いた手法・設計方針とその根拠をまとめたもの。コードのどこに何が・なぜあるかを
後から追えるようにする。

対象コミット時点の構成：scenario ①②が動作、③④は配線済み、torch backend と
量子チャネルは stub（インターフェース確定）。

---

## 1. 全体の設計哲学

### 1.1 「量子を入れる前に物差しを作る」
本フェーズで量子は実装しない。狙いは、4シナリオ（モデル同種/異種 ×
モダリティ同種/異種）の古典ベースラインを取り、**異種性のコスト（heterogeneity
tax）を一貫指標で定量化**し、後で QFL を同じ枠組みに差し込んで比較できる土台を
作ること。設計上の最優先事項は「正確さ」より「**差し替え可能性**」と
「**シナリオ間の一貫性**」。

### 1.2 3つの不変条件（invariant）
実装全体を貫く制約として以下を置いた。

1. **プロトコルは framework 非依存**：`core/` は torch を import しない。
   tensor は境界で numpy。→ numpy モックで全検証でき、torch は後から差し替え。
2. **通信は単一の継ぎ目を通る**：すべての送受信が `CommunicationChannel` を
   経由する。→ 量子チャネルへの差し替え点が1箇所に閉じる。
3. **指標はシナリオ非依存**：`metrics/analysis.py` は run summary に対する
   純粋関数。①②で自明値に縮退しても③④で同じコードが意味を持つ。

---

## 2. アーキテクチャ手法

### 2.1 プロトコル抽象化（最重要の設計判断）
**方針**：連合学習を「パラメータ集約」ではなく、
`broadcast → ローカル更新 → aggregate` という**汎用プロトコル**として抽象化した。

**根拠**：素朴に「重みを平均する」インターフェースにすると、実装できるのは①
（パラメータ平均が成立する同種モデル）だけになる。②③④はパラメータ平均が
成立しない。FedProto/FedMD は「勾配（パラメータ）ではなく抽象表現や予測出力を
やり取りする」手法であり、これを最初から同じ型に乗せる必要があった。

**実装**：
- `core/types.py` の `Payload`（`kind` で PARAMETERS/PROTOTYPES/LOGITS を区別、
  `tensors` が numeric 中身、`num_scalars()`/`nbytes()` で通信量を自己申告）。
- `core/interfaces.py` の `FederatedStrategy` が
  `initialize → broadcast → client_fit → aggregate` を定義。
- これにより FedAvg（パラメータ）/FedProto（プロトタイプ）/FedMD（logit）が
  **同一インターフェース**に乗る。

| 手法 | broadcast | client が返す | aggregate |
|---|---|---|---|
| FedAvg/FedProx | 共有パラメータ | 共有パラメータ | 重み付き平均 |
| FedProto | グローバルプロトタイプ | ローカルプロトタイプ | クラス別加重平均 |
| FedMD | 合意 logit | 公開データ上の logit | logit 平均 |

### 2.2 Registry パターン（config 駆動の差し替え）
**方針**：各差し替え可能ファミリ（strategy/model/channel/partitioner/dataset）を
文字列キーで登録し、config の `name:` で解決する。

**実装**：`core/registry.py` のデコレータ `@STRATEGIES.register("fedavg")` 等。
import 副作用で登録され、`STRATEGIES.get(name)` でクラスを引く。

**根拠**：シナリオ × 手法 × α × seed の組合せ爆発を、コード変更なしに config の
差し替えだけで回すため。

### 2.3 抽象基底クラスによる契約の明示
**方針**：差し替え点を ABC（`ModelBackend`, `FederatedStrategy`,
`CommunicationChannel`, `Partitioner`, `FederatedDataset`）として定義。

**根拠**：torch backend や量子チャネルを後から実装する人が「何を実装すれば
ハーネスに乗るか」を型で把握できる。`ModelBackend` は
get/set_parameters・local_train・class_prototypes・predict_logits・distill・
evaluate を要求する（`docs/torch_backend_guide.md` 参照）。

---

## 3. データ生成・分割の手法

### 3.1 非線形分離可能な合成マルチモーダルデータ
**方針**：各クラスを**複数サブクラスタ（mixture of Gaussians）の和**として生成し、
サブクラスタ中心をクラス間でインターリーブ配置する（`data/synthetic.py`）。

**根拠（重要な試行錯誤）**：当初は1クラス＝1ガウシアンの線形射影だったが、
**線形分離可能なため FedAvg が1ラウンドで 100% に飽和**し、異種性のコストが
全く見えなかった。サブクラスタ化で「単一の超平面では分離できない」タスクにし、
ReLU エンコーダの非線形性が意味を持つようにした。これで①が中間精度
（~0.87）に落ち着き、non-IID の影響が可視化された。

### 3.2 モダリティ相補性の作り込み
**方針**：concept 空間の次元を、画像モダリティと表現モダリティに**ほぼ排他的な
部分集合**として割り当てる（`_complementary_masks`）。各モダリティは
ラベル情報の一部しか持たず、両方揃って初めて高精度になる。

**根拠**：③④（モダリティ異種）が科学的に意味を持つには、「片方のモダリティだけ
では不十分」という相補性が必須。これが無いと「モダリティ欠損クライアント」の
性能劣化が現れず、③④が退化する。①②段階から作り込んでおく。

### 3.3 統計的異種性（non-IID）の生成
**方針**：Dirichlet(α) によるラベル分布歪み（`data/partition.py`）。α 小で強い
non-IID、α 大で IID に近づく。数量歪み・IID も同インターフェースで提供。

**根拠**：FedAvg/FedProx/SCAFFOLD 評価の標準的 non-IID 生成法。α を振ることで
「異種性の度合い」を連続的に制御し、heterogeneity tax の曲線を引ける。

---

## 4. モデル実装の手法

### 4.1 numpy 製マルチモーダルモデル（手書き backprop）
**方針**：モダリティ別線形エンコーダ（+ReLU）→ mean fusion → 共有線形 head を
numpy で実装し、SGD・交差エントロピーの forward/backward を手書きした
（`models/mock_models.py`）。

**根拠**：torch が使えない検証環境でハーネス全体（プロトコル・集約・評価・通信
計測）を end-to-end で動かすため。**モデルの精度自体が目的ではなく、
連合プロトコルの検証substrate**。torch backend で本物のエンコーダに差し替える。

### 4.2 mean fusion（パラメータフリー融合）
**方針**：融合を「存在するモダリティの埋め込みの平均」とし、学習パラメータを
持たせない。

**根拠**：これが③（モダリティ部分集合）を成立させる鍵。画像だけ持つクライアントは
{画像}だけを平均すればよく、全モダリティ持つクライアントは全部を平均する。
**同じ head が異なるモダリティ構成のクライアント間で共有できる**のは、すべての
エンコーダが同じ embed_dim に写像し、融合がパラメータフリーだから。

### 4.3 共有キーの分離（share_encoders フラグ）
**方針**：`shared_parameter_keys()` が「集約対象のパラメータ」を返す。
`share_encoders=True` なら全パラメータ（①）、`False` なら head のみ（③）。

**根拠（重要な罠の修正）**：当初は常に head のみ共有にしていた。すると①で
**グローバルモデルのエンコーダが未学習のまま**になり、global 精度が 0.175 に
崩壊した（per-client は各自エンコーダを学習するので 0.93）。これは
「『同じモデル』の意味がシナリオ依存」であることを示す。①は全体共有、③は head
のみ共有、と切替可能にして解決。これは研究上の論点（③の「同じモデル」とは何か =
決定事項 D4）と直結する。

---

## 5. 連合学習アルゴリズムの実装手法

### 5.1 FedAvg / FedProx（scenario ①）
**手法**：FedAvg は共有パラメータを example 数で加重平均（`models/base.py` の
`weighted_average`、全 dict に共通するキーのみ平均する設計で異種モデルでも
graceful degradation）。FedProx は近接項 `μ/2·‖w−w_global‖²` を共有パラメータに
追加（`local_train` に `proximal_mu`/`global_params` を thread する形で、
専用サブクラス不要）。

**根拠**：FedProx の近接項は McMahan/Li らの定式に従い、client drift を
抑える。サブクラス化せず引数で渡すことで、同じ学習ループを再利用。

### 5.2 FedProto（scenario ② 対照）
**手法**：原論文（Tan et al., AAAI 2022, arXiv:2105.00243）に従い実装。
クライアントは勾配ではなくクラスプロトタイプ（クラス別平均埋め込み）を送り、
サーバはクラス別に加重平均してグローバルプロトタイプを返す。ローカル学習は
分類損失に加えて「埋め込みを対応するグローバルプロトタイプへ近づける正則化項」
`proto_mu/2·‖z−g[y]‖²` を持つ（`models/mock_models.py` の local_train 内、
`strategies/fedproto.py` がプロトタイプを extra で渡す）。`proto_mu=1.0` を
デフォルトにしたのは原論文で λ=1 が最適とされるため。

**観測された挙動と根拠**：合成設定で精度が ~0.18（ほぼ当て推量）に潰れた。
原因は**独立に学習した埋め込み空間の符号・回転の任意性**で、整列前の空間の
プロトタイプを平均しても無意味な点になり、そこへ強く引き寄せると全クラスが
一点に潰れる。これは原論文が指摘する「アーキテクチャや入出力空間の違いによる
局所勾配の不整合」の現れ。**対照手法**として残し、「プロトタイプ法は埋め込み
整列に敏感」という知見と、蒸留法を主手法に選んだ根拠を示す。

### 5.3 FedMD（scenario ② 主）
**手法**：FedMD（Li & Wang, 2019, arXiv:1910.03581）に従う。
(1) サーバが公開（proxy）データを保持、(2) 各クライアントが公開データ上の logit を
アップロード、(3) サーバが平均して合意 logit を作りブロードキャスト、(4) 各
クライアントが合意 logit へソフトターゲット蒸留してから private データで学習
（`strategies/fedmd.py`、蒸留は `models/mock_models.py` の `distill`）。蒸留は
温度付きソフトマックス交差エントロピー（T=2.0）で実装。

**根拠と選択理由**：logit は「クラス確率」という**共通言語**なので、アーキテクチャが
違っても意味を持ち、**埋め込み整列を経由しない**。FedProto で観測した不安定性を
回避できる。実際に合成設定で local 下限（0.656）を上回る 0.685 を達成し、安定して
学習が成立した。代償は公開データ logit を毎ラウンド往復する通信増。

### 5.4 評価方法の使い分け
**方針**：global モデルがある手法（FedAvg/FedProx）は単一 global モデルを
共有テストで評価。global モデルがない手法（FedProto/FedMD）は per-client 評価
（`engine/server.py`）。FedProto はグローバルプロトタイプへの最近傍分類、FedMD は
各クライアントの head で分類。

**根拠**：手法の性質に評価を合わせる。FedProto は「プロトタイプが分類器」、FedMD は
「蒸留で揃えた各自の head が分類器」という本来の使い方に従う。

---

## 6. 通信モデルの手法（QFL 接続点）

### 6.1 チャネル抽象と通信会計
**方針**：`CommunicationChannel.transmit()` が全送受信を仲介し、
`TransmissionRecord`（round/方向/kind/scalar数/bytes、加えて未使用の qubits/ebits
フィールド）を記録（`comm/classical.py`）。

**根拠**：通信量を「後で QFL と比較できる形」で測るため。fp32 換算の bytes と
scalar 数を記録し、**量子チャネルが同じレコード schema を qubits/ebits で埋める**
ことで、古典 bytes vs 量子 qubits を同じ表で比較できる。

### 6.2 量子チャネルの stub 化
**方針**：`comm/quantum_stub.py` を、同一 `transmit` シグネチャを持つ stub として
用意。エンタングルメント生成失敗確率・デコヒーレンス・qubit エンコーディング
コストのパラメータ枠だけ置いた。

**根拠**：QFL 移行点を**コード上に明示**するため。実装時は (1) payload を qubit 数で
サイズ、(2) ebit 消費・生成失敗・再送、(3) メモリ減衰によるノイズ付与、を埋める。
ロードマップ #3/#4 に対応。

---

## 7. 評価指標の手法（①②③④の一貫性）

### 7.1 不均衡を考慮したユーティリティ指標
**方針**：accuracy に加え macro-F1・AUROC・AUPRC を sklearn で計算
（`metrics/classification.py`）。多クラスは one-vs-rest macro、存在しないクラスの
列は除外。

**根拠**：医療データは不均衡。accuracy 単独は誤解を招くため、クラス均等に重みづける
macro-F1 と、稀少陽性に強い AUPRC を併用。

### 7.2 基準線と heterogeneity tax
**方針**：Centralized（全データプール = 上限）と Local-only（各自単独 = 下限）を
測り、`gap-filled = (FL−Local)/(Central−Local)` で「FL が埋めた割合」を算出
（`engine/simulator.py` の baseline 関数、`metrics/analysis.py` の
`gap_filled_fraction`）。

**根拠**：FL の価値を相対化する標準的方法。**①→④で gap-filled が低下する量が
heterogeneity tax そのもの**。観測では①0.78→②0.19。

### 7.3 収束効率指標（QFL 比較の核心）
**方針**：`rounds_to_target`（目標精度への到達ラウンド数）、`comms_to_target`
（到達までの累積通信量）、`area_under_curve`（学習曲線の平均）を追加
（`metrics/analysis.py`）。

**根拠**：QFL が主張する「少ない通信で収束」を比較する軸。最終精度だけでなく
「good-enough までのラウンド数・通信予算」を測ることで、量子通信の効率優位を
後で同一指標で検証できる。観測では①が 6.7 ラウンドで 0.8 到達、②FedMD は never。

### 7.4 モダリティ分解（③④で有効）
**方針**：最終 per-client 精度を「クライアントが持つモダリティ数」でグループ化
（`modality_breakdown`）。①②では全員フルモダリティなので単一バケツに縮退。

**根拠**：③④で「単一モダリティクライアントが性能を引き下げているか」= **穴の
位置**を特定するため。①②段階から同じコードを通し、③④で複数バケツに分かれる。

### 7.5 multi-seed 集約
**方針**：`scripts/run_benchmark.py` が複数 seed で回し、平均±標準偏差で報告。

**根拠**：ベンチマークとして信頼されるには単発実行では不可。seed をまたいだ
平均±分散で、結果が偶然でないことを担保する。

---

## 8. 検証手法

### 8.1 スモークテスト駆動の罠発見
**方針**：実装初期から `scripts/smoke_test.py` で①②を end-to-end 実行し、
数値を見ながら設計の妥当性を確認した。

**根拠（効果）**：「動かさず設計だけ進めると見逃す罠」を実コードで早期発見できた。
具体的に発見・修正した罠：(a) head のみ共有による global モデル崩壊（§4.3）、
(b) 異なる埋め込み次元でのプロトタイプ平均クラッシュ（embed_dim は全クライアントで
一致が必要）、(c) FedProto の埋め込み整列不安定性（§5.2）。

### 8.2 最小テスト
**方針**：`tests/test_harness.py` で各戦略の動作・指標計算・通信会計・③配線を
検証（pytest または python 直接実行）。

**根拠**：リグレッション防止。特に「③の配線（modality_mode=random +
share_encoders=False）が動く」ことをテストで保証し、③④への道筋が壊れていない
ことを担保する。

---

## 9. 設計判断の要約（なぜそうしたか一覧）

| 判断 | 採用 | 根拠 |
|---|---|---|
| プロトコル抽象 | broadcast→update→aggregate | ②③④をパラメータ平均なしで乗せるため |
| tensor 境界 | numpy（torch非依存） | モックで全検証、torch は差し替え |
| 通信 | 単一 channel 経由 | 量子差し替え点を1箇所に閉じる |
| 合成データ | mixture-of-Gaussians | 線形分離飽和を避け tax を可視化 |
| 融合 | パラメータフリー mean | モダリティ部分集合を成立させる |
| 共有キー | share_encoders で分岐 | 「同じモデル」の意味がシナリオ依存 |
| ②主手法 | FedMD（蒸留） | logit は共通言語、埋め込み整列不要で安定 |
| ②対照 | FedProto | 整列敏感性を示し蒸留選択を正当化 |
| 指標 | シナリオ非依存の純粋関数 | ①②③④を同じ物差しで比較 |
| 収束効率 | rounds/comms-to-target | QFL の効率優位の比較軸 |
| 信頼性 | multi-seed 平均±分散 | 単発でなくベンチマーク水準に |

---

## 参考文献
- McMahan et al. (2017) Communication-Efficient Learning of Deep Networks from
  Decentralized Data (FedAvg)
- Li et al. (2020) Federated Optimization in Heterogeneous Networks (FedProx)
- Li & Wang (2019) FedMD: Heterogenous Federated Learning via Model Distillation,
  arXiv:1910.03581
- Lin et al. (2020) Ensemble Distillation for Robust Model Fusion in FL (FedDF)
- Tan et al. (2022) FedProto: Federated Prototype Learning across Heterogeneous
  Clients, AAAI 36(8):8432-8440, arXiv:2105.00243
