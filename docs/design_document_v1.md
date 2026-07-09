# マルチモーダル医療QFL：古典ベンチマーク構築フェーズ — 設計方針・全体像・関連研究・新規性分析

- **バージョン**: v1（living document、MTGまで随時更新）
- **作成日**: 2026-05-30
- **フェーズ**: 量子拡張前の「古典FL/MLによる4シナリオ・ベンチマーク」構築（1ヶ月タスク）
- **直近マイルストーン**: 2週間後のチームMTGで進捗共有（MVPを「ある程度の形」に）
- **ステータス凡例**: ✅確定 / 🟡推奨デフォルト（要チーム確認） / ⬜未決

---

## 0. エグゼクティブサマリ

本フェーズの目的は、チームの最終目標である **マルチモーダル医療データ向け量子連合学習（QFL）** に対する **比較基準線（古典FL/ML）を確立し、異種性が学習・通信・プライバシに与えるコスト（heterogeneity tax）を定量化する**ことである。同時に、後で量子コンポーネントを差し替えられる **データ非依存・通信層抽象化されたシミュレータ** を構築する。

- 対象は4シナリオ = **モデル同種/異種 × モダリティ同種/異種 の2×2**。
- データセットは **IU X-ray（Indiana University Chest X-rays, Open-i）** に確定（画像＋放射線レポートの相補的2モダリティ、credentialing不要）。
- 文献サーベイの結論：QFLの構成要素（VQCベースQFL、量子インターネットQFL、情報理論的安全な集約、量子マルチモーダル融合、医療QFL）は **個別には存在するが、いずれも単一モダリティ／トイデータ／ハードウェア異種性が中心**。**「モデル×モダリティ異種性」を体系的に扱い、現実的な量子インターネット条件下で、実マルチモーダル医療データに対してベンチマークする統合的研究は空白**。
- 本フェーズはその空白を埋めるための土台（古典ベースライン＋heterogeneity tax＋差し替え可能harness）を作る。

---

## 1. 背景・目的・本フェーズの位置づけ

### 1.1 チームのビジョン（前提）
量子通信・量子機械学習・連合学習を統合し、医療画像・電子カルテ・検査値・時系列バイタル等の異種・非IID・分散データを安全に活用できる **マルチモーダル量子連合学習基盤** を構築する。情報理論的安全性と実装可能性／学習効率の両立を追究し、量子技術が「あると面白い」ではなく「なければ成立しない」役割を果たす設計を重視する。

### 1.2 本フェーズが担うこと
QFL/QML実装の前段として、**4シナリオを古典FL/MLで実装したベンチマーク**を構築する。これは以下の3つを同時に満たす：

1. **比較基準線の確立** — QFLの優位（通信・プライバシ・非IID耐性）を後で主張するための古典側の数値。
2. **heterogeneity tax の定量化** — 異種性（統計的・モデル・モダリティ）が各指標をどれだけ劣化させるか。
3. **差し替え可能なharness** — 通信層・集約層を抽象化し、量子通信層・量子集約を後から config で差し込める設計。

> 本フェーズで「量子を入れない」のは妥協ではなく、**量子の効果を測るための物差しを先に作る**という設計判断である。

---

## 2. 問題設定：4シナリオ（2×2）

ナレッジ文の4シナリオは2軸の組み合わせとして整理する。

| | **同じモデル** | **異なるモデル** |
|---|---|---|
| **同種モダリティ** | ① 全クライアント同一アーキ・同一モダリティ集合。異種性=**統計的(non-IID)**のみ | ② 同一モダリティ集合・**異なるアーキ**。重み平均不可→蒸留/プロトタイプ |
| **異種モダリティ** | ③ 共有アーキ・**クライアントごとに異なるモダリティ部分集合** | ④ 異なるアーキ・異なるモダリティ部分集合（=現実の病院連携） |

### 2.1 確定した運用定義（曖昧さの解消）
- **③④の「異なる種類のデータ」**：✅ **modality-subset（missing-modality）解釈**。全クライアントは同一のペア付きマルチモーダルデータセット由来で、共通タスクを解くが、各クライアントは一部のモダリティのみ保有（例：施設Aは画像、施設Bはテキスト、施設Cは両方）。「別タスク・別データ」解釈は採らない（連合の意味が失われるため）。
- **③の「同じモデル」**：🟡 **共有アーキ＝モダリティ別エンコーダはローカル保持、共有する融合層＋分類ヘッドのみ集約**。これが「モダリティが違うのにモデルが同一」を整合させる唯一の解釈。要チーム確認。

実装上の最重要事実：**パラメータ平均（FedAvg系）が成立するのは①のみ**。②④は蒸留／プロトタイプ、③④はモダリティ処理が必須。よってシミュレータは複数の集約パラダイムを最初から切替可能にする。

---

## 3. 設計方針（要件定義）

要件は **研究要件 → データ要件 → システム/機能要件 → 非機能要件** の順に上位が下位を規定する。

### 3.1 研究要件
- 4シナリオそれぞれで、**Centralized（上限）/ Local-only（下限）/ FL** の3点を測り、`(FL − Local)/(Centralized − Local)` で「FLが埋めた割合」を評価。
- 評価軸：ユーティリティ（**医療不均衡のためMacro-F1 / AUROC / AUPRC**を主、Accuracyは副）、通信コスト、収束、異種性ロバスト性（Dirichlet α sweep）、公平性（per-client分散・worst-client）、プライバシ–精度トレードオフ。
- **統計的厳密性**：単発実行不可。**複数seed × 平均±分散**で報告（significanceも）。
- **通信メトリクスをQFL写像可能に**：古典では「ラウンド数 / 上り下り合計スカラ数 / bytes（fp32等）」を測るが、後でQFL層が「転送qubit数・消費ebit数」で語れるよう、通信コストを **"目標精度到達までのリソース量"** として抽象的に設計する。ここを今ぶらすと後のQFL比較が成立しない。

### 3.2 データ要件と IU X-ray
データへの要件（研究要件からの逆算）：
1. 複数モダリティがペアで存在（同一サンプル/患者）
2. 共通の予測タスク・ラベル
3. モダリティ部分集合に分割可能
4. **各モダリティが相補的な信号を持つ**（③④が科学的に意味を持つ条件）
5. 非IID分割が可能（Dirichlet等）
6. オープン（credentialing負荷が低い）

**採用：IU X-ray / Open-i（Indiana University Chest X-rays）** ✅
- 約7,470枚の胸部X線画像と約3,955のレポート。レポートは Comparison / Indication / Findings / Impression の4節構成。画像とレポートが study 単位でペア付き → **画像＋テキストの相補的2モダリティ**。
- ライセンス CC BY-NC-ND 4.0（非商用・無改変再配布制限／研究室内利用は問題なし）。Kaggle等から **credentialing不要** で入手可。
- 4シナリオすべてを **単一データセットで** 回せる＝クロスシナリオ比較が交絡しない。
- 難点：規模が小さい・単一施設バイアス。ベンチマーク用途では「小規模＝高速反復」で許容。

**ラベルリーク対策（要決定）** ⬜：テキストを使う場合、ラベルがレポート由来だと「text-onlyクライアントが抽出元の文章からラベルを当てる」自明問題が起きる。対策候補：(a) Impression節をマスク、(b) 画像由来の独立ラベルを採用、(c) findings の一部に限定。後述の決定事項で確定。

**補助・将来**：
- **RSNA Pneumonia Detection Challenge**（画像＋薄tabular：年齢/性別/体位）— ①②の大規模・頑健性確認に併用可（先輩の前処理資産あり）。③④には相補性不足のため使わない。
- **MIMIC-IV / MIMIC-CXR**（画像＋ノート＋検査値＋バイタルの真の4モダリティ）— リッチ版の本命だが PhysioNet credentialing + CITI + DUA が必須。**今日から並行申請**し、harness完成後に config 差し替えで拡張。

### 3.3 システム/機能要件とアーキテクチャ
**抽象化レイヤ（疎結合）**：
- **Data層**：データセット非依存ローダ。モダリティ集合・分割（IID / Dirichlet α / 数量スキュー / モダリティ割当）を config 化。
- **Model層**：クライアントごとに別アーキを差し込み可能（②④用）。
- **Aggregation層**：FedAvg / FedProx / SCAFFOLD / MOON（パラメータ系）、FedProto / FedMD / FedDF（モデル異種系）を切替。
- **Communication層（量子差し替え点）**：古典通信を抽象インターフェース化。後で量子通信層（エンタングルメント分配・テレポーテーション・再送・ノイズ）に差し替え。
- **Metrics層**：global / per-client / per-round のロギング、**通信量計測（スカラ数・bytes・ラウンド数）**。
- **Privacy層（Phase 2）**：DP（Opacus）、HE/SecAggのオーバヘッドsim。

**機能要件**：config駆動（Hydra想定、4×複数手法×複数α×複数seedの系統的sweep）、再現性（seed固定）、実験管理（W&B/MLflow）。

**技術スタック**：PyTorchベース、集約は軽量自作harness（②④のモデル異種でFlowerのparam平均前提と戦うより自作が素直。Flower simulation engineを下回りに使う折衷も可）。既存ベンチ（FedMultimodal、Med-MMFL、FLamby）を参考実装に。

### 3.4 非機能要件
- **モジュラリティ最優先**（特に通信層＝QFL統合点）。
- **データ/モダリティ非依存**（IU X-ray → MIMIC を config で交換できる＝dataset選定リスクを吸収）。
- **スケーラビリティ**：医療は cross-silo の小N現実的（例：5〜10施設）。cross-device は将来。

### 3.5 シナリオ別・手法マッピング
| シナリオ | 性質 | 手法 |
|---|---|---|
| ① 同モデル・同モダリティ | non-IID | FedAvg / FedProx / SCAFFOLD / MOON |
| ② 異モデル・同モダリティ | 重み平均不可 | 🟡 **FedProto主**（公開データ不要・クラスプロトタイプのみ送信＝低通信でQFL通信ストーリーに接続）、FedMD / FedDF を副 |
| ③ 同モデル・異モダリティ | modality欠損 | モダリティ別エンコーダ=ローカル、共有ヘッド=集約。cf. FedMSplit, CreamFL, MMFL healthcare review の DCCAE 系 |
| ④ 異モデル・異モダリティ | 最大異種 | ②の蒸留/プロトタイプ ＋ ③のモダリティ処理 |

### 3.6 評価設計（比較構造）
- **シナリオ内**：各シナリオで「FedAvg/FedProx/… vs Centralized vs Local」を比較。
- **シナリオ間**：①→④と異種性を上げたときの各指標の劣化曲線＝**heterogeneity tax**（論文ストーリーの骨格。「モデル異種とモダリティ異種、どちらがどの指標をより壊すか」）。

---

## 4. 関連研究サーベイ

> 目的：本研究の立ち位置と新規性を地に足のついた形で定めるため、古典側（マルチモーダルFL／異種FL）と量子側（QFL／量子インターネット／医療QFL）を整理する。

### 4.1 マルチモーダルFL（ベンチマーク・医療）
- **FedMultimodal**（Feng et al., KDD 2023, arXiv:2306.09486）：最初のマルチモーダルFLベンチマーク。10データセット・8モダリティ。データ分割→特徴抽出→FL→評価のパイプラインを提供し、**modality欠損・ラベル欠損・誤ラベル**へのロバスト性を標準化。本研究のharness設計の主要参考。
- **Med-MMFL**（2026, arXiv:2602.04416）：医療特化の初の包括的マルチモーダルFLベンチマーク。2〜4モダリティ（テキスト・病理画像・ECG・X線・レポート・MRI系列）、6手法を評価。Fed-MIMIC-CXR / Fed-BraTS。**FedAvg・FedProxが最も安定して強い**との報告。
- **FEDMEKI**（2024, arXiv:2408.09227）：医療マルチサイト・マルチモーダル・マルチタスク（7モダリティ、8タスク）。FLamby流の cross-silo 設定。
- **FLamby**（Ogier du Terrail et al., 2022）：cross-silo医療FLベンチマークの先駆。
- **Multimodal FL in Healthcare: a Review**（arXiv:2310.09650）：医療MMFLの異種性源（data space / statistical）を整理。**unimodalクライアントは1エンコーダ、multimodalは2エンコーダ、global=DCCAEで正準相関最大化**という構成は、本研究③の設計パターンそのもの。

### 4.2 モデル異種FL（②の系譜）
- **FedMD**（Li & Wang, 2019, arXiv:1910.03581）：公開データ上のlogitを共有し蒸留。各クライアントが独自アーキを持てる。**公開データ依存**が難点。
- **FedDF**（Lin et al., 2020）：ラベルなし/生成データでサーバ側アンサンブル蒸留。**ラベル付き公開データ不要**。
- **FedProto**（Tan et al., AAAI 2022）：クラスプロトタイプ（平均表現）を共有し局所更新を正則化。**公開データ不要・低通信**。
- **MOON**（Li et al., 2021）：モデルレベル対照学習で client drift を補正（主に統計的異種）。
- **Model-heterogeneous FL survey**（arXiv:2312.12091）：上記を体系化（蒸留系／局所正則化系／data-free系）。
- **MH-pFLGB**（Xie et al., 2024, arXiv:2407.00474）：医療画像向けモデル異種パーソナライズFL（global bypass）。**医療×モデル異種**の既存例。

### 4.3 モダリティ異種/欠損FL（③④の系譜）
- **FedMSplit**（Chen & Zhang, 2022）：全クライアントが同一センサを持つと仮定せず、動的グラフでマルチモーダル分散学習。
- **CreamFL**（Yu et al., 2023）：異種モデル＆異種モダリティのクライアントから公開データ上の知識のみ通信し、サーバの大モデルを対照表現アンサンブルで訓練。**③④の古典最良参照の一つ**。

### 4.4 量子連合学習（QFL）
- **Federated Quantum ML**（Chen & Yoo, 2021）：QFLの先駆。CNN特徴圧縮＋小VQC、サーバ集約のハイブリッド構成。
- **Chehimi & Saad（2022）ほか**：量子データに対するQFL、QNN/QCNNでcluster state分類等。
- **包括的サーベイ**：*Quantum Federated Learning: A Comprehensive Survey*（2025, arXiv:2508.15998）、*When FL Meets Quantum Computing: Survey and Research Opportunities*（2025, arXiv:2504.08814）。後者は古典クライアントが勾配平均、量子クライアントがVQCパラメータ差分を送り、サーバが融合する**量子–古典ハイブリッド異種性**を整理。
- **Q-RAIL**（2026, arXiv:2605.25783）：**ハードウェア異種性**（QPUごとのノイズ・信頼性）に対応する信頼性重み付き集約。MNIST/Fashion-MNIST/**OrganAMNIST（MedMNIST）**でIID/非IID評価。
- 観察：QFLの異種性研究は **ハードウェア異種・量子/古典クライアント混在** が中心で、**「モデルアーキ×データモダリティ」の異種性は扱われていない**。評価も MNIST 級が大半。

### 4.5 量子インターネット／量子通信を考慮したFL
- **Towards FL on the Quantum Internet**（2024, arXiv:2402.09902）：量子インターネット上のQFLを **ネットワーク制約下** で評価。ネットワークトポロジと学習の性質が性能を大きく左右し、**より包括的な研究が必要**と結論。量子中継器がエンタングルメントスワッピング/蒸留を担い、異種ノード（異なるQPUアーキ）を接続。ただし単一モダリティで、エンタングルメント生成失敗・メモリ減衰・再送をマルチモーダルFL収束と結びつけてはいない。
- **Practical QFL and its experimental demonstration**（2025, arXiv:2501.12709）：分散量子秘密鍵で局所更新を保護し、**情報理論的安全性**を持つ安全集約。4クライアント量子ネットワークで実証。→ **IT-privacy QFL は既に実証段階**。
- **QKD-secured Federated Edge Learning in Quantum Internet**（Xu et al., arXiv:2210.04308）：QKDで鍵・モデルを暗号化し盗聴に対する理想的安全性。資源割当を最適化。
- **QFL via Blind Quantum Computing**（Li, Lu, Deng, 2021）：ブラインド量子計算でプライバシを担保するQFL。

### 4.6 医療QFL・量子マルチモーダル
- **FedQTN（Federated Quantum Tensor Networks for Healthcare）**（2024, arXiv:2405.07735）：非IID医療**画像**にQTNを連合学習、DP分析、ROC-AUC 0.91–0.98。**画像のみ**。
- **QFL in Healthcare（QCNN）**（2023, PubMed 40768459）：連合QCNNを **Pneumonia MNIST / CT-kidney** で評価。**トイ/MNIST級・画像のみ**。
- **Advancing Healthcare Using QC and FL（review）**（2026, Springer）：Quantum Split Federated Learning（QSFL）等を整理。医療のシステム異種性に言及。
- **量子マルチモーダル融合**：entanglingレイヤによる中間/後期融合で複数量子回路出力を統合し、**欠損/破損モダリティに頑健**（Pokharel et al., 2025）。→ **量子マルチモーダルは2025年に出始め**。
- **最も近い先行研究**：Quantum-Enhanced FL ＋ Explainable Multimodal ＋ Heart Disease（2025, 文献内参照）。**量子×マルチモーダル×医療×FLは萌芽的に出現済み**。

### 4.7 サーベイのまとめ — 何が在り、何が無いか
**在る（単体では新規でない）**：VQCベースQFL／量子インターネットQFL（初期）／IT-privacy安全集約・QKD-secured FL／量子マルチモーダル融合（2025・萌芽）／医療QFL（画像・トイ）／古典マルチモーダルFL（医療・ベンチ）／古典モデル異種医療FL／QFLのハードウェア異種性対応。

**無い（＝空白）**：
1. 「**モデルアーキ × データモダリティ**」の異種性を **体系的に** 扱うQFL（既存QFL異種性はハードウェア/量子–古典）。
2. **実マルチモーダル医療データ** に対するQFL（既存はMNIST級・画像のみ）。
3. **現実的な量子インターネット層**（エンタングルメント失敗・メモリ減衰・再送）を **マルチモーダルFLの収束/通信/精度** と結びつけた評価。
4. 上記を測る **ベンチマーク/シミュレータ**（古典側はFedMultimodal/Med-MMFL、量子側は単一モダリティ・トイに留まる）。

---

## 5. 新規性・貢献の所在（量子拡張前提）

> 「どう量子を入れるか」は今後の課題。ここでは **どこに新規性・貢献を見出せるか** を、4.7の空白に基づいて特定する。

### 5.1 基本スタンス（正直な位置づけ）
個別の構成要素はいずれも存在する。**「初の○○」ではない**。しかし **統合**——モデル×モダリティ異種性を扱い、現実的な量子インターネット条件下で、実マルチモーダル医療データに対してベンチマークするQFL——は **開かれた、防御可能な新規性空間** である。最も近い先行（FedQTN、量子インターネットQFL、量子マルチモーダル融合、heart-disease量子マルチモーダルFL）はいずれも空白1〜4のどれかを欠いている。

### 5.2 新規性候補（強い順の私見）
- **A. ベンチマーク/シミュレータそのもの（最も具体的・確実）**
  「モデル×モダリティ異種性の2×2」を実マルチモーダル医療データ上で (Q)FL について体系評価する基盤。**本フェーズの古典版がその第一歩**。ベンチマークは具体的貢献として認められやすく、コミュニティ資産になりうる。
- **B. 量子インターネット・ネットワーク層を考慮したQFL（チームの強みと直結）**
  エンタングルメント分配・量子中継・量子メモリ・再送・量子インターネット特有ノイズを **FLの通信層として明示モデル化** し、マルチモーダルFLの収束/通信/精度と関連付ける（ナレッジ・ロードマップ#3/#4）。既存の量子インターネットQFLは初期段階で、この結合は未開拓。
- **C. モダリティ異種性 × 量子融合 × 連合（萌芽の先取り）**
  Pokharel 2025 の量子マルチモーダル融合（欠損モダリティ頑健）を、**連合＋量子通信＋modality-subset＋医療** の設定へ拡張。組み合わせが新しい。
- **D. 医療多モダリティ向け量子特徴選択（差別化要素）**
  λ や Light-cone に基づく量子特徴選択を、**どのモダリティ/時点/領域を通信するか** の選択機構へ拡張し、**通信量削減と解釈可能性を両立**（ロードマップ#2）。主流QFL文献に乏しく、独自性が高い。
- **E. 情報理論的安全性 vs 古典DP/HE の体系比較（本フェーズが直接準備）**
  IT-privacy QFL は実証済（2501.12709）だが、**マルチモーダル医療×異種性の matrix 上で DP/HE と体系比較**した例はない。本フェーズの DP/HE 比較（Phase 2）が土台。

### 5.3 本フェーズ（古典ベンチ）が各新規性をどう準備するか
- A → 2×2のベースライン値・heterogeneity tax 曲線・評価指標を確定。量子版は同じ枠組みで上書き比較できる。
- B → **通信層を抽象化**しておくことで、量子インターネット層を後から差し込み、古典通信コストと直接比較可能。
- C → ③④のモダリティ処理（共有ヘッド集約／modality-subset）を古典で確立 → 量子融合へ置換。
- D → 古典で「どの変数/時点が効くか」「通信量と精度の関係」を測る → 量子特徴選択の評価軸を用意。
- E → DP（ε–精度）と HE/SecAge オーバヘッドを古典で測る → IT-privacy の比較対象を用意。

> つまり本フェーズは「量子を入れない準備」ではなく、**新規性A〜Eすべての測定基盤を一度に整える**作業である。

---

## 6. スコープと段階計画

### 6.1 MVP（2週間後MTGまで） — 「ある程度の形」の定義
- データ非依存 harness 完成。
- IU X-ray で **①（FedAvg/FedProx・非IID）の実測結果**。
- ②（FedProto）の動作確認。
- ③の骨格（modality-subset 割当 ＋ 共有ヘッド集約）。
- ①の Centralized / Local 基準線。
- 通信＋ユーティリティのロギング。
- 本ドキュメント（要件・サーベイ・新規性）。

### 6.2 Phase 2（MTG後）
- ④の本格化、α sweep の系統実行、複数seed統計。
- プライバシ軸（DP ε–精度、HE/SecAgg オーバヘッド）。
- MIMIC 取得時にリッチ多モダリティへ差し替え。

### 6.3 長期（3年ロードマップとの対応）
本フェーズ = ロードマップ「ハイブリッドQFL実装シミュレータ（未踏ターゲット）」「classical FL（DP/HE）との比較」の古典基盤。Phase 2以降で量子インターネット層（#3/#4）、量子特徴選択（#2）、動的トポロジー（#5）へ接続。

---

## 7. 決定事項・未解決の論点

| ID | 内容 | 状態 |
|---|---|---|
| D1 | ③④のデータ構成 = ペア付きマルチモーダル上の modality-subset 解釈 | 🟡 推奨確定（要チーム確認） |
| D2 | harness をデータ/モダリティ非依存・pluggable に | ✅ 確定（最優先） |
| D3 | 主データセット = IU X-ray（全4シナリオ）。RSNA補助、MIMIC並行申請 | ✅ 確定 |
| D4 | ③の「同じモデル」= 共有ヘッド集約＋モダリティ別エンコーダはローカル | 🟡 推奨確定（要チーム確認） |
| D5 | モデル異種手法 = FedProto主、FedMD/FedDF副 | 🟡 推奨確定 |
| O1 | ラベルリーク対策（Impressionマスク等） | ⬜ 未決 |
| O2 | クライアント数（cross-silo 5〜10施設想定） | ⬜ 未決 |
| O3 | ②④の公開代理データの要否（FedMDを使う場合のみ必要） | ⬜ 未決 |
| O4 | 量子コンポーネントの入れ方（VQC局所/量子集約/量子通信層） | ⬜ 今後の課題（本フェーズ対象外） |

---

## 8. 直近のアクション
1. IU X-ray を取得し、画像–レポートのペアリング（XML parentImage タグ）と前処理を確立。
2. harness 骨格（Client / Server / Strategy / CommunicationChannel / DataPartition / ModalityConfig）と Hydra config を設計。
3. ① を FedAvg/FedProx ＋ Dirichlet 非IID で動作させ、Centralized/Local 基準線を取る。
4. **MIMIC の credentialing（PhysioNet + CITI + DUA）を並行申請**（取得に数日〜2週間）。
5. D1/D4 をチームに確認、O1（ラベル定義）を決める。

---

## 参考文献（主要）
- McMahan et al. (2017) Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)
- Li et al. (2020) Federated Optimization in Heterogeneous Networks (FedProx)
- Karimireddy et al. (2020) SCAFFOLD
- Li et al. (2021) Model-Contrastive Federated Learning (MOON)
- Li & Wang (2019) FedMD, arXiv:1910.03581
- Lin et al. (2020) Ensemble Distillation for Robust Model Fusion in FL (FedDF)
- Tan et al. (2022) FedProto, AAAI
- Yu et al. (2023) CreamFL（Multimodal FL with heterogeneous models/modalities）
- Chen & Zhang (2022) FedMSplit
- Feng et al. (2023) FedMultimodal, KDD, arXiv:2306.09486
- Med-MMFL (2026) arXiv:2602.04416
- FEDMEKI (2024) arXiv:2408.09227
- Ogier du Terrail et al. (2022) FLamby
- Multimodal FL in Healthcare: a Review, arXiv:2310.09650
- Xie et al. (2024) MH-pFLGB, arXiv:2407.00474
- Chen & Yoo (2021) Federated Quantum Machine Learning
- Quantum Federated Learning: A Comprehensive Survey (2025) arXiv:2508.15998
- When FL Meets Quantum Computing: Survey & Research Opportunities (2025) arXiv:2504.08814
- Towards Federated Learning on the Quantum Internet (2024) arXiv:2402.09902
- Practical Quantum Federated Learning and its Experimental Demonstration (2025) arXiv:2501.12709
- Xu et al. Privacy-preserving Resource Allocation for FEL in Quantum Internet, arXiv:2210.04308
- Li, Lu, Deng (2021) Quantum Federated Learning through Blind Quantum Computing
- FedQTN: Federated Hierarchical Tensor Networks for Healthcare (2024) arXiv:2405.07735
- Quantum Federated Learning in Healthcare (QCNN) (2023) PubMed 40768459
- Q-RAIL (2026) arXiv:2605.25783
- Pokharel et al. (2025) Quantum multimodal fusion via entanglement
- Demner-Fushman et al. (2016) IU X-ray / Open-i, JAMIA
- Shih et al. (2019) RSNA Pneumonia Detection Challenge dataset
