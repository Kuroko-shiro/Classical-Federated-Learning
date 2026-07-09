# IU X-ray 実データ版（方針B）セットアップ

合成データから IU X-ray 実データへ移行するための手順。ResNet-50 + BERT-base の
マルチモーダル連合学習（Saha 準拠）。**M5 Mac (MPS) で実行**する。

## 1. 環境構築（M5）

```bash
# quantum 環境に追加（既存環境を汚したくなければ別env推奨）
pip install torch torchvision transformers pillow scikit-learn pandas

# MPSで未対応演算をCPUにフォールバック（重要）
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

確認:
```bash
python -c "import torch; print('mps:', torch.backends.mps.is_available())"
```

## 2. データ配置

Kaggle (raddar/chest-xrays-indiana-university) から取得済みの:
- `indiana_reports.csv`
- `indiana_projections.csv`
- 画像フォルダ `images/images_normalized/`（PNG群）

をプロジェクト配下に置く（例: `data/`）。

## 3. まず中央集権スモークテスト（最優先）

FLを載せる前に、データ読み込み+ResNet50+BERT+学習が回るか確認する。

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/iu_smoke_test.py \
    --reports data/indiana_reports.csv \
    --projections data/indiana_projections.csv \
    --images data/images/images_normalized \
    --subset 200 --epochs 1 --batch 8
```

期待される結果:
- `device = mps` と表示
- manifest 統計が出る（約3337サンプル、both views 約2943）
- train loss が表示され、わずかでも下がる
- test metrics（accuracy/hamming_acc/macro_f1/auroc/auprc）がエラーなく出る

**これが通れば、実データ基盤は正しい。** 通らなければ、エラーメッセージを共有してください
（MPS未対応演算、メモリ不足、トークナイザDLなど、典型的な初回エラーを一緒に潰します）。

## 4. ラベル設計（実装済み）

- `Problems` 列 → CheXpert 14クラスにマッピング（`data/iu_xray_labels.py`）。
- 入力テキストは `findings` セクションのみ（リーク回避、Saha 準拠）。
- findings 欠損は除外（3851 → 3337件）。
- マルチラベル（平均1.38ラベル/サンプル）。No Finding 51.6%。

## 5. モダリティ設計（case X）

- **画像モダリティ** = 正面 + 側面を1つに統合（両ビューの平均テンソル）。
- **テキストモダリティ** = findings（BERT）。
- ①②は両モダリティ揃い。③④で片方を欠かせる（`modalities=["image"]` 等）。

## 6. 次のステップ（スモーク通過後）

1. 連合用データローダ（マニフェスト → 拠点別 DataLoader、Dirichlet 非IID）を組む。
2. ①②を実データで（FedAvg/FedProx/FedMD）。
3. ③（UniFL/LOOT/MIN）、④（FedMD+MIN）。

## メモ리の目安（M5 32GB）

- batch 8〜16、224px、BERT max_len 256 が現実的な出発点。
- 連合学習では拠点を**逐次処理**（同時に1モデルだけGPUに載せる）してメモリを節約。
- OOM が出たら batch を下げる、max_len を 128 に、img_size を 192 に。

## ファイル一覧（実データ版）

```
src/qflbench/data/iu_xray_labels.py    Problems→CheXpert14クラス
src/qflbench/data/iu_xray_prep.py      マニフェスト構築・分割・非IID（torch不要）
src/qflbench/data/iu_xray_torch.py     torch Dataset（画像読込・BERTトークン化）
src/qflbench/models/iu_xray_torch_model.py  ResNet50+BERT+融合+ヘッド（ModelBackend準拠）
src/qflbench/metrics/classification.py multilabel_metrics 追加済み
scripts/iu_smoke_test.py               中央集権スモークテスト
```
