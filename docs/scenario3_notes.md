# シナリオ3（モダリティ異種）実装メモ

## 何を実装したか（Saha 2024 に忠実）

シナリオ3＝「同じモデル・異なるモダリティ部分集合」を、Saha論文（arXiv:2402.05294）の
設定に忠実に実装した。4手法を用意。

| 手法 | 由来 | 何をするか | 通信 |
|---|---|---|---|
| **UniFL** | Saha参照線 | 全クライアントが画像のみ（ユニモーダルFL）。これを超えられるかが問い | 小 |
| **FedAvg** | 不一致ベースライン | 一部マルチモーダル+残りユニモーダルで素のFedAvg | 小 |
| **LOOT** | Saha 7.2節 | サーバ側で各モデルの埋め込みを他モデル平均へ整列（数値最強） | 大 |
| **MIN+FedAvg** | Saha 6章 | 事前に欠損モダリティを補完してから FedAvg | 小 |

## 設計判断（重要）

- **「同じモデル」の実装**: Saha忠実版として `all_modalities=True`。全クライアントが
  全モダリティのエンコーダを持つ（フルモデル）。ユニモーダルクライアントは欠損
  モダリティを学習できず、そのエンコーダが平均を希釈する＝不一致の害（Saha 4章）。
  モデル全体を共有・集約（`share_encoders=True`）。
- **モダリティ割当**: `mixed:K`（先頭K個がマルチモーダル、残りは画像のみ）で M:U 比率を表現。
  `--num-multimodal` で K を指定。`uni` は全員画像のみ（UniFL参照線）。
- **LOOT**: 論文の「leave-one-out teacher（1つを生徒、残りK-1を教師に埋め込み類似度最大化）」を、
  公開データ経由の broadcast→fit→aggregate プロトコルとして実装。**整列→ローカル学習の順**
  （逆だとhead不整合で崩壊する。修正済み）。埋め込み交換を通信量に計上。
- **MIN-lite**: 論文のMINはVQ-GAN+BERTでレポート生成する重い構成。ここでは本質
  （マルチモーダルclientで翻訳器を事前学習→ユニモーダルclientで欠損補完→擬似congruentに変換）
  を、**特徴レベルの線形翻訳（リッジ回帰）**で実装。生成機構はtorch/IU X-ray移行時に本格化。

## 動作確認済みの結果（合成, α=0.2, 12ラウンド, 2seed）

| 手法 | final_acc | UniFL比 |
|---|---|---|
| UniFL | 0.449 | (参照線) |
| FedAvg | 0.683 | +0.235 |
| LOOT | 0.665 | +0.217 |
| MIN+FedAvg | 0.750 | +0.301 |

※短ラウンドの暫定値。傾向確認用。

## 明日の実験手順

```bash
# ③をα別に（全部 results/scenario3/ に蓄積、上書きされない）
python scripts/run_benchmark.py --scenario 3 --alpha 0.1 --rounds 40 --seeds 3 --num-multimodal 2
python scripts/run_benchmark.py --scenario 3 --alpha 0.2 --rounds 40 --seeds 3 --num-multimodal 2
python scripts/run_benchmark.py --scenario 3 --alpha 0.5 --rounds 40 --seeds 3 --num-multimodal 2
python scripts/run_benchmark.py --scenario 3 --alpha 1.0 --rounds 40 --seeds 3 --num-multimodal 2

# M:U比率を変えた追加実験（Sahaは1:3と3:1を比較）
python scripts/run_benchmark.py --scenario 3 --alpha 0.2 --rounds 40 --seeds 3 --num-multimodal 1  # M:U=1:5寄り
python scripts/run_benchmark.py --scenario 3 --alpha 0.2 --rounds 40 --seeds 3 --num-multimodal 4  # M:U=4:2

# プロット再生成
python scripts/run_benchmark.py --plot-only --scenario 3
```

## 確認すべきこと（Sahaとの照合）

1. **強い非IID（α=0.1）で、不一致FedAvgがUniFLを下回るか？**
   Sahaの中心的発見「非IIDでは不一致MMFL < ユニモーダルFL」が再現されるか。
   → されれば「古典で埋めきれない穴」を確認。QFLの標的。
2. **LOOTとMINがその穴をどこまで埋めるか。** Sahaでは部分的に埋めるが、
   強い非IIDでは完全には埋まらない。
3. **②と③で穴の性質が違うか。** ②=収束コスト、③=性能上限、という対比。

## 既知の注意点

- FedProtoの overflow 警告と同様、LOOTの整列lrが大きすぎると崩壊する（0.01で安定）。
- MIN-liteは特徴レベル補完なので、Saha本来のレポート生成とは質が異なる。
  数値の絶対値ではなく「補完が不一致を緩和する」傾向の確認に使う。
- gap_filled は合成タスクでは依然 unstable（IU X-rayで有効化）。
