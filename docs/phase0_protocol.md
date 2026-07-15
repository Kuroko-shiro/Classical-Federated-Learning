# Phase 0: 評価基盤の固定と古典ベンチマーク再確定

更新日: 2026-07-15

Phase 0の目的は、low-rank/QFL実装へ進む前に、IU X-ray古典ベンチマークを
査読に耐える評価手順へ固定することである。旧test trajectoryの最大値は主結果に使わない。

## 固定事項

| 項目 | 値 |
|---|---|
| split | `splits/iu_split.json`, split seed 0 |
| public | 200件。FedMD/LOOTのみに利用 |
| train pool | 2,510件 |
| validation | 各client partitionの10%、seed `20260715` |
| test | 627件すべて |
| clients | 4 |
| rounds/local epochs | 40 / 2 |
| optimizer | AdamW, lr `1e-4` |
| centralized | 40 epochs, lr `3e-5` |
| selection | validation macro-AUROC最大 |
| primary result | 選択checkpointのtest macro-AUROC |
| seeds | 0, 1, 2（mean ± SD、95% CI） |

validationは既存のclient partitionから決定的に導出する。既定10%の場合、
train 2,259件、validation 251件となる。public/testとの重複は起動時に検査する。

## 出力

各runは `results/iu/<run_name>_<timestamp>/` に次を保存する。

- `config.json`: 全引数、git commit、split SHA-256、各集合の件数
- `validation.csv`: round別validation指標
- `communication.jsonl`: round・方向・client別の実payload bytes
- `best_validation.npz`: validationで選択したモデル
- `test.json`: test一回評価の結果と選択round

## 実行順序

共通引数を以下とする。

```bash
COMMON="--reports data/indiana_reports.csv \
--projections data/indiana_projections.csv \
--images data/images/images_normalized \
--img-cache data/img_cache_224.pt --split splits/iu_split.json \
--clients 4 --rounds 40 --train-subset 2510 --test-subset 627 \
--val-fraction 0.1 --val-seed 20260715 \
--local-epochs 2 --batch 8 --num-workers 2"
```

### P−1: 共通評価による主要比較

各コマンドは `seed=0,1,2` で実行する。

```bash
for SEED in 0 1 2; do
  caffeinate -dimsu python scripts/iu_federated.py $COMMON \
    --scenario 1 --method fedavg --alpha 100 --seed $SEED

  caffeinate -dimsu python scripts/iu_federated.py $COMMON \
    --scenario 2 --method heterofl --alpha 100 --seed $SEED

  caffeinate -dimsu python scripts/iu_federated_s4.py $COMMON \
    --method fedmd --mm-ratio 1:3 --alpha 100 --seed $SEED

  caffeinate -dimsu python scripts/iu_federated_s4.py $COMMON \
    --method heterofl --mm-ratio 1:3 --alpha 100 --seed $SEED

  caffeinate -dimsu python scripts/iu_federated_s4.py $COMMON \
    --method heterofl --mm-ratio 1:3 --alpha 0.1 --seed $SEED
done
```

### P0: centralized上限

```bash
for SEED in 0 1 2; do
  caffeinate -dimsu python scripts/iu_baselines.py \
    --reports data/indiana_reports.csv \
    --projections data/indiana_projections.csv \
    --images data/images/images_normalized \
    --img-cache data/img_cache_224.pt --split splits/iu_split.json \
    --mode centralized --clients 4 --alpha 100 --epochs 40 --eval-every 1 \
    --lr 3e-5 --train-subset 2510 --test-subset 627 \
    --val-fraction 0.1 --val-seed 20260715 --batch 8 --seed $SEED
done
```

Local下限は対象条件ごとに取得する。α、幅異種、モダリティ比を使い回さない。

```bash
python scripts/iu_baselines.py ... --mode local --alpha 100 \
  --vary-embed --mm-ratio 1:3 --epochs 80 --eval-every 2 --seed 0
```

### 集計

```bash
python scripts/iu_summarize_runs.py \
  --results-root results/iu --output results/iu/phase0_summary.csv
```

保存済みPhase 0 checkpointを別実行で再評価する場合:

```bash
python scripts/iu_evaluate_checkpoint.py \
  --checkpoint results/iu/<run>/best_validation.npz \
  --reports data/indiana_reports.csv \
  --projections data/indiana_projections.csv \
  --images data/images/images_normalized \
  --img-cache data/img_cache_224.pt --test-subset 627
```

## Definition of Done

- 主要比較が3 seedsすべて完走している
- 全runでtest sizeが627、split hashが同一である
- best roundがvalidationだけで選択されている
- testは各seed・各runにつき1回だけである
- upload/download、client別、累積byteが欠損していない
- centralized `C` と各条件のLocal下限が確定している
- `phase0_summary.csv` にmean、SD、95% CIが出力されている
- 旧peak/conv値は「historical/補助解析」と明示されている

ここまでを満たした後、Phase 1のbackbone-wide low-rank（同一rank/異種rank）へ進む。
