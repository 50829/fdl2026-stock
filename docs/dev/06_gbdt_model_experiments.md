# 06 GBDT / Ensemble Model Experiments

Date: 2026-05-30

## Goal

Run the tree-model branch proposed after the GRU experiments:

- LightGBM baseline.
- XGBoost baseline.
- A simple prediction-level weight ensemble over GRU, LightGBM, and XGBoost.

All experiments use the same main target as the current GRU baseline:

- Target: `label_5d__cs_rank`
- Raw return for 5-day top-k backtest: `label_5d`
- Daily return for rolling tranche backtest: `label_1d`

## Code Added

- `src/models/sdd/run_gbdt.py`
  - Loads processed tabular features and labels.
  - Trains `lightgbm` or `xgboost`.
  - Saves model, feature importance, valid/test predictions, IC metrics, top-k backtest, and rolling tranche backtest.

- `src/models/sdd/run_prediction_ensemble.py`
  - Merges saved predictions from GRU, LightGBM, and XGBoost.
  - Converts each model prediction to daily percentile rank.
  - Tests a weight grid and evaluates the same IC/backtest metrics.

Dependencies installed:

- `lightgbm==4.6.0`
- `xgboost-cpu==3.2.0`
- `scikit-learn==1.8.0`

## Commands

Pilot LightGBM:

```bash
python -m src.models.sdd.run_gbdt \
  --model lightgbm \
  --processed-dir data/processed_pilot \
  --out-root outputs/sdd_gbdt_pilot \
  --num-boost-round 1000 \
  --early-stopping-rounds 80 \
  --num-threads 8 \
  --log-period 50
```

Full LightGBM:

```bash
python -m src.models.sdd.run_gbdt \
  --model lightgbm \
  --processed-dir data/processed \
  --out-root outputs/sdd_gbdt_full \
  --num-boost-round 1500 \
  --early-stopping-rounds 100 \
  --num-threads 16 \
  --log-period 50
```

Pilot XGBoost:

```bash
python -m src.models.sdd.run_gbdt \
  --model xgboost \
  --processed-dir data/processed_pilot \
  --out-root outputs/sdd_gbdt_pilot \
  --num-boost-round 1000 \
  --early-stopping-rounds 80 \
  --num-threads 16 \
  --log-period 50 \
  --learning-rate 0.03 \
  --xgb-max-depth 6 \
  --xgb-min-child-weight 100
```

Full XGBoost:

```bash
python -m src.models.sdd.run_gbdt \
  --model xgboost \
  --processed-dir data/processed \
  --out-root outputs/sdd_gbdt_full \
  --num-boost-round 1200 \
  --early-stopping-rounds 100 \
  --num-threads 16 \
  --log-period 50 \
  --learning-rate 0.03 \
  --xgb-max-depth 6 \
  --xgb-min-child-weight 100
```

Prediction ensemble:

```bash
python -m src.models.sdd.run_prediction_ensemble \
  --out-root outputs/sdd_ensemble_full \
  --grid-step 0.25
```

## Main Results

### Pilot

| Model | Split | Samples | MSE | IC | ICIR | Top-k Return | Rolling Return |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | valid | 943810 | 0.327852 | 0.113883 | 0.645285 | 2.070836 | 0.997728 |
| LightGBM | test | 381587 | 0.326905 | 0.123661 | 0.838511 | 0.431298 | 0.662484 |
| XGBoost | valid | 943810 | 0.327683 | 0.115643 | 0.660977 | 1.500389 | 0.690373 |
| XGBoost | test | 381587 | 0.326798 | 0.123966 | 0.834106 | 0.369250 | 0.629033 |

Pilot conclusion: both GBDT models clearly have signal, and both exceed the existing GRU IC on the comparable horizon. XGBoost is slightly better on pilot IC, LightGBM is slightly better on pilot test ICIR and backtest.

### Full

| Model | Split | Samples | MSE | IC | ICIR | Top-k Return | Rolling Return |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | valid | 943810 | 0.326122 | 0.125789 | 0.761766 | 1.113027 | 1.552527 |
| LightGBM | test | 1280375 | 0.328445 | 0.107852 | 0.805562 | 4.499941 | 3.932256 |
| XGBoost | valid | 943810 | 0.326166 | 0.125502 | 0.776915 | 2.763909 | 2.260177 |
| XGBoost | test | 1280375 | 0.328438 | 0.106861 | 0.794579 | 21.121385 | 6.990298 |
| GRU layer1 | valid | 766867 | 0.327678 | 0.095154 | 0.511638 | 0.126012 | -0.054890* |
| GRU layer1 | test | 1019298 | 0.327920 | 0.085123 | 0.648694 | 0.358659 | 0.875401* |

`*` GRU rolling returns are from `outputs/sdd_rolling_tranche_eval`, because the original GRU metrics file did not include rolling metrics.

Full conclusion:

- LightGBM and XGBoost both beat the current GRU on IC/ICIR.
- XGBoost has the best valid ICIR (`0.776915`), while LightGBM has slightly better test IC/ICIR (`0.107852 / 0.805562` vs `0.106861 / 0.794579`).
- Backtest return is much more sensitive to ranking tails and turnover than IC. XGBoost test top-k return is very high, but this should not be interpreted as a stable superiority without additional turnover/hold-days sensitivity.

## Ensemble Result

The ensemble grid used daily rank-normalized predictions and weights in increments of `0.25`.

Valid best by ICIR:

- `w_lightgbm=0.0`
- `w_xgboost=1.0`
- `w_gru=0.0`
- IC `0.115881`, ICIR `0.689124` on the common GRU/GBDT sample set.

Applying the valid-selected weight to test:

- `w_lightgbm=0.0`
- `w_xgboost=1.0`
- `w_gru=0.0`
- test IC `0.094145`, ICIR `0.636387`
- test top-k return `2.183343`
- test rolling return `2.492439`

Test hindsight best by ICIR:

- `w_lightgbm=0.5`
- `w_xgboost=0.0`
- `w_gru=0.5`
- test IC `0.095484`, ICIR `0.654287`

This hindsight result is not a valid model-selection result. The usable conclusion is that the weight ensemble did not provide a stable validation-selected gain over the standalone GBDT models.

## Feature Importance

Top LightGBM gain features:

1. `turnover_rate__cs_rank`
2. `log_amount__cs_rank`
3. `turnover_mean_5__cs_rank`
4. `volatility_60__cs_rank`
5. `buy_lg_amount_ratio__cs_rank`
6. `momentum_5__ts_z60`
7. `macd_dif__cs_rank`
8. `close_vwap_gap__cs_rank`
9. `log_total_mv__cs_robust_z`
10. `momentum_20__cs_rank`

Top XGBoost gain features:

1. `turnover_rate__cs_rank`
2. `turnover_mean_5__cs_rank`
3. `log_amount__cs_rank`
4. `momentum_20__cs_rank`
5. `volume_ratio_60__cs_rank`
6. `macd_dif__missing`
7. `open_gap__cs_rank`
8. `volatility_60__cs_rank`
9. `stock_minus_industry_mom_20__cs_rank`
10. `close_vwap_gap__cs_rank`

The common pattern is that turnover/liquidity, short-to-medium momentum, volatility, and price-volume interaction dominate the tree models.

## Current Recommendation

Use full LightGBM and full XGBoost as the main non-neural baselines. For reporting, prioritize IC and ICIR:

- LightGBM full test: IC `0.107852`, ICIR `0.805562`.
- XGBoost full test: IC `0.106861`, ICIR `0.794579`.
- Current GRU layer1 full test: IC `0.085123`, ICIR `0.648694`.

Do not claim the simple prediction ensemble as an improvement. The next useful experiment is a proper neural-output-as-feature model using out-of-fold GRU train predictions, or a GRU trained on the top GBDT-selected features. The simple in-sample neural-output feature variant is not recommended because it can leak GRU train overfit into the GBDT training matrix.
