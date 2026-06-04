# Label1d Model and Strategy Comparison

Date: 2026-06-01

## Goal

Train a parallel model family using `label_1d__cs_rank` and compare it with the current `label_5d__cs_rank` final model under the same strategy code.

The comparison answers two questions:

1. Whether a 1-day target is better for short-horizon trading.
2. Whether it should replace the current `rolling_p10_h5` live strategy.

## Data and Labels

Processed data:

- Features: `data/processed/features.parquet`
- Labels: `data/processed/labels.parquet`
- Universe: `data/processed/universe.parquet`

Existing final model target:

- `label_5d__cs_rank`
- Raw return for top-k style evaluation: `label_5d`
- Daily return for rolling tranche: `label_1d`

New model target:

- `label_1d__cs_rank`
- Raw return and daily return: `label_1d`

The label definition remains point-in-time safe:

```text
decision at T close
buy at T+1 close/reference
label_1d[T] = close[T+2] / close[T+1] - 1
```

## Models Trained

Output root:

- `outputs/models/label1d_top40_20260601`
- `outputs/models/label1d_fusion_top40_20260601`

Base models:

| model | target | features | output |
| --- | --- | ---: | --- |
| LightGBM | `label_1d__cs_rank` | top40 | `outputs/models/label1d_top40_20260601/lightgbm` |
| XGBoost | `label_1d__cs_rank` | top40 | `outputs/models/label1d_top40_20260601/xgboost` |

Fusion:

- Method: residual-rank MLP over LightGBM/XGBoost meta features.
- OOF train years: 2021, 2022, 2023.
- Valid split: 2024.
- Test split: 2025-01-02 to 2026-05-08.
- Valid-selected best alpha: `0.0`.

Conclusion: for `label_1d`, residual-rank MLP did not improve validation performance. The selected fusion model collapses to LightGBM.

## Model Metrics

| model | split | IC mean | ICIR | rolling return | rolling Sharpe | max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| label1d LightGBM | valid | 0.0834 | 0.5601 | 0.2681 | 0.7622 | -0.3859 |
| label1d LightGBM | test | 0.0818 | 0.6173 | 3.3419 | 4.0114 | -0.1742 |
| label1d XGBoost | valid | 0.0823 | 0.5480 | 0.4148 | 0.9931 | -0.4044 |
| label1d XGBoost | test | 0.0816 | 0.6012 | 3.6296 | 4.1966 | -0.1574 |
| label1d residual-rank MLP | valid | 0.0834 | 0.5601 | 0.2681 | 0.7622 | -0.3859 |
| label1d residual-rank MLP | test | 0.0818 | 0.6173 | 3.3419 | 4.0114 | -0.1742 |
| label5d final residual-rank MLP | test | 0.1068 | 0.8215 | 4.3305 | 6.0133 | -0.0696 |

The `label5d` final model still has better test IC and better rolling tranche risk-adjusted performance.

## Strategy Backtest

Output root:

- `outputs/strategy/label1d_vs_label5d_20260601`

Splits:

- Valid: 2024.
- Test: 2025-01-02 to 2026-05-08.

Compared models:

- `label5d_final`
- `label1d_lgb`
- `label1d_xgb`
- `label1d_fusion_valid_alpha` (`alpha=0`, same as label1d LightGBM)

Compared strategies:

- `rolling_p10_h5`
- `rolling_p20_h3`
- `rolling_p20_h5`
- `rolling_p20_h10`
- `topk20_drop2`
- `topk20_drop3`
- `rankbuf_p20_b30_s100_min2_max10`
- `rankbuf_p20_b50_s100_min2_max10`

Metrics:

- `outputs/strategy/label1d_vs_label5d_20260601/strategy_metrics_all.csv`
- `outputs/strategy/label1d_vs_label5d_20260601/valid/strategy_metrics.csv`
- `outputs/strategy/label1d_vs_label5d_20260601/test/strategy_metrics.csv`

Plots:

- `outputs/strategy/label1d_vs_label5d_20260601/valid/equity_overview.svg`
- `outputs/strategy/label1d_vs_label5d_20260601/test/equity_overview.svg`
- `outputs/strategy/label1d_vs_label5d_20260601/valid/equity_key_model_strategy_matrix.svg`
- `outputs/strategy/label1d_vs_label5d_20260601/test/equity_key_model_strategy_matrix.svg`
- `outputs/strategy/label1d_vs_label5d_20260601/valid/plots_by_family/`
- `outputs/strategy/label1d_vs_label5d_20260601/test/plots_by_family/`
- `outputs/strategy/label1d_vs_label5d_20260601/valid/plots_by_strategy/`
- `outputs/strategy/label1d_vs_label5d_20260601/test/plots_by_strategy/`

Plot style was refreshed after the initial version made every `model__strategy` line fall into the `other` family. The revised charts use:

- `label5d_final`: black
- `label1d_lgb`: orange
- `label1d_xgb`: green
- `label1d_fusion_valid_alpha`: magenta

Strategy variants are separated by line style. The clearest files for model comparison are the per-strategy charts under `plots_by_strategy/`, because each chart fixes one strategy and varies only the model.

## Strategy Results

Valid top results by Sharpe:

| split | model | strategy | total return | annual return | Sharpe | max drawdown | turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| valid | label1d_lgb | rankbuf_p20_b30_s100_min2_max10 | 2.8180 | 3.0353 | 3.3217 | -0.3791 | 0.8128 |
| valid | label1d_lgb | rankbuf_p20_b50_s100_min2_max10 | 2.8180 | 3.0353 | 3.3217 | -0.3791 | 0.8128 |
| valid | label1d_xgb | rankbuf_p20_b30_s100_min2_max10 | 2.3985 | 2.5747 | 3.0247 | -0.3980 | 0.8041 |
| valid | label1d_lgb | topk20_drop3 | 1.9981 | 2.1372 | 2.8252 | -0.3595 | 0.5017 |
| valid | label5d_final | rolling_p20_h5 | 1.1690 | 1.2395 | 2.6213 | -0.3144 | 0.3835 |
| valid | label5d_final | rolling_p10_h5 | 1.0758 | 1.1394 | 2.4308 | -0.3207 | 0.3901 |

Test top results by Sharpe:

| split | model | strategy | total return | annual return | Sharpe | max drawdown | turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | label1d_lgb | topk20_drop3 | 7.3751 | 4.2493 | 6.7488 | -0.1246 | 0.5279 |
| test | label1d_lgb | rankbuf_p20_b50_s100_min2_max10 | 7.9520 | 4.5294 | 6.6350 | -0.1089 | 0.7994 |
| test | label1d_xgb | topk20_drop3 | 7.7470 | 4.4303 | 6.6282 | -0.1261 | 0.5625 |
| test | label1d_lgb | rankbuf_p20_b30_s100_min2_max10 | 7.8531 | 4.4816 | 6.6119 | -0.1089 | 0.7991 |
| test | label1d_lgb | rolling_p20_h3 | 5.5183 | 3.3170 | 6.1911 | -0.1227 | 0.6457 |
| test | label5d_final | rolling_p20_h3 | 3.5345 | 2.2632 | 5.9817 | -0.0852 | 0.6486 |
| test | label5d_final | rolling_p10_h5 | 2.6823 | 1.7998 | 5.8310 | -0.0653 | 0.3861 |

Observations:

- `label1d` is competitive for high-refresh strategies such as `topk20_drop3` and `rank_buffer`.
- `label1d` has higher turnover and generally larger drawdowns.
- `label5d_final` remains better for the current `rolling_p10_h5` strategy, especially on drawdown control.
- The valid/test behavior is not fully stable: `label1d` looks much weaker in `rolling_p10_h5` on valid, but very strong in high-turnover strategies on test. Treat this as a new candidate, not as an immediate replacement.

## 2026-06-02 Live Plan Comparison

Output root:

- `outputs/live/label1d_vs_label5d_20260602_from_20260601`

Source features:

- `outputs/live/rolling_p10_h5_20260602_from_20260601_20260601_220111/live_features_20260601.parquet`

Live candidate count: `4142`.

Top10 overlap:

- Count: `3`
- Overlap: `300030.SZ`, `603628.SH`, `688021.SH`

Label5d final Top2:

| rank | ts_code | industry | close | pred |
| ---: | --- | --- | ---: | ---: |
| 1 | 000576.SZ | 电气设备 | 7.65 | 0.3724 |
| 2 | 300030.SZ | 医疗保健 | 7.09 | 0.3568 |

Label1d LightGBM Top2:

| rank | ts_code | industry | close | pred |
| ---: | --- | --- | ---: | ---: |
| 1 | 000010.SZ | 建筑工程 | 2.21 | 0.2073 |
| 2 | 603628.SH | 电气设备 | 11.08 | 0.1910 |

Strict rolling tranche turnover if switching from label5d Top2 to label1d Top2:

- Top2 overlap: `0`
- Portfolio turnover: `20%`
- Sell: `000576.SZ`, `300030.SZ`
- Buy: `000010.SZ`, `603628.SH`

## Recommendation

Do not replace the current live `rolling_p10_h5` strategy with `label1d` yet.

Current practical recommendation:

- Keep `label5d_final + rolling_p10_h5` as the conservative live strategy.
- Track `label1d_lgb + topk20_drop3` and `label1d_lgb + rank_buffer` as experimental high-turnover alternatives.
- If using `label1d`, add stronger turnover/risk constraints before live deployment, because its best strategies rely on frequent replacement.

For 2026-06-02 specifically:

- Conservative plan: use the existing label5d Top2: `000576.SZ`, `300030.SZ`.
- Experimental label1d plan: `000010.SZ`, `603628.SH`.
- Because the strict Top2 overlap is zero, switching signals introduces immediate churn without enough validation stability.

## Implementation Notes

The `src.models.sdd.*` scripts expose `run_cli()` but do not execute it under `python -m` because they lack a module-level `if __name__ == "__main__": run_cli()`. For this experiment I invoked the underlying functions directly.

During initial label1d base-model runs, `raw_return_col == daily_return_col == label_1d` caused duplicate columns when writing parquet predictions. The models had already been saved, so prediction files and summaries were repaired by reloading the saved models and writing deduplicated prediction frames.
