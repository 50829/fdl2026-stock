# 06 GBDT 与集成模型实验

日期：2026-05-30

## 实验目标

在 GRU 实验后，补跑树模型分支：

- LightGBM 基线。
- XGBoost 基线。
- 对 GRU、LightGBM、XGBoost 做简单的预测层加权集成。

所有实验都使用与当前 GRU 基线一致的主目标：

- 目标列：`label_5d__cs_rank`
- 5 日 top-k 回测使用的原始收益列：`label_5d`
- rolling tranche 回测使用的日收益列：`label_1d`

## 新增代码

- `src/model_experiments/run_gbdt.py`
  - 读取处理后的表格特征和标签。
  - 训练 `lightgbm` 或 `xgboost`。
  - 保存模型、特征重要性、valid/test 预测、IC 指标、top-k 回测和 rolling tranche 回测。

- `src/model_experiments/run_prediction_ensemble.py`
  - 合并 GRU、LightGBM、XGBoost 已保存的预测。
  - 将每个模型的预测转换为每日截面百分位排名。
  - 遍历权重网格，并计算同一套 IC/回测指标。

已安装依赖：

- `lightgbm==4.6.0`
- `xgboost-cpu==3.2.0`
- `scikit-learn==1.8.0`

## 运行命令

LightGBM 小样本：

```bash
python -m src.model_experiments.run_gbdt \
  --model lightgbm \
  --processed-dir data/processed_pilot \
  --out-root outputs/models/20260530_200657__gbdt_pilot \
  --num-boost-round 1000 \
  --early-stopping-rounds 80 \
  --num-threads 8 \
  --log-period 50
```

LightGBM 全量：

```bash
python -m src.model_experiments.run_gbdt \
  --model lightgbm \
  --processed-dir data/processed \
  --out-root outputs/models/20260530_200734__gbdt_full \
  --num-boost-round 1500 \
  --early-stopping-rounds 100 \
  --num-threads 16 \
  --log-period 50
```

XGBoost 小样本：

```bash
python -m src.model_experiments.run_gbdt \
  --model xgboost \
  --processed-dir data/processed_pilot \
  --out-root outputs/models/20260530_200657__gbdt_pilot \
  --num-boost-round 1000 \
  --early-stopping-rounds 80 \
  --num-threads 16 \
  --log-period 50 \
  --learning-rate 0.03 \
  --xgb-max-depth 6 \
  --xgb-min-child-weight 100
```

XGBoost 全量：

```bash
python -m src.model_experiments.run_gbdt \
  --model xgboost \
  --processed-dir data/processed \
  --out-root outputs/models/20260530_200734__gbdt_full \
  --num-boost-round 1200 \
  --early-stopping-rounds 100 \
  --num-threads 16 \
  --log-period 50 \
  --learning-rate 0.03 \
  --xgb-max-depth 6 \
  --xgb-min-child-weight 100
```

预测集成：

```bash
python -m src.model_experiments.run_prediction_ensemble \
  --out-root outputs/models/20260530_201321__ensemble_full \
  --grid-step 0.25
```

## 主要结果

### 小样本

| 模型 | 数据集 | 样本数 | MSE | IC | ICIR | Top-k 收益 | Rolling 收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | valid | 943810 | 0.327852 | 0.113883 | 0.645285 | 2.070836 | 0.997728 |
| LightGBM | test | 381587 | 0.326905 | 0.123661 | 0.838511 | 0.431298 | 0.662484 |
| XGBoost | valid | 943810 | 0.327683 | 0.115643 | 0.660977 | 1.500389 | 0.690373 |
| XGBoost | test | 381587 | 0.326798 | 0.123966 | 0.834106 | 0.369250 | 0.629033 |

小样本结论：两个 GBDT 模型都有明确的信号，并且在可比周期上都超过已有 GRU 的 IC。XGBoost 的小样本 IC 略好，LightGBM 的小样本 test ICIR 和回测略好。

### 全量

| 模型 | 数据集 | 样本数 | MSE | IC | ICIR | Top-k 收益 | Rolling 收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | valid | 943810 | 0.326122 | 0.125789 | 0.761766 | 1.113027 | 1.552527 |
| LightGBM | test | 1280375 | 0.328445 | 0.107852 | 0.805562 | 4.499941 | 3.932256 |
| XGBoost | valid | 943810 | 0.326166 | 0.125502 | 0.776915 | 2.763909 | 2.260177 |
| XGBoost | test | 1280375 | 0.328438 | 0.106861 | 0.794579 | 21.121385 | 6.990298 |
| GRU layer1 | valid | 766867 | 0.327678 | 0.095154 | 0.511638 | 0.126012 | -0.054890* |
| GRU layer1 | test | 1019298 | 0.327920 | 0.085123 | 0.648694 | 0.358659 | 0.875401* |

`*` GRU rolling 收益来自 `outputs/models/20260530_194341__rolling_tranche_eval`，因为原始 GRU 指标文件没有包含 rolling 指标。

全量结论：

- LightGBM 和 XGBoost 在 IC/ICIR 上都超过当前 GRU。
- XGBoost 的 valid ICIR 最高（`0.776915`），LightGBM 的 test IC/ICIR 略好（`0.107852 / 0.805562`，对比 XGBoost 的 `0.106861 / 0.794579`）。
- 回测收益相比 IC 更受尾部排名和换手影响。XGBoost 的 test top-k 收益很高，但在补充换手和持有天数敏感性分析前，不能解读为稳定优势。

## 集成结果

集成网格使用每日截面排名归一化后的预测，权重步长为 `0.25`。

按 ICIR 选择的 valid 最优权重：

- `w_lightgbm=0.0`
- `w_xgboost=1.0`
- `w_gru=0.0`
- 在 GRU/GBDT 共同样本集上，IC 为 `0.115881`，ICIR 为 `0.689124`。

将 valid 选出的权重应用到 test：

- `w_lightgbm=0.0`
- `w_xgboost=1.0`
- `w_gru=0.0`
- test IC 为 `0.094145`，ICIR 为 `0.636387`
- test top-k 收益为 `2.183343`
- test rolling 收益为 `2.492439`

按 ICIR 事后选择的 test 最优权重：

- `w_lightgbm=0.5`
- `w_xgboost=0.0`
- `w_gru=0.5`
- test IC 为 `0.095484`，ICIR 为 `0.654287`

这个事后结果不能作为有效的模型选择依据。可用结论是：简单权重集成没有提供一个能由验证集稳定选出的、超过单独 GBDT 模型的增益。

## 特征重要性

LightGBM gain 排名前列的特征：

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

XGBoost gain 排名前列的特征：

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

共同模式是：换手/流动性、短中期动量、波动率和量价交互特征主导树模型。

## 当前建议

使用全量 LightGBM 和全量 XGBoost 作为主要非神经网络基线。报告中优先展示 IC 和 ICIR：

- LightGBM 全量 test：IC `0.107852`，ICIR `0.805562`。
- XGBoost 全量 test：IC `0.106861`，ICIR `0.794579`。
- 当前 GRU layer1 全量 test：IC `0.085123`，ICIR `0.648694`。

不要把简单预测集成声称为改进。下一步更有价值的实验是：使用 out-of-fold GRU 训练期预测，把神经网络输出作为特征；或者在 GBDT 选出的 top 特征上训练 GRU。不建议做简单的样本内神经网络输出特征版本，因为这可能把 GRU 的训练集过拟合泄漏进 GBDT 训练矩阵。
