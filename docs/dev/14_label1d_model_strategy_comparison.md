# 1 日标签模型与策略对比

日期：2026-06-01

## 目标

使用 `label_1d__cs_rank` 训练一组平行模型，并在同一套策略代码下与当前 `label_5d__cs_rank` 最终模型对比。

本对比回答两个问题：

1. 1 日目标是否更适合短周期交易。
2. 它是否应该替代当前的 `rolling_p10_h5` 实盘策略。

## 数据与标签

处理后数据：

- 特征：`data/processed/features.parquet`
- 标签：`data/processed/labels.parquet`
- 股票池：`data/processed/universe.parquet`

现有最终模型目标：

- `label_5d__cs_rank`
- top-k 风格评估使用的原始收益列：`label_5d`
- rolling tranche 使用的日收益列：`label_1d`

新模型目标：

- `label_1d__cs_rank`
- 原始收益列和日收益列都使用：`label_1d`

标签定义保持时点安全：

```text
T 日收盘后决策
T+1 收盘价或参考价买入
label_1d[T] = close[T+2] / close[T+1] - 1
```

## 已训练模型

输出根目录：

- `outputs/models/label1d_top40_20260601`
- `outputs/models/label1d_fusion_top40_20260601`

基础模型：

| 模型 | 目标 | 特征 | 输出 |
| --- | --- | ---: | --- |
| LightGBM | `label_1d__cs_rank` | top40 | `outputs/models/label1d_top40_20260601/lightgbm` |
| XGBoost | `label_1d__cs_rank` | top40 | `outputs/models/label1d_top40_20260601/xgboost` |

融合模型：

- 方法：在 LightGBM/XGBoost meta 特征上训练 residual-rank MLP。
- OOF 训练年份：2021、2022、2023。
- 验证集切分：2024。
- 测试集切分：2025-01-02 至 2026-05-08。
- 验证集选择的最佳 alpha：`0.0`。

结论：对于 `label_1d`，residual-rank MLP 没有改善验证集表现。被选中的融合模型退化为 LightGBM。

## 模型指标

| 模型 | 数据集 | IC 均值 | ICIR | rolling 收益 | rolling Sharpe | 最大回撤 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| label1d LightGBM | valid | 0.0834 | 0.5601 | 0.2681 | 0.7622 | -0.3859 |
| label1d LightGBM | test | 0.0818 | 0.6173 | 3.3419 | 4.0114 | -0.1742 |
| label1d XGBoost | valid | 0.0823 | 0.5480 | 0.4148 | 0.9931 | -0.4044 |
| label1d XGBoost | test | 0.0816 | 0.6012 | 3.6296 | 4.1966 | -0.1574 |
| label1d residual-rank MLP | valid | 0.0834 | 0.5601 | 0.2681 | 0.7622 | -0.3859 |
| label1d residual-rank MLP | test | 0.0818 | 0.6173 | 3.3419 | 4.0114 | -0.1742 |
| label5d final residual-rank MLP | test | 0.1068 | 0.8215 | 4.3305 | 6.0133 | -0.0696 |

`label5d` 最终模型仍然有更好的 test IC，以及更好的 rolling tranche 风险调整后表现。

## 策略回测

输出根目录：

- `outputs/strategy/label1d_vs_label5d_20260601`

数据切分：

- 验证集：2024。
- 测试集：2025-01-02 至 2026-05-08。

对比模型：

- `label5d_final`
- `label1d_lgb`
- `label1d_xgb`
- `label1d_fusion_valid_alpha` (`alpha=0`, same as label1d LightGBM)

对比策略：

- `rolling_p10_h5`
- `rolling_p20_h3`
- `rolling_p20_h5`
- `rolling_p20_h10`
- `topk20_drop2`
- `topk20_drop3`
- `rankbuf_p20_b30_s100_min2_max10`
- `rankbuf_p20_b50_s100_min2_max10`

指标文件：

- `outputs/strategy/label1d_vs_label5d_20260601/strategy_metrics_all.csv`
- `outputs/strategy/label1d_vs_label5d_20260601/valid/strategy_metrics.csv`
- `outputs/strategy/label1d_vs_label5d_20260601/test/strategy_metrics.csv`

图表文件：

- `outputs/strategy/label1d_vs_label5d_20260601/valid/equity_overview.svg`
- `outputs/strategy/label1d_vs_label5d_20260601/test/equity_overview.svg`
- `outputs/strategy/label1d_vs_label5d_20260601/valid/equity_key_model_strategy_matrix.svg`
- `outputs/strategy/label1d_vs_label5d_20260601/test/equity_key_model_strategy_matrix.svg`
- `outputs/strategy/label1d_vs_label5d_20260601/valid/plots_by_family/`
- `outputs/strategy/label1d_vs_label5d_20260601/test/plots_by_family/`
- `outputs/strategy/label1d_vs_label5d_20260601/valid/plots_by_strategy/`
- `outputs/strategy/label1d_vs_label5d_20260601/test/plots_by_strategy/`

初版图表把所有 `model__strategy` 曲线都归到 `other` 族，因此后来刷新了绘图风格。新版图表使用：

- `label5d_final`: black
- `label1d_lgb`: orange
- `label1d_xgb`: green
- `label1d_fusion_valid_alpha`: magenta

策略变体通过线型区分。最清楚的模型对比文件是 `plots_by_strategy/` 下的单策略图，因为每张图固定一个策略，只改变模型。

## 策略结果

验证集上按 Sharpe 排名靠前的结果：

| 数据集 | 模型 | 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 换手率 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| valid | label1d_lgb | rankbuf_p20_b30_s100_min2_max10 | 2.8180 | 3.0353 | 3.3217 | -0.3791 | 0.8128 |
| valid | label1d_lgb | rankbuf_p20_b50_s100_min2_max10 | 2.8180 | 3.0353 | 3.3217 | -0.3791 | 0.8128 |
| valid | label1d_xgb | rankbuf_p20_b30_s100_min2_max10 | 2.3985 | 2.5747 | 3.0247 | -0.3980 | 0.8041 |
| valid | label1d_lgb | topk20_drop3 | 1.9981 | 2.1372 | 2.8252 | -0.3595 | 0.5017 |
| valid | label5d_final | rolling_p20_h5 | 1.1690 | 1.2395 | 2.6213 | -0.3144 | 0.3835 |
| valid | label5d_final | rolling_p10_h5 | 1.0758 | 1.1394 | 2.4308 | -0.3207 | 0.3901 |

测试集上按 Sharpe 排名靠前的结果：

| 数据集 | 模型 | 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 换手率 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| test | label1d_lgb | topk20_drop3 | 7.3751 | 4.2493 | 6.7488 | -0.1246 | 0.5279 |
| test | label1d_lgb | rankbuf_p20_b50_s100_min2_max10 | 7.9520 | 4.5294 | 6.6350 | -0.1089 | 0.7994 |
| test | label1d_xgb | topk20_drop3 | 7.7470 | 4.4303 | 6.6282 | -0.1261 | 0.5625 |
| test | label1d_lgb | rankbuf_p20_b30_s100_min2_max10 | 7.8531 | 4.4816 | 6.6119 | -0.1089 | 0.7991 |
| test | label1d_lgb | rolling_p20_h3 | 5.5183 | 3.3170 | 6.1911 | -0.1227 | 0.6457 |
| test | label5d_final | rolling_p20_h3 | 3.5345 | 2.2632 | 5.9817 | -0.0852 | 0.6486 |
| test | label5d_final | rolling_p10_h5 | 2.6823 | 1.7998 | 5.8310 | -0.0653 | 0.3861 |

观察：

- `label1d` 在 `topk20_drop3`、`rank_buffer` 等高刷新策略上有竞争力。
- `label1d` 换手更高，整体回撤也更大。
- `label5d_final` 对当前 `rolling_p10_h5` 策略仍然更好，尤其是在回撤控制上。
- valid/test 表现还不够稳定：`label1d` 在 valid 的 `rolling_p10_h5` 中明显更弱，但在 test 的高换手策略中很强。应把它视为新候选，而不是立即替换当前方案。

## 2026-06-02 每日交易计划对比

输出根目录：

- `outputs/live/label1d_vs_label5d_20260602_from_20260601`

来源特征：

- `outputs/live/rolling_p10_h5_20260602_from_20260601_20260601_220111/live_features_20260601.parquet`

每日候选股票数量：`4142`。

Top10 重合：

- 数量：`3`
- 重合股票：`300030.SZ`、`603628.SH`、`688021.SH`

Label5d 最终模型 Top2：

| 排名 | ts_code | 行业 | close | pred |
| ---: | --- | --- | ---: | ---: |
| 1 | 000576.SZ | 电气设备 | 7.65 | 0.3724 |
| 2 | 300030.SZ | 医疗保健 | 7.09 | 0.3568 |

Label1d LightGBM Top2：

| 排名 | ts_code | 行业 | close | pred |
| ---: | --- | --- | ---: | ---: |
| 1 | 000010.SZ | 建筑工程 | 2.21 | 0.2073 |
| 2 | 603628.SH | 电气设备 | 11.08 | 0.1910 |

如果从 label5d Top2 切换到 label1d Top2，严格 rolling tranche 口径下的换手：

- Top2 重合数：`0`
- 组合换手：`20%`
- 卖出：`000576.SZ`、`300030.SZ`
- 买入：`000010.SZ`、`603628.SH`

## 建议

暂时不要用 `label1d` 替代当前实盘 `rolling_p10_h5` 策略。

当前实际建议：

- 保留 `label5d_final + rolling_p10_h5` 作为保守实盘策略。
- 跟踪 `label1d_lgb + topk20_drop3` 和 `label1d_lgb + rank_buffer`，作为实验性的高换手替代方案。
- 如果使用 `label1d`，实盘前需要加入更强的换手和风险约束，因为它表现最好的策略依赖频繁换仓。

针对 2026-06-02：

- 保守方案：使用已有 label5d Top2：`000576.SZ`、`300030.SZ`。
- 实验性 label1d 方案：`000010.SZ`、`603628.SH`。
- 严格 Top2 重合为零，因此切换信号会立即引入换仓，而且验证稳定性还不够。

## 实现备注

`src.model_experiments.*` 脚本暴露了 `run_cli()`，但当时缺少模块级 `if __name__ == "__main__": run_cli()`，所以不能直接通过 `python -m` 执行。这个实验中调用的是底层函数。

最初运行 label1d 基础模型时，`raw_return_col == daily_return_col == label_1d` 导致写 parquet 预测文件时出现重复列。模型当时已经保存，因此后续通过重新加载已保存模型、写出去重后的预测表，修复了预测文件和 summary。
