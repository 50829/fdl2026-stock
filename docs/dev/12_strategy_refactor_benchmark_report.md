# 12 策略代码与回测报告统一记录

本文记录本轮对策略代码、输出文档和回测图的统一调整。

## 1. 指数数据检查

当前仓库内已有指数成分权重数据：

```text
data/raw/index_weight.zip
```

它不是指数点位行情，而是指数成分权重文件。压缩包内每个 CSV 的字段为：

```text
index_code, con_code, trade_date, weight
```

当前包含的指数：

```text
000300.SH: 20160129 - 20260401
399006.SZ: 20160104 - 20260331
```

已检查的数据文件：

```text
data/raw/index_weight.zip
data/processed/features.parquet
data/processed/labels.parquet
data/processed/universe.parquet
data/processed/splits.json
data/processed_pilot/splits.json
```

因此本轮新增两类默认基线：

```text
benchmark_000300_sh_weight
benchmark_equal_weight_universe
```

`benchmark_000300_sh_weight` 使用 `000300.SH` 月度成分权重，并用预测文件中已有个股 `label_1d` 计算成分组合日收益。每个交易日使用不晚于当天的最近一期指数权重；若某些成分股当天不在预测股票池中，则在可匹配成分内重新归一化权重。

`benchmark_equal_weight_universe` 使用预测文件中的全股票池等权收益：

```text
benchmark_equal_weight_universe
```

它不是沪深 300 或中证 500 等真实指数，只用于回答“模型策略是否跑赢同一候选股票池平均表现”。

如果后续补充真实指数点位行情，可继续通过管线参数加入：

```bash
python -m src.pipelines.run_strategy_backtest \
  --benchmark-path data/raw/index/000300.csv \
  --benchmark-name hs300
```

外部点位基线文件需要包含 `trade_date`，并至少包含 `equity`、`return`、`close`、`adj_close`、`price`、`nav` 之一。管线会按当前 valid/test split 的交易日对齐外部基线，避免策略和指数使用不同时间区间计算指标。

## 2. 代码结构统一

策略核心逻辑集中在：

```text
src/strategy/backtest.py
```

统一导出入口：

```text
src/strategy/__init__.py
```

批量实验入口：

```text
src/pipelines/run_strategy_backtest.py
```

保留的独立脚本：

```text
src/strategy/backtest_hs300_risk_balanced_tail_seeds.py
```

它用于复现实验要求中“初始随机 100 股、seed=0-9、Top30+Tail70”的专门策略。通用策略网格和统一报告以后优先走 `src/pipelines/run_strategy_backtest.py`。

## 3. 统一输出结构

每个策略和每个基线都输出同一组文件：

```text
equity_curve.csv
trades.csv
holdings.csv
metrics.json
```

每个模型和 split 汇总输出：

```text
strategy_metrics.csv
equity_comparison.svg
```

整个实验根目录输出：

```text
summary.json
strategy_report.md
```

本轮统一回测输出目录：

```text
outputs/strategy/unified_final/
```

关键结果文件：

```text
outputs/strategy/unified_final/strategy_report.md
outputs/strategy/unified_final/final/valid/strategy_metrics.csv
outputs/strategy/unified_final/final/test/strategy_metrics.csv
outputs/strategy/unified_final/final/valid/equity_comparison.svg
outputs/strategy/unified_final/final/test/equity_comparison.svg
```

## 4. 图形逻辑

`equity_comparison.svg` 默认使用对数净值标度：

```text
y-scale: log equity
```

这样不同收益倍率的策略可以在同一张图上更容易比较复利增长斜率和回撤形态。若需要线性净值图，可加：

```bash
--linear-scale
```

比较图现在同时包含策略曲线和基线曲线。当前默认基线为：

```text
benchmark_000300_sh_weight
benchmark_equal_weight_universe
```

## 5. 本轮实验命令

```bash
python -m src.pipelines.run_strategy_backtest \
  --models final \
  --splits valid test \
  --out-root outputs/strategy/unified_final \
  --transaction-cost-bps 5 \
  --index-code 000300.SH
```

使用的是我们的最终模型预测文件：

```text
outputs/models/sdd_final_model_handoff/valid/valid_pred.parquet
outputs/models/sdd_final_model_handoff/test/test_pred.parquet
```

选股信号只使用 `pred`。`label_1d` 只用于事后收益计算和只看历史窗口的风险估计，不作为当日选股信号。

## 6. 基线与主要结果

沪深 300 权重基线：

| split | total_return | sharpe | max_drawdown |
| --- | ---: | ---: | ---: |
| valid | 0.1044 | 0.5859 | -0.1497 |
| test | 0.2731 | 1.3193 | -0.1053 |

等权股票池基线：

| split | total_return | sharpe | max_drawdown |
| --- | ---: | ---: | ---: |
| valid | -0.0487 | 0.0290 | -0.2802 |
| test | 0.5604 | 1.6879 | -0.1706 |

Final 模型策略在 test 上 Sharpe 前五：

| 策略 | total_return | sharpe | max_drawdown | avg_turnover |
| --- | ---: | ---: | ---: | ---: |
| `rolling_p20_h3` | 4.2487 | 5.8629 | -0.0968 | 0.6737 |
| `rolling_p10_h5` | 4.4753 | 5.7727 | -0.0920 | 0.5474 |
| `topk20_drop5` | 3.9108 | 5.7124 | -0.0934 | 0.7901 |
| `rolling_p20_h5` | 3.5849 | 5.6279 | -0.0828 | 0.5111 |
| `rankbuf_p20_b50_s100_min2_max10` | 3.8449 | 5.5123 | -0.1134 | 0.6353 |

需要注意，策略选择仍应先看 valid，再评估 test。valid 上 Sharpe 最高的仍是：

```text
rolling_p10_h5
```

如果要求持仓更分散，20 股版本中可继续重点看 `topk20_drop3`、`risk_filtered_rank_buffer` 和 `rank_buffer` 的折中表现。
