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

外部点位基线文件需要包含 `trade_date`，并至少包含 `equity`、`return`、`close`、`adj_close`、`price`、`nav` 之一。管线会按当前 valid/test 切分的交易日对齐外部基线，避免策略和指数使用不同时间区间计算指标。

## 2. 代码结构统一

策略代码已拆分，`backtest.py` 只保留兼容导出，不再承载所有实现：

```text
src/strategy/backtest.py        # 兼容导出层
src/strategy/config.py          # 策略参数 dataclass
src/strategy/data.py            # 预测数据读取和日度面板准备
src/strategy/engine.py          # 通用回测撮合、收益、换手、持仓输出
src/strategy/benchmarks.py      # 等权、指数权重、外部点位基线
src/strategy/plotting.py        # SVG 回测图
src/strategy/grid.py            # 策略参数网格
src/strategy/io.py              # 输出文件写入
```

每个策略单独一个文件：

```text
src/strategy/strategies/rolling_tranche.py
src/strategy/strategies/topk_drop.py
src/strategy/strategies/rank_buffer.py
src/strategy/strategies/risk_balanced_tail.py
src/strategy/strategies/risk_filtered_rank_buffer.py
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

每个模型和切分汇总输出：

```text
strategy_metrics.csv
equity_comparison.svg
```

整个实验根目录输出：

```text
summary.json
strategy_report.md
```

回测输出目录默认带时间戳，避免覆盖旧实验：

```text
outputs/strategy/YYYYMMDD_HHMMSS__strategy_backtest/
```

可以用 `--run-name` 改变目录名中的实验名；只有显式传 `--no-timestamp` 时才会写入 `outputs/strategy/<run_name>/`。

本轮重跑结果目录：

```text
outputs/strategy/unified_final_20260601_001207/strategy_report.md
outputs/strategy/unified_final_20260601_001207/final/valid/strategy_metrics.csv
outputs/strategy/unified_final_20260601_001207/final/test/strategy_metrics.csv
outputs/strategy/unified_final_20260601_001207/final/valid/equity_comparison.svg
outputs/strategy/unified_final_20260601_001207/final/test/equity_comparison.svg
```

## 4. 图形逻辑

`equity_comparison.svg` 默认使用明确的 log10 净值标度。SVG 里会写入：

```text
y-axis: log10 equity
Equity (log10 scale)
```

纵轴坐标按 `log10(equity)` 映射；等距的纵向间隔表示相同的收益倍数，而不是相同的净值差值。若需要线性净值图，可加：

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
  --out-root outputs/strategy \
  --run-name unified_final \
  --transaction-cost-bps 5 \
  --index-code 000300.SH
```

使用的是我们的最终模型预测文件：

```text
outputs/models/20260531_162154__final_model_handoff/valid/valid_pred.parquet
outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet
```

选股信号只使用 `pred`。`label_1d` 只用于事后收益计算和只看历史窗口的风险估计，不作为当日选股信号。

## 6. 基线与主要结果

沪深 300 权重基线：

| 数据集 | 总收益 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: |
| valid | 0.1044 | 0.5859 | -0.1497 |
| test | 0.2731 | 1.3193 | -0.1053 |

等权股票池基线：

| 数据集 | 总收益 | Sharpe | 最大回撤 |
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

## 7. 拆分图与风险预算策略更新

本轮进一步改进画图逻辑，不再只输出一张全策略大图。每个切分现在输出：

```text
equity_overview.svg
equity_top_valid_sharpe.svg
equity_all_debug.svg
plots_by_family/rolling_tranche.svg
plots_by_family/topk_drop.svg
plots_by_family/rank_buffer.svg
plots_by_family/risk_balanced_tail.svg
plots_by_family/risk_filtered_rank_buffer.svg
plots_by_family/risk_budget_rank_buffer.svg
```

报告主文档只引用 overview、top valid Sharpe 和 all debug。全量图保留为调试图，按 family 拆分的图用于看同一类策略的参数差异。

本轮新增策略：

```text
risk_budget_rank_buffer
```

逻辑：

1. 仍以模型 `pred` 为主要 alpha 信号。
2. 只用当前日期以前的 `label_1d` 计算历史波动率。
3. 候选排序使用 `alpha_rank - volatility_penalty * volatility_rank`。
4. 保留 rank buffer 卖出逻辑，限制每日最大更新数量。
5. 持仓权重使用 inverse-vol，并设置单票权重上限。

本轮完整回测命令：

```bash
python -m src.pipelines.run_strategy_backtest \
  --models final \
  --splits valid test \
  --out-root outputs/strategy \
  --run-name risk_budget_plot_refactor \
  --transaction-cost-bps 5 \
  --index-code 000300.SH
```

输出目录：

```text
outputs/strategy/risk_budget_plot_refactor_20260601_003245/
```

风险预算策略结果：

| 数据集 | 策略 | 总收益 | Sharpe | 最大回撤 | 平均换手 |
| --- | --- | ---: | ---: | ---: | ---: |
| valid | `riskbudget_p20_top150_keep80_b40_s120_pen25` | 0.8769 | 1.8754 | -0.4302 | 0.5959 |
| valid | `riskbudget_p20_top200_keep100_b50_s150_pen35` | 0.6462 | 1.6134 | -0.3938 | 0.4576 |
| valid | `riskbudget_p30_top200_keep120_b60_s180_pen35` | 0.5662 | 1.4590 | -0.3969 | 0.4346 |
| test | `riskbudget_p20_top150_keep80_b40_s120_pen25` | 1.6292 | 4.0447 | -0.0977 | 0.5377 |
| test | `riskbudget_p20_top200_keep100_b50_s150_pen35` | 1.0565 | 3.2692 | -0.0855 | 0.4144 |
| test | `riskbudget_p30_top200_keep120_b60_s180_pen35` | 1.0281 | 3.2104 | -0.0819 | 0.4064 |

结论：

风险预算策略确实能降低部分配置的换手和 test 回撤，但收益和 Sharpe 明显低于当前最强的纯 alpha/缓冲策略。它适合作为“风险更保守”的候选，不适合作为当前主推策略。下一步如果继续优化，应把目标从单纯 inverse-vol 改为 alpha、波动、回撤和相对 000300 暴露的多目标打分，而不是只惩罚低波动。
