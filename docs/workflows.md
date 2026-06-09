# 工作流

## 训练或评估模型

优先使用命名实验默认配置：

```bash
python -m src.experiments gbdt --experiment gbdt_full
python -m src.experiments fusion --experiment fusion_methods
python -m src.experiments final-handoff --experiment final_model_handoff
```

每次运行都会写出 `run_meta.json`，记录命令参数、输入产物、注册表快照和 git 状态。

## 运行策略回测

使用命名策略运行配置：

```bash
python -m src.experiments strategy-backtest --strategy-run default_grid
```

策略运行配置定义在 `configs/registry/strategies.yaml`；模型预测文件路径来自 `configs/registry/models.yaml`。

真实交易约束第一版使用：

```bash
python -m src.experiments strategy-backtest --strategy-run label1d_vs_label5d_realistic
```

这一版默认：

```text
模型：label5d_final、label1d_lgb
策略：rolling_p10_h5、topk20_drop3、rankbuf_p20_b50_s100_min2_max10
成本：5 bps 基础交易成本 + 20 bps 滑点
买入约束：data/processed/universe.parquet 中的 in_universe、非 ST、passes_liquidity
```

成本敏感性矩阵使用：

```bash
python -m src.experiments strategy-sensitivity --strategy-run label1d_vs_label5d_cost_sensitivity
```

默认总成本档位：

```text
5 bps
10 bps
20 bps
50 bps
```

输出包括：

```text
sensitivity_metrics.csv
best_by_valid_cost.csv
plots/valid_sharpe.svg
plots/test_sharpe.svg
plots/test_total_return.svg
plots/test_max_drawdown.svg
sensitivity_report.md
```

市场压力降仓和组合回撤控制使用：

```bash
python -m src.experiments strategy-backtest --strategy-run label1d_vs_label5d_risk_controls
```

默认风控参数：

```text
市场压力：过去 5 个已实现市场近似收益累计小于 -3%，仓位上限降到 50%
组合回撤：回撤低于 -8% 仓位上限 50%，低于 -12% 仓位上限 25%，低于 -18% 仓位上限 20%
市场压力滞后：2 个交易日，避免用到尚未实现完的 label_1d 收益
```

风险收益权衡参数扫描使用：

```bash
python -m src.experiments strategy-risk-sweep --strategy-run label1d_vs_label5d_risk_return_sweep
```

默认会比较：

```text
无风控
市场压力温和/中等降仓
组合回撤温和/中等降仓
市场压力 + 组合回撤温和/中等/强降仓
```

输出包括：

```text
risk_sweep_metrics.csv
risk_sweep_selected.csv
risk_sweep_pareto_valid.csv
plots/valid_risk_return_scatter.svg
plots/test_risk_return_scatter.svg
risk_sweep_report.md
```

策略和执行假设的下一阶段计划见：

```text
docs/dev/15_next_stage_strategy_execution_plan.md
```

其中记录了 VWAP 的定义、成交假设重构、交易成本敏感性、涨跌停约束、换手约束、组合层策略和模型后续计划。

## 刷新策略报告

如果只是修改图表或报告逻辑，不需要重新跑完整回测，可以直接从已有策略输出目录刷新长表和 HTML 报告：

```bash
python -m src.experiments strategy-report outputs/strategy/20260605_144431__label1d_vs_label5d
```

刷新后会写出：

```text
metrics_long.csv
equity_long.parquet
report.html
```

## 生成每日排序

使用选定的每日交易产物包：

```bash
python -m src.experiments live-rank \
  --decision-date 20260603 \
  --trade-date 20260604 \
  --artifact-registry configs/registry/artifacts.yaml \
  --out-dir outputs/live/20260604__final__from_20260603
```

每日交易使用的模型文件来自 `configs/registry/artifacts.yaml`。

## 规范化本地输出目录

检查本地输出目录是否仍有不符合命名规则的旧目录：

```bash
python -m src.experiments normalize-outputs --dry-run
```
