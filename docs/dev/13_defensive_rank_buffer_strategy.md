# 13 防守型 rank buffer 策略记录

本文记录本轮新增的防守型策略，用来处理 2024 年初小盘、低流动性股票集中下跌带来的组合回撤问题。

## 1. 问题背景

前面检查 2024 年初回撤时看到，原来的高收益策略不是单纯输给大盘，而是暴露在更强的小盘和低流动性风险上：

```text
rolling_p10_h5:   2024-01/02 窗口最大回撤 -39.88%，最差日在 2024-02-05
topk20_drop3:     2024-01/02 窗口最大回撤 -42.65%，最差日在 2024-02-05
等权股票池基线:    2024-01/02 窗口最大回撤 -28.02%，最差日在 2024-02-01
沪深300权重基线:   2024-01/02 窗口最大回撤  -5.27%，最差日在 2024-01-31
```

这说明早 2024 的大回撤主要来自候选池中的小微盘、弱流动性股票暴跌，而不是沪深 300 这种大盘指数同步发生 40% 级别回撤。只靠 `pred` 做截面排序会把模型最强的短期 alpha 和小盘风险混在一起；收益高，但极端市场下尾部风险不可控。

因此新增策略目标不是追求最高收益，而是增加一个防守档位：在仍使用我们模型预测 `pred` 的前提下，过滤掉最脆弱的标的，并在市场出现连续下跌时主动降仓。

## 2. 策略逻辑

新策略文件：

```text
src/strategy/strategies/defensive_rank_buffer.py
```

策略名：

```text
defensive_rank_buffer
```

它基于原 `rank_buffer`，但增加三类风险控制。

第一，市值与流动性过滤：

```text
log_total_mv__cs_rank >= min_size_rank
log_amount__cs_rank   >= min_amount_rank
```

这里的 rank 特征来自：

```text
data/processed/features.parquet
```

回测入口默认把这些特征合并到预测结果中：

```text
log_total_mv__cs_rank
log_amount__cs_rank
volatility_20__cs_rank
turnover_rate__cs_rank
```

第二，波动过滤：

```text
volatility_20__cs_rank <= max_volatility_rank
```

这会剔除最近波动过高的股票，避免在市场压力期继续买入已经处于高波动状态的标的。

第三，市场压力降仓：

```text
过去 market_window 日全股票池等权累计收益 <= market_stress_threshold
```

当前配置用过去 5 日、阈值 -8%。触发后组合总暴露降到 `stress_gross_exposure`，剩余权重视作现金。这个信号只使用当前交易日前的历史收益，不使用未来收益。

持仓仍然沿用 rank buffer 的基本思路：

```text
持有股票如果仍满足风险过滤，且排名没有跌出 sell_rank，则继续持有；
达到 min_hold_days 后，如果风险过滤失败或 rank 跌出 sell_rank，则卖出；
新买股票必须同时通过风险过滤，并在 buy_rank 内按 pred 排序补齐；
单票权重不超过 max_position_weight；
压力期只降低总仓位，不放宽单票权重上限。
```

## 3. 代码改动

新增或修改的主要位置：

```text
src/strategy/strategies/defensive_rank_buffer.py
src/strategy/config.py
src/strategy/data.py
src/strategy/grid.py
src/strategy/engine.py
src/strategy/strategies/__init__.py
src/pipelines/run_strategy_backtest.py
```

新增配置项：

```text
min_size_rank
min_amount_rank
max_volatility_rank
market_window
market_stress_threshold
stress_gross_exposure
```

当前策略网格新增三组配置：

```text
defensive_p20_b60_s180_size-35_amt-35
defensive_p20_b80_s220_size-20_amt-20
defensive_p30_b100_s260_size-20_amt-20
```

## 4. 本轮实验

运行命令：

```bash
python -m src.pipelines.run_strategy_backtest \
  --models final \
  --splits valid test \
  --out-root outputs/strategy \
  --run-name defensive_filter \
  --transaction-cost-bps 5 \
  --index-code 000300.SH
```

使用的模型预测：

```text
outputs/models/20260531_162154__final_model_handoff/valid/valid_pred.parquet
outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet
```

输出目录：

```text
outputs/strategy/defensive_filter_20260601_005706/
```

图已经按策略家族拆分，且纵轴使用 log10 净值标度：

```text
outputs/strategy/defensive_filter_20260601_005706/final/valid/equity_overview.svg
outputs/strategy/defensive_filter_20260601_005706/final/valid/plots_by_family/defensive_rank_buffer.svg
outputs/strategy/defensive_filter_20260601_005706/final/test/equity_overview.svg
outputs/strategy/defensive_filter_20260601_005706/final/test/plots_by_family/defensive_rank_buffer.svg
```

## 5. 回测结果

Valid split：

| 策略 | total_return | annual_return | sharpe | max_drawdown | avg_turnover | avg_n_holdings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `defensive_p20_b60_s180_size-35_amt-35` | 0.1589 | 0.1660 | 0.8410 | -0.1572 | 0.3925 | 12.29 |
| `defensive_p20_b80_s220_size-20_amt-20` | 0.3854 | 0.4042 | 1.6795 | -0.1140 | 0.4037 | 13.14 |
| `defensive_p30_b100_s260_size-20_amt-20` | 0.3712 | 0.3892 | 1.7521 | -0.1145 | 0.4019 | 18.02 |

Test split：

| 策略 | total_return | annual_return | sharpe | max_drawdown | avg_turnover | avg_n_holdings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `defensive_p20_b60_s180_size-35_amt-35` | 0.0585 | 0.0453 | 0.3876 | -0.1281 | 0.3489 | 12.84 |
| `defensive_p20_b80_s220_size-20_amt-20` | 0.0596 | 0.0462 | 0.3845 | -0.1074 | 0.3696 | 13.67 |
| `defensive_p30_b100_s260_size-20_amt-20` | 0.0746 | 0.0577 | 0.4711 | -0.1036 | 0.3785 | 19.62 |

2024 年 1-2 月压力窗口对比：

| 策略/基线 | 窗口收益 | 窗口最大回撤 | 最差日期 |
| --- | ---: | ---: | --- |
| `rolling_p10_h5` | -0.1168 | -0.3988 | 2024-02-05 |
| `topk20_drop3` | -0.1815 | -0.4265 | 2024-02-05 |
| `defensive_p20_b80_s220_size-20_amt-20` | 0.0984 | -0.1089 | 2024-02-01 |
| `defensive_p30_b100_s260_size-20_amt-20` | 0.1389 | -0.0638 | 2024-02-27 |
| `benchmark_equal_weight_universe` | -0.0976 | -0.2802 | 2024-02-01 |
| `benchmark_000300_sh_weight` | 0.0555 | -0.0527 | 2024-01-31 |

## 6. 结论

这个防守策略有效降低了 2024 年初的小盘流动性冲击。原高收益策略在该窗口有接近 40% 的回撤，新策略压到约 6%-11%。

代价也很明显：防守策略在 test 上收益很低，Sharpe 也明显低于原来的 `rolling_tranche`、`topk_drop`、`rank_buffer`。原因是它过滤掉了大量模型最偏好的小盘、高弹性标的，而这些标的在 2025-2026 test 区间贡献了很强 alpha。

因此当前不建议把 `defensive_rank_buffer` 当作主策略。更合理的用法是：

```text
常态：使用 rolling_p10_h5 或 rolling_p20_h3 作为进攻主策略；
压力期：当全股票池短期下跌、流动性恶化或组合回撤超过阈值时，切换到 defensive_rank_buffer；
组合层：用 defensive_rank_buffer 作为风险控制 sleeve，而不是替代全部 alpha 暴露。
```

后续可以继续做一个动态组合策略：平时持有 `rolling_p10_h5`，市场压力信号触发时切换到防守策略或降仓，压力解除后再逐步恢复进攻仓位。
