# 11 策略回测实验记录

本文档记录策略代码修正、三类策略实现和基于最终模型预测分数的回测结果。

## 1. 本轮修正

原 `src/strategy/backtest_hs300_risk_balanced_tail_seeds.py` 存在严重未来函数：它直接用目标日真实 `daily_return` 排序选股，相当于提前知道未来收益。

本轮已将它改为兼容入口，实际调用新的模型预测策略回测：

```text
src/pipelines/run_strategy_backtest.py
```

核心策略实现位于：

```text
src/strategy/backtest.py
```

新实现只使用预测文件中的：

```text
trade_date, ts_code, pred
```

`label_1d` 只用于事后计算每日组合收益，不参与选股。

## 2. 已实现策略

### 2.1 Rolling Tranche

每天按 `pred` 排序买入一批股票，每批持有固定天数，到期卖出。

本轮参数：

```text
rolling_p10_h5: target_positions=10, hold_days=5, daily_buy=2
rolling_p20_h3: target_positions=20, hold_days=3, daily_buy=7
rolling_p20_h5: target_positions=20, hold_days=5, daily_buy=4
rolling_p20_h10: target_positions=20, hold_days=10, daily_buy=2
rolling_p30_h5: target_positions=30, hold_days=5, daily_buy=6
```

实现细节：

- 若持仓不足目标数量，会从当日高分候选中补足。
- 权重使用等权。
- 交易成本按权重变化的一边换手近似扣除。

### 2.2 TopK-Drop

每天保持 TopK 组合，卖出持仓中当前分数排名最低的 Drop 只，并买入未持仓股票中分数最高的 Drop 只。

本轮参数：

```text
topk20_drop1
topk20_drop2
topk20_drop3
topk20_drop5
topk30_drop3
```

### 2.3 Rank Buffer

缓冲区策略，避免分数小幅波动导致频繁换仓。

本轮参数：

```text
rankbuf_p20_b30_s100_min2_max10
rankbuf_p20_b50_s100_min2_max10
rankbuf_p30_b50_s150_min2_max10
```

含义示例：

```text
p20: 目标持仓 20
b30: 只从 top30 买入
s100: 跌出 top100 才卖
min2: 至少持有 2 天
max10: 最多持有 10 天后复核
```

### 2.4 修正后的 Risk-Balanced Tail

保留原脚本想表达的 core/tail 思路，但修正为：

```text
core = 当日 pred top30
tail = 在高分候选中按历史低相关、低波动、低下行波动筛选
权重 = core 90%, tail 10%
```

该策略不再使用未来真实收益选股。

## 3. 回测设置

模型：

```text
final:     outputs/models/20260531_162154__final_model_handoff/{valid,test}/..._pred.parquet
lgb_top40: outputs/models/20260530_205006__feature_selection/lightgbm_top40/lightgbm/{valid,test}/..._pred.parquet
```

收益列：

```text
label_1d
```

交易成本：

```text
5 bps
```

输出目录：

```text
outputs/strategy/model_pred_strategies/
```

主要输出：

```text
outputs/strategy/model_pred_strategies/final/valid/strategy_metrics.csv
outputs/strategy/model_pred_strategies/final/test/strategy_metrics.csv
outputs/strategy/model_pred_strategies/lgb_top40/valid/strategy_metrics.csv
outputs/strategy/model_pred_strategies/lgb_top40/test/strategy_metrics.csv
```

回测曲线：

```text
outputs/strategy/model_pred_strategies/final/valid/equity_comparison.svg
outputs/strategy/model_pred_strategies/final/test/equity_comparison.svg
outputs/strategy/model_pred_strategies/lgb_top40/valid/equity_comparison.svg
outputs/strategy/model_pred_strategies/lgb_top40/test/equity_comparison.svg
```

每个策略子目录还保存：

```text
equity_curve.csv
trades.csv
holdings.csv
metrics.json
```

## 4. Final 模型结果

### 4.1 Valid 上按 Sharpe 排序

| 策略 | 持仓数 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 平均换手 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rolling_p10_h5` | 10 | 1.7886 | 1.9093 | 2.6902 | -0.3988 | 0.5521 |
| `topk20_drop3` | 20 | 1.4913 | 1.5871 | 2.4223 | -0.4265 | 0.5802 |
| `topk20_drop2` | 20 | 1.4295 | 1.5203 | 2.3911 | -0.4190 | 0.4318 |
| `topk20_drop1` | 20 | 1.3028 | 1.3836 | 2.3341 | -0.3597 | 0.2698 |
| `rankbuf_p20_b50_s100_min2_max10` | 20 | 1.4074 | 1.4964 | 2.2899 | -0.4270 | 0.6707 |

### 4.2 Valid 选择策略在 Test 上表现

| 策略 | 持仓数 | Test 总收益 | Test 年化收益 | Test Sharpe | Test 最大回撤 | 平均换手 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rolling_p10_h5` | 10 | 4.4753 | 2.7679 | 5.7727 | -0.0920 | 0.5474 |
| `topk20_drop3` | 20 | 3.5697 | 2.2722 | 5.5009 | -0.0927 | 0.5498 |
| `topk20_drop2` | 20 | 3.2365 | 2.0845 | 5.4734 | -0.0791 | 0.4093 |
| `topk20_drop1` | 20 | 2.1724 | 1.4614 | 4.5436 | -0.1081 | 0.2492 |
| `rankbuf_p20_b50_s100_min2_max10` | 20 | 3.8449 | 2.4249 | 5.5123 | -0.1134 | 0.6353 |

## 5. LightGBM Top40 Baseline 结果

### 5.1 Valid 上按 Sharpe 排序

| 策略 | 持仓数 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 平均换手 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rolling_p10_h5` | 10 | 2.4697 | 2.6528 | 3.2612 | -0.3900 | 0.5678 |
| `topk20_drop3` | 20 | 1.8089 | 1.9314 | 2.7519 | -0.4084 | 0.5959 |
| `topk20_drop1` | 20 | 1.5567 | 1.6579 | 2.6027 | -0.3601 | 0.2665 |
| `topk20_drop2` | 20 | 1.6003 | 1.7050 | 2.5757 | -0.4096 | 0.4492 |
| `topk20_drop5` | 20 | 1.6024 | 1.7073 | 2.4982 | -0.4227 | 0.8298 |

### 5.2 Valid 选择策略在 Test 上表现

| 策略 | 持仓数 | Test 总收益 | Test 年化收益 | Test Sharpe | Test 最大回撤 | 平均换手 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rolling_p10_h5` | 10 | 4.9489 | 3.0198 | 5.8174 | -0.0950 | 0.5579 |
| `topk20_drop3` | 20 | 3.8787 | 2.4435 | 5.7225 | -0.0874 | 0.5656 |
| `topk20_drop1` | 20 | 2.9560 | 1.9240 | 5.3644 | -0.1226 | 0.2542 |
| `topk20_drop2` | 20 | 3.6486 | 2.3161 | 5.8082 | -0.0781 | 0.4180 |
| `topk20_drop5` | 20 | 3.9592 | 2.4878 | 5.6710 | -0.0957 | 0.8077 |

## 6. 结果解读

1. 严格按 valid Sharpe，两个模型的最佳策略都是：

```text
rolling_p10_h5
```

它收益和 Sharpe 最高，但只持有 10 只股票，组合更集中。

2. 如果希望组合更分散、贴合作业建议的 20 只持仓，final 模型上更推荐：

```text
topk20_drop3
```

它是 final 模型 valid 上 Sharpe 最高的 20 持仓策略，test 表现也稳定：

```text
valid Sharpe = 2.4223
test Sharpe  = 5.5009
test maxDD   = -0.0927
```

## 7. Seeded Core/Tail 风险均衡策略复测

针对 `src/strategy/backtest_hs300_risk_balanced_tail_seeds.py`，本轮保留原设计：

```text
总持仓 100 股
core Top30 占 90% 权重，每只约 3%
tail 70 股占 10% 权重
初始持仓为 seed=0..9 的随机 100 股等权组合
每天最多换 25 只股票
```

修补点：

- 原逻辑中“再按下一日收益率选出 70 股”存在未来函数，已改为按最终模型 `pred` 选出 tail 70。
- 真实 `label_1d` 只用于过去 60 日风险指数和事后收益计算。
- 风险指数仍由三项组成：与 core Top30 组合历史收益相关性、个股波动率、下行波动率。
- 为避免重复计算，目标组合按交易日预先缓存；风险计算改为 NumPy 向量化。
- 初始随机 100 股只从首个回测交易日实际可见股票池抽样，不再从整个测试期股票全集抽样。

运行命令：

```bash
python src/strategy/backtest_hs300_risk_balanced_tail_seeds.py \
  --input outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet \
  --output-dir outputs/strategy/hs300_risk_balanced_tail_seeded_final_test \
  --stages 10,100,300 \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

汇总结果：

| 阶段天数 | seeds | 平均总收益 | 平均 Sharpe | 平均最大回撤 | 平均年化波动 | 平均换手 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 10 | 0.0939 | 22.5577 | -0.0308 | 0.3822 | 0.9824 |
| 100 | 10 | 1.0328 | 20.1024 | -0.0576 | 0.2475 | 0.9569 |
| 300 | 10 | 3.6918 | 11.6610 | -0.0838 | 0.2284 | 0.9369 |

输出文件：

```text
outputs/strategy/hs300_risk_balanced_tail_seeded_final_test/summary_aggregate.csv
outputs/strategy/hs300_risk_balanced_tail_seeded_final_test/summary_by_seed.csv
outputs/strategy/hs300_risk_balanced_tail_seeded_final_test/daily_performance_by_seed.csv
outputs/strategy/hs300_risk_balanced_tail_seeded_final_test/equity_stage_10d.svg
outputs/strategy/hs300_risk_balanced_tail_seeded_final_test/equity_stage_100d.svg
outputs/strategy/hs300_risk_balanced_tail_seeded_final_test/equity_stage_300d.svg
```

结论：这个版本显式考虑了风险、波动和均衡，回撤明显低于更集中的 10/20 股排序策略，但换手仍偏高。后续如果继续优化，应优先加入更强的换手约束、行业/风格暴露约束，以及对 tail 风险池规模和风险权重的 valid 集调参。

3. 如果更强调换手和回撤，final 模型可以选择：

```text
topk20_drop2
```

相比 `topk20_drop3`，它 valid Sharpe 略低，但换手更低，test 最大回撤更小：

```text
avg_turnover: 0.4093
test maxDD:   -0.0791
```

4. Rank Buffer 在 test 上表现不错，但 valid 上没有超过 TopK-Drop。本轮不建议作为主策略，只作为稳健性对照。

5. 修正后的 Risk-Balanced Tail 不再有未来函数，但表现不如简单 TopK-Drop，且持仓 100 只、平均换手较高，不适合作为本阶段主策略。

6. LightGBM top40 在策略回测上仍然略强于 final 模型。这与此前模型交接文档一致：final 模型的优势主要在 ICIR 和风险解释，简单收益策略上 LightGBM 仍是强 baseline。

## 7. 当前建议

报告主策略建议使用：

```text
Final model + topk20_drop3
```

理由：

- 使用最终深度融合模型，符合课程中必须包含深度学习模型的要求。
- 20 只持仓，比 10 只 rolling 更分散。
- valid 上是 final 模型 20 持仓策略中的最高 Sharpe。
- test 上收益、Sharpe 和回撤都稳定。
- 策略逻辑简单，接近作业要求的 “持有 n 只、每天卖出/买入 k 只”。

如果实际模拟交易更重视降低换手，可以改为：

```text
Final model + topk20_drop2
```

最终报告中建议同时列出：

```text
Final + topk20_drop3
Final + topk20_drop2
LightGBM top40 + topk20_drop3
```

这样可以说明最终模型策略表现、低换手对照，以及强树模型 baseline。

## 8. 局限

当前回测仍是简化版：

- 使用 `label_1d` 做日收益近似。
- 未显式处理涨跌停、停牌无法成交、最小交易单位和滑点。
- 交易成本只按权重变化用固定 bps 扣除。
- 没有行业和市值中性化约束。

这些限制需要在报告中说明。下一步如果继续完善策略，应优先加入涨跌停/停牌约束和交易清单生成逻辑。

## 9. Risk-Filtered Rank Buffer 实验

本轮新增低换手风险过滤策略：

```text
risk_filtered_rank_buffer
```

流程：

```text
1. 当日按最终模型 pred 排序，取 Top100/Top150 作为收益候选池。
2. 用当前日期以前的 60 日 label_1d 历史收益计算风险指数。
3. 风险指数由与核心组合相关性、个股波动率、下行波动率组成。
4. 只在低风险候选池中补仓。
5. 持仓跌出 sell_rank 或风险恶化，并满足最小持有天数后才卖。
6. 每日主动卖出数量受 max_stock_updates 限制，缺失股票补仓补足目标持仓。
```

实现位置：

```text
src/strategy/backtest.py
```

运行命令：

```bash
python -m src.pipelines.run_strategy_backtest \
  --models final \
  --splits valid test \
  --out-root outputs/strategy/risk_filtered_rank_buffer_final \
  --transaction-cost-bps 5
```

### 9.1 Valid 结果

| 策略 | 持仓数 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 平均换手 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `riskbuf_p20_top150_keep80_b40_s100_min3_max10` | 20 | 1.2486 | 1.3252 | 2.3473 | -0.3943 | 0.5545 |
| `riskbuf_p20_top100_keep70_b50_s120_min3_max10` | 20 | 1.2347 | 1.3102 | 2.2616 | -0.4112 | 0.5657 |
| `riskbuf_p30_top150_keep80_b60_s150_min3_max10` | 30 | 0.9143 | 0.9664 | 1.9713 | -0.3876 | 0.5033 |
| `riskbuf_p30_top100_keep70_b50_s120_min5_max15` | 30 | 0.9367 | 0.9903 | 1.9342 | -0.3971 | 0.4394 |

Valid 上最好的 riskbuf 是：

```text
riskbuf_p20_top150_keep80_b40_s100_min3_max10
```

但它没有超过 `topk20_drop3`：

```text
riskbuf valid Sharpe = 2.3473
topk20_drop3 valid Sharpe = 2.4223
```

### 9.2 Test 结果

| 策略 | 持仓数 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 平均换手 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `riskbuf_p20_top150_keep80_b40_s100_min3_max10` | 20 | 2.2588 | 1.5135 | 4.9009 | -0.0753 | 0.5217 |
| `riskbuf_p20_top100_keep70_b50_s120_min3_max10` | 20 | 2.6059 | 1.7200 | 4.8658 | -0.0781 | 0.5331 |
| `riskbuf_p30_top100_keep70_b50_s120_min5_max15` | 30 | 2.1049 | 1.4204 | 4.5625 | -0.0772 | 0.4169 |
| `riskbuf_p30_top150_keep80_b60_s150_min3_max10` | 30 | 1.7983 | 1.2318 | 4.4274 | -0.0775 | 0.4706 |

解读：

- Riskbuf 明显降低了 test 最大回撤，最好约 `-0.0753`，优于 `topk20_drop3` 的 `-0.0927`。
- 收益和 Sharpe 低于 TopK/rolling，说明风险过滤牺牲了部分 alpha 暴露。
- 20 股版本优于 30 股版本，说明当前模型收益信号集中在更靠前的股票，过度分散会摊薄收益。

本轮结论：riskbuf 适合作为风险控制对照策略，不建议替代当前主策略。最终主策略仍建议保持 `Final + topk20_drop3`，低回撤备选可以比较 `topk20_drop2` 与 `riskbuf_p20_top150_keep80_b40_s100_min3_max10`。
