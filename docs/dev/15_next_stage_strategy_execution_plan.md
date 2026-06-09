# 15 下一阶段策略与执行重构计划

日期：2026-06-05

## 1. 当前判断

下一阶段优先方向不是继续训练更复杂的模型，而是把策略和交易执行协议做实。

原因：

- 当前模型已经有可用 alpha。`label5d_final` 的 IC/ICIR 更强，`label1d_lgb` 在高刷新策略上表现更强。
- `label1d` 的强结果主要来自 `topk_drop`、`rank_buffer` 等高换手策略，最容易被滑点、涨跌停和无法成交影响。
- 现在回测收益使用 `label_1d` 作为事后收益近似，尚未充分模拟真实成交价、盘中成交时段、涨跌停和交易失败。
- 继续堆复杂模型之前，需要先知道当前 alpha 在真实交易约束下还能剩多少。

因此下一阶段主线是：

```text
真实成交假设 -> 成本敏感性 -> 交易约束 -> 稳健策略选择 -> 再决定是否重训模型
```

## 2. VWAP 是什么

`VWAP` 是 Volume Weighted Average Price，中文通常叫“成交量加权平均价”。

定义：

```text
VWAP = sum(price_i * volume_i) / sum(volume_i)
```

如果用分钟数据计算，可以理解为：

```text
某个时间段内，每一分钟成交价按成交量加权后的平均成交价
```

例子：

| 时间 | 价格 | 成交量 | 价格乘成交量 |
| --- | ---: | ---: | ---: |
| 14:30 | 10.00 | 10000 | 100000 |
| 14:31 | 10.10 | 20000 | 202000 |
| 14:32 | 9.90 | 10000 | 99000 |

则：

```text
VWAP = (100000 + 202000 + 99000) / (10000 + 20000 + 10000)
     = 10.025
```

它和普通平均价不同。普通平均价只平均价格：

```text
(10.00 + 10.10 + 9.90) / 3 = 10.00
```

VWAP 会给成交量更大的价格更高权重，因此更接近“大资金在该时间段内实际可能成交到的平均价格”。

## 3. 为什么要引入 VWAP 成交假设

当前标签构造代码中的 `label_1d` 明确使用：

```text
label_1d[T] = close[T+2] / close[T+1] - 1
```

含义是：

```text
T 日收盘后生成信号
T+1 日收盘价买入
T+2 日收盘价卖出
```

这在回测上是时点安全的，但真实交易时很难保证所有股票都精确成交在收盘价。因此需要比较几种成交口径：

```text
T+1 open -> T+2 close
T+1 close -> T+2 close
T+1 full-day VWAP -> T+2 close
T+1 14:30-14:55 VWAP -> T+2 close
T+1 closing auction approximate price -> T+2 close
```

如果只有日频数据，暂时无法精确计算分钟级 `14:30-14:55 VWAP`。需要新增分钟数据或可替代的盘中成交数据。

在没有分钟数据前，可以先做两件事：

1. 用 `open` 和 `close` 做上下界压力测试。
2. 用额外滑点惩罚近似 VWAP 偏差，例如 `10bps / 20bps / 50bps`。

## 4. 当前模型状态

### 4.1 `label5d_final`

用途：当前稳健主模型。

结构：

```text
LightGBM top40 + residual-rank MLP
final_pred = pred_lgb + 1.5 * residual_rank_pred
```

特点：

- `label_5d__cs_rank` 目标。
- test IC 和 ICIR 更好。
- 在 rolling tranche 类策略上回撤更低。
- 更适合作为保守实盘主策略。

当前候选策略：

```text
label5d_final + rolling_p10_h5
label5d_final + rolling_p20_h3
label5d_final + topk20_drop3
```

### 4.2 `label1d_lgb`

用途：短周期进攻候选。

结构：

```text
LightGBM top40
target = label_1d__cs_rank
```

特点：

- IC/ICIR 不如 `label5d_final`。
- 在 `topk_drop`、`rank_buffer` 等高刷新策略中表现强。
- `label1d_fusion_valid_alpha` 的最优 alpha 为 `0`，因此融合模型实际退化成 `label1d_lgb`。

当前候选策略：

```text
label1d_lgb + topk20_drop3
label1d_lgb + topk20_drop5
label1d_lgb + rankbuf_p20_b50_s100_min2_max10
```

## 5. 当前策略状态

### 5.1 保守主策略

```text
label5d_final + rolling_p10_h5
```

定位：

- 当前最稳健的主策略候选。
- test 最大回撤较低。
- 与 5 日标签周期一致。

风险：

- 持仓更集中，实际组合只有约 7-10 只活跃股票。
- 收益弹性不如 `label1d` 高刷新策略。

### 5.2 进攻候选策略

```text
label1d_lgb + topk20_drop3
```

定位：

- 当前最值得继续推进的短周期进攻策略。
- 比 `topk20_drop5` 换手低。
- 比 `rank_buffer` 参数更简单，解释更清楚。

风险：

- 仍然依赖较高换手。
- 真实成交约束后收益可能明显下降。

### 5.3 激进研究策略

```text
label1d_lgb + topk20_drop5
label1d_lgb + rankbuf_p20_b50_s100_min2_max10
```

定位：

- 研究策略，不直接作为实盘首选。
- 用来观察 `label1d` alpha 在高换手条件下的上限。

风险：

- 换手接近或超过 `0.8`。
- 对交易成本、滑点、涨跌停失败非常敏感。

### 5.4 风控对照策略

```text
defensive_rank_buffer
risk_filtered_rank_buffer
risk_budget_rank_buffer
risk_tail_core30_tail70
```

定位：

- 不作为当前主策略。
- 用来做压力期和风险暴露对照。
- 后续可发展为组合层风险控制 sleeve。

## 6. 下一阶段任务清单

### 6.1 成交假设重构

目标：让策略回测不再只有一个固定 `label_1d` 近似收益。

需要新增：

```text
execution_price_model
```

候选配置：

```text
close_to_close
open_to_close
vwap_to_close
late_vwap_to_close
close_with_slippage
```

第一阶段先实现：

```text
close_to_close
close_with_slippage
```

原因：

- 当前预测文件已经带有 `label_1d`，可以先保持原来的 `T+1 close -> T+2 close` 收益口径。
- `close_with_slippage` 用额外滑点惩罚近似真实成交偏差，能先测试高换手策略是否还站得住。
- `VWAP` 如果没有分钟数据，只能作为后续扩展。
- `open_to_close` 需要把本地日线 `open/close` 面板按 `buy_date/sell_date` 接入预测样本；当前仓库没有 `data/processed/panel.parquet`，所以放到第二阶段。

需要输出：

```text
metrics_long.csv 增加 execution_price_model 字段
equity_long.parquet 增加 fee_cost/slippage_cost/total_cost 字段
summary.json 记录交易约束配置
```

当前已经实现：

```text
transaction_cost_bps 手续费/冲击基础成本
slippage_bps 额外滑点成本
total_cost_bps = transaction_cost_bps + slippage_bps
execution_price_model = close_to_close / close_with_slippage
```

### 6.2 交易成本敏感性

核心策略必须跑以下成本：

```text
5 bps
10 bps
20 bps
50 bps
```

重点观察：

- Sharpe 是否稳定。
- 最大回撤是否恶化。
- `label1d_lgb + topk20_drop3` 是否仍然优于 `label5d_final + rolling_p10_h5`。
- `topk20_drop5` 是否在高成本下被淘汰。

输出要求：

```text
cost_sensitivity_metrics.csv
cost_sensitivity_heatmap.svg
```

当前已经实现为：

```text
命令：python -m src.experiments strategy-sensitivity --strategy-run label1d_vs_label5d_cost_sensitivity
指标长表：sensitivity_metrics.csv
valid 选择表：best_by_valid_cost.csv
热力图：plots/<split>_<metric>.svg
```

当前矩阵口径：

```text
total_cost_bps = transaction_cost_bps + slippage_bps
transaction_cost_bps 上限默认 5 bps
total_cost_bps 超过 5 bps 的部分记为 slippage_bps
```

### 6.3 涨跌停与无法成交约束

下一阶段至少加入：

```text
涨停不能买入
跌停不能卖出
停牌不能交易
成交额过低跳过
买入候选不足时顺延到下一名
卖出失败时持仓保留并记录失败原因
```

当前已经实现的日线约束：

```text
从 data/processed/universe.parquet 合并 in_universe/is_st/passes_liquidity/amount_mean_20
生成 is_buyable
新增买入只允许来自 is_buyable 股票
已有持仓仍保留在完整日截面中估值和卖出判断
```

当前尚未实现：

```text
涨停不能买入
跌停不能卖出
停牌不能交易
卖出失败后保留原持仓并记录失败原因
```

原因是当前 `universe.parquet` 只有 ST、股票池和流动性字段，没有涨跌停状态和停牌状态字段。需要补充日频交易状态或分钟数据后再做精确约束。

需要在 `trades.csv` 中新增字段：

```text
requested_action
executed_action
blocked_reason
target_weight
executed_weight
execution_price
```

### 6.4 换手约束

对高换手策略新增组合层约束：

```text
max_daily_names_to_trade
max_daily_turnover
min_score_gap_to_trade
```

优先测试：

```text
topk20_drop3，最多每日换 3 只
topk20_drop5，最多每日换 3 只或 4 只
rank_buffer，每日最大主动卖出 3-4 只
```

目的：

- 让 `label1d` 策略更接近可执行交易。
- 防止回测收益来自过度频繁调仓。

### 6.4.1 市场压力降仓与组合回撤控制

当前已实现组合层风控：

```text
命令：python -m src.experiments strategy-backtest --strategy-run label1d_vs_label5d_risk_controls
```

市场压力降仓：

```text
过去 5 个已实现市场近似收益累计小于 -3%
组合仓位上限降到 50%
market_stress_lag = 2，避免使用尚未完整实现的 label_1d 收益
```

组合回撤控制：

```text
组合自身回撤 <= -8%：仓位上限 50%
组合自身回撤 <= -12%：仓位上限 25%
组合自身回撤 <= -18%：仓位上限 20%
```

注意：

```text
这是组合层降仓，不是涨跌停、停牌、卖出失败模拟。
如果需要精确处理无法卖出，需要补充日频交易状态或分钟数据。
```

为了避免单点风控过硬、牺牲过多收益，当前新增风险收益权衡扫描：

```text
命令：python -m src.experiments strategy-risk-sweep --strategy-run label1d_vs_label5d_risk_return_sweep
```

扫描对象：

```text
无风控
只做市场压力降仓
只做组合回撤控制
市场压力 + 组合回撤组合控制
```

筛选原则：

```text
先看 valid 最大回撤是否不低于 -25%
再看风险收益综合分
综合分会奖励 Sharpe 和年化收益
综合分会惩罚超过回撤上限、换手过高、平均仓位过低
最后只用 test 做外推验证
```

输出：

```text
risk_sweep_metrics.csv
risk_sweep_selected.csv
risk_sweep_pareto_valid.csv
plots/valid_risk_return_scatter.svg
plots/test_risk_return_scatter.svg
```

### 6.5 组合层策略

不要二选一押注 `label1d` 或 `label5d`。下一阶段应测试组合层混合：

```text
70% label5d_final + rolling_p10_h5
30% label1d_lgb + topk20_drop3
```

候选权重：

```text
80 / 20
70 / 30
60 / 40
```

目标：

- 保留 `label5d` 的低回撤特征。
- 引入 `label1d` 的短周期进攻 alpha。
- 降低单一策略在特定市场环境下失效的风险。

### 6.6 策略选择协议固定

必须固定以下流程：

```text
valid 集选择参数
test 集只做最终验证
live 阶段只执行，不再调参
```

策略评分不只看 Sharpe。建议增加综合分：

```text
score = Sharpe - drawdown_penalty - turnover_penalty
```

初始版本：

```text
drawdown_penalty = max(0, abs(max_drawdown) - 0.12) * 5
turnover_penalty = max(0, avg_turnover - 0.50) * 1
```

这不是最终公式，只作为第一版防止选择过度高换手策略。

### 6.7 模型后续计划

在真实成交和交易约束回测完成前，不优先训练更复杂模型。

后续模型实验排队如下：

1. 多目标模型：同时预测 `label_1d__cs_rank` 和 `label_5d__cs_rank`。
2. regime-aware 模型：区分普通市场和压力市场。
3. 风险中性化训练：降低市值、流动性、行业暴露。
4. cost-aware ranking：训练或融合时惩罚高换手信号。
5. 定期滚动再训练：按月或按季度刷新模型。

模型实验启动条件：

```text
成交假设回测完成
交易成本敏感性完成
核心策略在 20bps 以上仍有稳定优势
```

## 7. 推荐执行顺序

第一阶段：

```text
1. 已实现 close_to_close / close_with_slippage
2. 已实现基于 universe.parquet 的新增买入过滤
3. 已跑 label1d_vs_label5d_realistic 核心策略
4. 已实现 5/10/20/50 bps 成本敏感性矩阵
5. 后续把成本热力图并入 HTML 总报告
```

第二阶段：

```text
1. 加入涨跌停、停牌和成交失败约束
2. 重跑核心策略
3. 对 label1d_lgb + topk20_drop3 做换手约束版本
4. 对 label5d + label1d 做组合层混合策略
```

第三阶段：

```text
1. 根据真实约束后的结果决定是否继续训练模型
2. 如果 label1d 仍强，训练多目标模型
3. 如果回撤仍主要来自小盘/流动性暴露，做风险中性化或 regime 模型
```

## 8. 当前建议

当前暂定：

```text
保守主策略：label5d_final + rolling_p10_h5
进攻候选：label1d_lgb + topk20_drop3
激进观察：label1d_lgb + topk20_drop5
高换手观察：label1d_lgb + rank_buffer
```

在完成成交假设和成本敏感性之前，不把 `topk20_drop5` 或 `rank_buffer` 作为实盘主策略。
