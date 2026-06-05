# 10 策略设计方案

本文档记录模型交接后下一阶段要尝试的策略设计。当前重点不是继续改模型，而是把每日股票打分 `pred` 转换成可执行、可回测、可解释的交易规则。

## 1. 背景

当前最终模型是：

```text
Residual-rank deep_ln, alpha=1.5
```

最终预测文件：

```text
outputs/models/20260531_162154__final_model_handoff/valid/valid_pred.parquet
outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet
```

策略侧只应使用：

```text
trade_date, ts_code, pred
```

其中 `pred` 是每日截面排序分数，不是收益率预测值，也不是交易指令。`label_5d__cs_rank`、`label_5d`、`label_1d` 只能用于回测评估，不能用于选股。

## 2. 主流策略口径

学界常见做法是预测横截面股票收益或排序，然后按预测分数构造 top quantile / top decile 组合，或构造 long-short spread portfolio。由于本作业模拟交易是 A 股 long-only、满仓约束，不能直接使用 long-short，适合改成 long-only top ranked portfolio。

工业界和开源量化框架中常见的简化口径是：

```text
模型输出分数 -> 每日排序 -> 选 TopK -> 控制换手 -> 等权或风险约束权重
```

Qlib 的 `TopkDropoutStrategy` 是典型例子：持有 TopK，只卖出持仓中排名变差的一部分，再买入新的高分股票。它的核心思想是保持组合始终暴露在高分股票上，同时通过 `Drop` 控制换手。

更复杂的工业组合构建会使用均值方差优化、风险模型、行业暴露约束和交易成本惩罚。但当前模型输出不是可校准收益率，而是排序分数；因此第一阶段不建议直接做 Markowitz 优化。

## 3. 策略设计原则

本项目策略设计应遵守：

1. 使用 `pred` 排序，不使用任何 label 字段。
2. 参数只在 valid 集选择，test 集只做最终评估。
3. 优先等权，避免把不可解释的排序分数当收益率做权重。
4. 控制换手，显式计入交易成本。
5. 保持策略简单，便于模拟交易人工执行和报告解释。
6. 至少和 LightGBM top40 基线使用同一策略做对比。

## 4. 推荐主策略：滚动分层持仓

当前模型主目标是：

```text
label_5d__cs_rank
```

因此最自然的策略是每天买一批，每批持有 5 个交易日。

推荐初始参数：

```text
target_positions = 20
hold_days = 5
daily_buy = 4
weight = equal weight
```

交易逻辑：

```text
每天按 pred 降序排序
跳过已经持仓股票
买入前 daily_buy 只候选股票
每批股票持有 hold_days 个交易日
到期卖出
组合中同时持有最近 hold_days 天买入的 tranche
```

优点：

- 与 5 日标签周期一致。
- 每天都能生成交易列表。
- 换手较可控。
- 易于向报告和模拟交易解释。

需要注意：

- 如果某天卖出失败，持仓可能超过目标天数。
- 如果买入候选不足，需要准备候补列表。
- valid 上 rolling tranche 曾经出现过收益不稳定，因此必须做参数敏感性。

## 5. 对照策略一：TopK 剔除

这是与作业要求和 Qlib 口径最接近的策略。

推荐参数网格：

```text
K = 20, 30
Drop = 1, 2, 3, 5
```

交易逻辑：

```text
初始买入 pred 最高的 K 只股票
之后每天对持仓重新排序
卖出当前持仓中 pred 排名最低的 Drop 只
买入未持仓股票中 pred 最高的 Drop 只
保持持仓数量接近 K
```

重点比较：

```text
K=20, Drop=2
K=20, Drop=4
K=30, Drop=3
```

解读：

- `Drop` 越大，组合越贴近最新模型排序，但换手更高。
- `Drop` 越小，换手更低，但持仓更新慢。
- `K=30` 通常比 `K=20` 更分散，收益弹性可能下降，但回撤可能更稳。

## 6. 对照策略二：排名缓冲

Rank Buffer 用于减少分数小幅波动导致的频繁换仓。

推荐初始参数：

```text
target_positions = 20
buy_rank <= 30
sell_rank > 100
min_hold_days = 2
max_hold_days = 10
```

交易逻辑：

```text
新买入只从每日 top buy_rank 中选择
已持仓股票只要排名没有跌出 sell_rank，就继续持有
持仓不足 target_positions 时，从 top buy_rank 中补足
持有未满 min_hold_days 时尽量不卖
持有超过 max_hold_days 后强制复核，若不在 top buy_rank 则卖出
```

优点：

- 降低无意义换手。
- 对模型日频排序噪声更稳健。
- 更接近真实组合管理中的缓冲区规则。

风险：

- 参数更多，更容易过拟合。
- 需要严格只用 valid 选参数。

## 7. 交易约束

第一版至少实现以下约束：

```text
exclude_st = true
exclude_bj = true
exclude_suspended = true
min_amount_mean_20 = 使用 processed universe 过滤结果
max_single_weight = 5%
transaction_cost_bps = 5, 10, 20 做敏感性
```

如果数据支持，后续增加：

```text
涨停不能买入
跌停不能卖出
单行业持仓数上限
单行业权重上限
最小交易单位
```

当前 processed 数据已经在特征和标签层面做了可交易股票池过滤。策略侧第一版可以先复用预测文件中的股票范围，再补充更细的交易约束。

## 8. 权重设计

第一版使用等权：

```text
weight_i = 1 / number_of_positions
```

原因：

- `pred` 不是收益率，直接按分数加权不严谨。
- 等权最容易复现和解释。
- 回测结果更能反映选股排序能力，而不是权重工程。

可选扩展：

```text
rank_linear_weight
rank_softmax_weight
inverse_volatility_weight
```

这些只能作为附加实验，不作为第一版主策略。

## 9. 参数选择协议

使用 valid 集选择策略参数：

```text
valid: 20240102-20241231
test:  20250102-20260508
```

候选网格：

```text
rolling_tranche:
  target_positions: 10, 20, 30
  hold_days: 3, 5, 10

topk_drop:
  K: 20, 30
  Drop: 1, 2, 3, 5

rank_buffer:
  target_positions: 20, 30
  buy_rank: 20, 30, 50
  sell_rank: 80, 100, 150
  min_hold_days: 1, 2
  max_hold_days: 5, 10
```

排序指标优先级：

```text
1. Sharpe
2. 最大回撤
3. 总收益
4. 换手率
5. valid/test 一致性
```

如果收益最高的参数换手很高或回撤很大，不应作为最终策略。

## 10. 输出产物

建议策略输出统一放在：

```text
outputs/strategy/
```

每个实验目录包含：

```text
outputs/strategy/{strategy_name}/
  valid/
    trades.csv
    holdings.csv
    equity_curve.csv
    metrics.json
  test/
    trades.csv
    holdings.csv
    equity_curve.csv
    metrics.json
  summary.json
```

字段建议：

```text
trades.csv:
  trade_date, action, ts_code, score, shares_or_weight, reason

holdings.csv:
  trade_date, ts_code, weight, entry_date, holding_days, score

equity_curve.csv:
  trade_date, gross_return, cost, net_return, equity, turnover
```

模拟交易每日列表额外输出：

```text
daily_buy_list.csv
daily_sell_list.csv
```

## 11. 代码位置建议

策略相关代码建议新增：

```text
src/strategy/
```

推荐结构：

```text
src/strategy/
  __init__.py
  base.py
  selectors.py
  constraints.py
  portfolio.py
  rules.py
  orders.py
  io.py
```

职责划分：

```text
base.py         策略配置和结果结构
selectors.py    pred 排序、TopK 候选选择
constraints.py  股票池、交易限制、行业/权重约束
portfolio.py    持仓状态和等权计算
rules.py        RollingTranche / TopKDrop / RankBuffer
orders.py       目标持仓转买卖列表
io.py           保存 trades/holdings/equity/metrics
```

运行入口建议放：

```text
src/pipelines/run_strategy_backtest.py
```

而不是继续放入 `src/model_experiments/`，因为策略已经属于模型交付后的组合构建流程，不是模型训练实验本身。

## 12. 下一步执行顺序

建议按以下顺序实现：

1. 实现统一策略状态机和输出格式。
2. 复现已有 `rolling_tranche` 快速回测结果，确认新旧口径一致。
3. 跑 rolling tranche 参数网格。
4. 跑 TopK-Drop 参数网格。
5. 跑 Rank Buffer 小网格。
6. 对最终模型和 LightGBM top40 基线使用同一策略做对照。
7. 固定 valid 选出的最终策略，在 test 上只评一次。

最终报告建议呈现：

```text
模型对比：GRU / LightGBM top40 / Residual-rank deep_ln
策略对比：Rolling Tranche / TopK-Drop / Rank Buffer
敏感性：交易成本、持仓数量、持有天数
风险指标：Sharpe、最大回撤、换手、收益曲线
```

## 13. 当前推荐

当前主推策略：

```text
Residual-rank deep_ln alpha=1.5
+ rolling tranche
+ target_positions=20
+ hold_days=5
+ daily_buy=4
+ equal weight
```

同时保留：

```text
TopK-Drop K=20 Drop=2/4
LightGBM top40 + 同策略基线
```

这样既符合课程要求中的简单交易策略，又能体现模型输出、策略构建、交易成本和风险分析之间的完整链路。
