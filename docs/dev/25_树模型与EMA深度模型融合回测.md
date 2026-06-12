# 树模型与 EMA 深度模型融合回测

本轮实验检验一个具体问题：

> EMA 后的深度模型是否应该和当前主模型 `LGB+XGB rank mean` 融合？

实验只做受限融合，不做复杂 stacking：

```text
final_score = (1 - alpha) * tree_rank + alpha * deep_rank
```

其中：

- `tree_rank`：`LGB+XGB rank mean` 主模型。
- `deep_rank`：某个 EMA 深度模型的每日横截面 rank。
- `alpha`：深度模型权重。

## 输入模型

树模型：

```text
outputs/models/20260612_151735__nsntk_inspired_label1d/main_model_stability
```

深度模型：

| 实验 | 深度模型输入 |
|---|---|
| `tree_plus_mlp_ema0999_3seed` | `MLP EMA 0.999，3 seed rank mean` |
| `tree_plus_gru_ema0995_3seed` | `GRU EMA 0.995，3 seed rank mean` |
| `tree_plus_tcn_ema0995` | `TCN EMA 0.995，单 seed` |

正式输出目录：

```text
outputs/models/20260613_012152__tree_deep_ema_fusion_backtest_leftjoin
```

核心文件：

```text
outputs/models/20260613_012152__tree_deep_ema_fusion_backtest_leftjoin/all_alpha_grid.csv
outputs/models/20260613_012152__tree_deep_ema_fusion_backtest_leftjoin/selected_by_valid_sharpe.csv
outputs/models/20260613_012152__tree_deep_ema_fusion_backtest_leftjoin/prediction_coverage.csv
```

## 评估协议

alpha 网格：

```text
0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30
```

策略：

```text
topk10_drop2
交易成本 5 bps
```

选择规则：

1. 只在 valid 上选择 `alpha`。
2. 选择标准是 valid 的 `topk10_drop2` 夏普。
3. test 只用 valid 选出的 `alpha` 评估一次。

## 覆盖率处理

这点非常重要。

第一轮试跑时使用 inner join，导致 GRU/TCN 的 `alpha=0` 也被限制到深度模型覆盖的股票池，基准不再等于原始树模型。这个结果被废弃。

正式实验改为：

```text
以树模型股票池为全集 left join；
深度模型缺失分数时，deep_rank 回退为 tree_rank。
```

这样：

- `alpha=0` 永远等于原始树模型。
- 深度模型只在自己有预测的位置影响排序。
- 不会因为深度模型覆盖率不同而改变基准股票池。

覆盖率：

| 实验 | valid 覆盖率 | test 覆盖率 |
|---|---:|---:|
| MLP EMA 0.999 3 seed | 100.00% | 100.00% |
| GRU EMA 0.995 3 seed | 81.25% | 79.61% |
| TCN EMA 0.995 | 81.25% | 79.61% |

## 按 valid 选择后的正式结果

| 实验 | valid 选中 alpha | valid 夏普 | test IC | test ICIR | test 总收益 | test 夏普 | 最大回撤 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 树模型基线 | 0.00 | 3.6758 | 0.0820 | 0.6094 | 16.1583 | 7.1961 | -12.25% | 0.6675 |
| Tree + MLP EMA | 0.00 | 3.6758 | 0.0820 | 0.6094 | 16.1583 | 7.1961 | -12.25% | 0.6675 |
| Tree + GRU EMA | 0.00 | 3.6758 | 0.0820 | 0.6094 | 16.1583 | 7.1961 | -12.25% | 0.6675 |
| Tree + TCN EMA | 0.02 | 3.7195 | 0.0822 | 0.6108 | 15.0982 | 7.5192 | -12.68% | 0.7387 |

正式结论：

- MLP EMA 融合没有通过 valid 选择，最优是 `alpha=0`。
- GRU EMA 融合没有通过 valid 选择，最优也是 `alpha=0`。
- TCN EMA 融合在 valid 上选出 `alpha=0.02`，test 夏普从 7.1961 提升到 7.5192，ICIR 从 0.6094 提升到 0.6108。
- 但是 TCN 融合的 test 总收益从 16.1583 降到 15.0982，换手从 0.6675 升到 0.7387，最大回撤略变差。

因此，TCN 融合只能算轻微稳定性改善，不能算全面优于主模型。

## test 上的探索性最优

下面结果不能作为正式模型选择依据，因为它们用了 test 网格最优，只能帮助理解潜力。

| 实验 | test 最优 alpha | test ICIR | test 总收益 | test 夏普 | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| Tree + MLP EMA | 0.02 | 0.6089 | 16.5870 | 7.8961 | -14.07% |
| Tree + GRU EMA | 0.00 | 0.6094 | 16.1583 | 7.1961 | -12.25% |
| Tree + TCN EMA | 0.05 | 0.6128 | 17.1575 | 8.0233 | -12.42% |

这些数字说明：

- 深度模型和树模型确实不是完全重复信号。
- TCN 可能有少量互补信息。
- 但 alpha 的选择对 valid/test 很敏感，不能直接把 test 最优当作最终策略。

## 报告写法

建议报告里写：

1. 深度模型经过 EMA 后有较强独立信号，因此进一步尝试与树模型做预测层融合。
2. 为避免过拟合，只使用一个简单的 rank 层线性融合，并用 valid 选择深度模型权重。
3. MLP 和 GRU 融合未通过 valid 选择，说明单模型 ICIR 高不代表一定能改善最终组合。
4. TCN 以极小权重 `alpha=0.02` 被选中，test 夏普小幅提升，但总收益下降、换手上升，因此最终是否采用需要谨慎。
5. 最终主模型仍建议保持 `LGB+XGB rank mean`；若报告需要展示深度融合，可把 `Tree+TCN alpha=0.02` 作为探索性增强版本，而不是替代主模型。

## 当前决策

不建议马上把深度融合替换为默认实盘预测模型。

更稳妥的选择：

```text
主模型：LGB+XGB rank mean
候选增强：LGB+XGB + 0.02 * TCN EMA
```

如果后续还想采用 TCN 融合，需要继续补：

1. 月度 IC 和逐年收益曲线对比。
2. 成本敏感性测试。
3. 换手控制后的融合回测。
4. 组合层约束下的收益、回撤、换手对比。
