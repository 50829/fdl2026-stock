# 序列长度消融、树深融合与行业图传播实验

本文记录 2026-06-13 补跑的三组实验：GRU/TCN 序列长度消融、树模型与深度模型融合、行业图传播基线。所有实验沿用 label1d、`topk10_drop2`、5bps 交易成本口径。

## 实验输出

主要输出目录：

- 序列长度消融与融合：`outputs/models/20260613_014931__seq_len_fusion_label1d`
- 行业图传播基线：`outputs/models/20260613_094806__gcn_industry_graph_propagation_label1d`
- 报告图表：`outputs/report_figures/20260613_095357__final_report_figures_with_seq_len`

新增可复现实验入口：

```bash
PYTHONPATH=. conda run --no-capture-output -n fdl python -m src.experiments seq-len-fusion \
  --run-name seq_len_fusion_label1d \
  --seq-lens 20 30 60 \
  --train-seq-lens 20 30 \
  --ema-decay 0.995

PYTHONPATH=. conda run --no-capture-output -n fdl python -m src.experiments gcn-propagation \
  --run-name gcn_industry_graph_propagation_label1d

PYTHONPATH=. conda run --no-capture-output -n fdl python -m src.experiments final-report-figures \
  --run-name final_report_figures_with_seq_len \
  --seq-len-run outputs/models/20260613_014931__seq_len_fusion_label1d
```

## 序列长度消融

实验设置：

- GRU：`seq_len=20/30/60`，其中 20/30 重新训练 3 个 seed，60 复用既有 3-seed EMA 结果。
- TCN：`seq_len=20/30/60`，其中 20/30 重新训练 1 个 seed，60 复用既有 EMA 结果。
- EMA decay 固定为 `0.995`。
- valid 只用于模型/融合权重选择，test 只评估一次。

候选模型 test 结果：

| 模型 | 序列长度 | 覆盖率 | IC | ICIR | TopK10 Drop2 Sharpe | 最大回撤 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRU 3-seed EMA | 20 | 90.84% | 0.0744 | 0.5593 | 3.2391 | -13.03% | 0.4378 |
| GRU 3-seed EMA | 30 | 87.39% | 0.0744 | 0.5531 | 2.7565 | -14.30% | 0.4365 |
| GRU 3-seed EMA | 60 | 79.61% | 0.0735 | 0.5215 | 2.5654 | -15.77% | 0.4322 |
| TCN EMA | 20 | 90.84% | 0.0701 | 0.5635 | 2.4108 | -17.35% | 0.4632 |
| TCN EMA | 30 | 87.39% | 0.0741 | 0.5616 | 3.2998 | -12.15% | 0.4427 |
| TCN EMA | 60 | 79.61% | 0.0737 | 0.5652 | 2.6123 | -11.84% | 0.4334 |

结论：

- 降低序列长度确实提高覆盖率：`seq_len=60` 约 79.6%，`seq_len=30` 约 87.4%，`seq_len=20` 约 90.8%。
- GRU 的短序列覆盖率提升明显，但 test ICIR 没有明显优于 `seq_len=60`；`seq_len=20` 的 Sharpe 更好。
- TCN 的 `seq_len=30` 是这一组里最均衡的单深度模型：覆盖率高于 60，Sharpe 高于 20/60，回撤接近 60。
- 深度模型本身仍弱于树模型主线，不能替代 LGB+XGB rank 主模型。

## 树模型与深度模型融合

融合方式：

```text
final_score = (1 - alpha_deep) * tree_rank + alpha_deep * deep_rank
```

如果深度模型没有覆盖某只股票，则用树模型 rank 回填，避免缩小回测股票池。`alpha_deep` 在 valid 上按 TopK10 Drop2 Sharpe 选择，test 只评估一次。

test 结果：

| 融合候选 | valid 选中 alpha_deep | IC | ICIR | TopK10 Drop2 总收益 | Sharpe | 最大回撤 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 树模型主线 | 0.00 | 0.0820 | 0.6094 | 16.1583 | 7.1961 | -12.25% | 0.6675 |
| GRU seq20 | 0.00 | 0.0820 | 0.6094 | 16.1583 | 7.1961 | -12.25% | 0.6675 |
| GRU seq30 | 0.00 | 0.0820 | 0.6094 | 16.1583 | 7.1961 | -12.25% | 0.6675 |
| GRU seq60 | 0.00 | 0.0820 | 0.6094 | 16.1583 | 7.1961 | -12.25% | 0.6675 |
| TCN seq20 | 0.00 | 0.0820 | 0.6094 | 16.1583 | 7.1961 | -12.25% | 0.6675 |
| TCN seq30 | 0.05 | 0.0825 | 0.6114 | 14.6989 | 7.4040 | -11.09% | 0.7492 |
| TCN seq60 | 0.02 | 0.0822 | 0.6108 | 15.0982 | 7.5192 | -12.68% | 0.7387 |

结论：

- GRU 融合权重全部被 valid 选为 0，说明在当前协议下 GRU 没有稳定的样本外增益。
- TCN 有微弱增益：`seq30 alpha=0.05` 提高 ICIR 和 Sharpe，并把最大回撤从 -12.25% 降到 -11.09%，但总收益下降、换手升高。
- `seq60 alpha=0.02` 的 Sharpe 更高，但回撤略差于树模型，覆盖率也最低。
- 报告里更稳妥的表述是：TCN 信号可作为小权重融合项，而不是主模型。

## 行业图传播基线

本次没有直接训练复杂 GCN，而是先做一个可解释的行业图传播基线：

```text
graph_score = (1 - alpha_graph) * self_rank + alpha_graph * industry_neighbor_mean_rank
```

其中 `industry_neighbor_mean_rank` 是同一交易日、同一行业内其他股票的平均 rank。该实验等价于一层固定邻接、无训练权重的图传播，用来判断“行业邻居信息”是否值得继续做完整 GCN。

valid 选择结果：

| split | alpha_graph | IC | ICIR | TopK10 Drop2 总收益 | Sharpe | 最大回撤 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|---:|
| valid | 0.00 | 0.0831 | 0.5531 | 3.9826 | 3.6758 | -31.85% | 0.6421 |
| test | 0.00 | 0.0820 | 0.6094 | 16.1583 | 7.1961 | -12.25% | 0.6675 |

这些数字与主模型相同，是因为 valid 选择了 `alpha_graph=0`。此时图传播被关闭，`final_score` 退化为树模型分数的日内 rank 单调变换；IC 和 TopK10 Drop2 只看排序，因此结果与主模型一致。这不是图模型有效，而是图模型没有被选中。

valid 网格中 `alpha_graph=0.05` 的 IC 略高，但 Sharpe 明显低于 0，因此最终选择 0。结论是：简单行业图传播没有带来交易层面的净增益。下一步如果继续做 GCN，应使用更强的图构造，而不是仅靠静态行业平均。

## 报告图表

图表输出目录：`outputs/report_figures/20260613_095357__final_report_figures_with_seq_len`

建议放进报告的图：

- `label1d_tree_model_icir.svg`：LGB / XGB / LGB+XGB 主模型 ICIR 对比。
- `label1d_tree_model_sharpe.svg`：树模型 TopK10 Drop2 Sharpe 对比。
- `deep_raw_vs_ema_icir.svg`：MLP/GRU/TCN raw vs EMA 的 ICIR。
- `tree_deep_fusion_sharpe.svg`：树深融合 Sharpe。
- `main_model_monthly_ic.svg`：主模型月度 IC。
- `main_model_yearly_icir.svg`：主模型年度 ICIR。
- `topk10_drop2_equity.svg`：主模型、分数平滑、最优树深融合净值曲线。
- `topk10_drop2_drawdown.svg`：对应回撤曲线。
- `topk10_drop2_turnover.svg`：对应 20 日平均换手曲线。

策略曲线汇总：

| 曲线 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 平均换手 |
|---|---:|---:|---:|---:|---:|
| 树模型主线 | 16.1583 | 8.1858 | 7.1961 | -12.25% | 0.6675 |
| 分数平滑 0.6 | 5.4163 | 3.2641 | 5.3635 | -14.72% | 0.6910 |
| 最优树深融合 | 15.0982 | 7.7400 | 7.5192 | -12.68% | 0.7387 |

## 报告写法建议

最终模型选择建议：

- 主模型：LGB+XGB rank 融合。
- 可选增强：TCN 小权重融合，强调这是“稳定性增强/探索性增强”，不是收益主引擎。
- 不建议把 GRU 或当前行业图传播写成有效增益，因为 valid 选择结果并不支持。

报告中的结论可以写为：

1. 树模型仍是当前最强、最稳的主模型。
2. 深度序列模型通过 EMA 和短序列长度可以显著提高覆盖率和稳定性，但单模型收益不足。
3. TCN 与树模型存在少量互补信息，小权重融合能提升 Sharpe/ICIR，但伴随更高换手和总收益下降。
4. 简单行业图传播没有贡献，说明跨股票关系建模需要更细的图结构，例如滚动相关性图、行业内残差图、供应链/主题图，不能只靠静态行业平均。

## 滚动相关图传播补充

输出目录：`outputs/models/20260613_100914__gcn_rolling_corr_label1d`

命令：

```bash
conda run --no-capture-output -n fdl python -m src.experiments gcn-rolling-corr \
  --run-name gcn_rolling_corr_label1d \
  --window-days 60 \
  --top-k-neighbors 10 \
  --min-obs 40 \
  --min-corr 0.20 \
  --rebalance M
```

设计：

- 每个月重新建图。
- 图边来自该月开始前 60 个交易日的 `ret_1__cs_rank` 滚动相关性。
- 每只股票最多保留 top10 个正相关邻居。
- 预测分数仍然是：

```text
final_score = (1 - alpha_graph) * self_rank + alpha_graph * corr_neighbor_rank
```

valid 选择结果：

| split | alpha_graph | IC | ICIR | TopK10 Drop2 总收益 | Sharpe | 最大回撤 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|---:|
| valid | 0.00 | 0.0831 | 0.5531 | 3.9826 | 3.6758 | -31.85% | 0.6421 |
| test | 0.00 | 0.0820 | 0.6094 | 16.1583 | 7.1961 | -12.25% | 0.6675 |

同样，这里的 test 结果等于主模型，是因为 valid 选择 `alpha_graph=0`。非零 alpha 的预测文件和回测结果并不相同，说明代码确实执行了图传播；只是验证集认为这些图传播分数不值得加入最终模型。

非零 alpha 的观察：

- 正 alpha 有时能轻微提高 valid IC，但会显著降低 valid Sharpe，并提高换手。
- 负 alpha 在 test 上能提高 ICIR，但收益、夏普和回撤都不如主模型，不能按 test 反向选择。
- 因此 valid 严格选择 `alpha_graph=0`，滚动相关图不进入主模型。

结论：

当前图传播实验是负结果。它说明“相关性邻居均值”不足以改善主模型，后续若继续做 GCN，应改成可训练图模型，或者把图关系用于行业中性化、风险暴露约束，而不是直接平均邻居预测分数。

## 最终报告表格

最终表格输出目录：`outputs/report_tables/20260613_105139__final_report_tables`

| 文件 | 内容 |
|---|---|
| `最终主结果表.csv` | label1d 主模型、label5d 对照、树深融合、图实验总表 |
| `序列长度消融表.csv` | GRU/TCN seq_len=20/30/60 覆盖率、ICIR、回测结果 |
| `深度模型EMA与多种子表.csv` | MLP/GRU/TCN raw、EMA、多 seed ensemble |
| `时间衰减_分数平滑_稳定性表.csv` | 时间衰减、预测平滑、主模型稳定性 |
| `GCN图实验表.csv` | 静态行业图和滚动相关图对比 |

最终报告正文已经更新：`docs/report/实验报告初稿.md` 第 12 节“最终实验冻结版补充”。正式报告应优先引用这一节和最终表格，避免继续引用早期零散实验结论。
