# 20 label1d expanding walk-forward 实验报告

日期：2026-06-09

## 1. 实验目的

本实验接在 `19_label1d_window_ablation_report.md` 后面，目标是验证窗口消融结论是否能跨年份成立。

前一个实验只用了固定验证集和测试集。这个实验改成 expanding walk-forward：

```text
训练从 2016 年开始。
每次用验证年前一年的所有历史数据训练。
逐年外推验证 2021、2022、2023、2024、2025、2026。
```

这样可以更接近真实研究流程：

```text
不能用未来数据训练。
每一年都只允许使用当年之前的数据。
模型窗口方案必须跨年份稳定，而不是只在某个切分上好看。
```

## 2. 实验口径

输出目录：

```text
outputs/models/20260609_172836__label1d_window_walkforward
```

统一入口：

```bash
conda run -n fdl python -m src.experiments label1d-window-walkforward \
  --model lightgbm \
  --processed-dir data/processed \
  --run-name label1d_window_walkforward \
  --feature-root outputs/models/20260609_152416__label1d_window_ablation/features \
  --variants all_windows no_20d short_5_10 \
  --target label_1d__cs_rank \
  --raw-return-col label_1d \
  --daily-return-col label_1d \
  --valid-years 2021 2022 2023 2024 2025 2026 \
  --min-year 2016 \
  --filter-in-universe \
  --num-threads 8 \
  --num-boost-round 800 \
  --early-stopping-rounds 80 \
  --log-period 200 \
  --topk-drops 3 5 \
  --transaction-cost-bps 5.0
```

模型：

```text
LightGBM
```

目标：

```text
label_1d__cs_rank
```

收益列：

```text
label_1d
```

股票池：

```text
filter_in_universe = true
```

策略评估：

```text
topk = 20
step_days = 1
drop = 3 和 5
transaction_cost_bps = 5.0
```

注意：2026 年样本不是完整年份。当前数据只覆盖到 2026 年 6 月初附近，因此 2026 只能作为近期样本参考，不能和 2021-2025 完整年份完全等权解释。

## 3. 对比窗口

| 变体 | 特征数 | 含义 |
| --- | ---: | --- |
| `all_windows` | 40 | 保留当前 top40 全部窗口特征 |
| `no_20d` | 29 | 删除显式依赖 20 日窗口的特征 |
| `short_5_10` | 22 | 只保留无显式窗口依赖、5 日和 10 日窗口特征 |

这三组来自窗口消融后的候选：

```text
all_windows：稳健默认候选。
no_20d：验证集和 topk20_drop5 候选。
short_5_10：进攻候选，测试期表现强。
```

## 4. 总体结果

| 变体 | 平均 IC | 最低 IC | 平均 ICIR | 最低 ICIR | IC 为正年份数 | rolling 平均 Sharpe | rolling 最深回撤 | drop3 平均 Sharpe | drop3 最低 Sharpe | drop3 最深回撤 | drop3 平均换手 | drop5 平均 Sharpe | drop5 最低 Sharpe | drop5 最深回撤 | drop5 平均换手 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all_windows` | 0.071493 | 0.048203 | 0.584399 | 0.323170 | 6 | 3.073835 | -0.326884 | 4.954250 | 2.825156 | -0.359530 | 0.500552 | 5.448182 | 3.179483 | -0.386410 | 0.745090 |
| `short_5_10` | 0.070904 | 0.051369 | 0.581011 | 0.344526 | 6 | 3.314640 | -0.318778 | 5.645741 | 2.762557 | -0.401597 | 0.542738 | 5.751045 | 3.182268 | -0.405528 | 0.800904 |
| `no_20d` | 0.071817 | 0.049943 | 0.578472 | 0.333231 | 6 | 3.048459 | -0.334324 | 5.158828 | 2.632250 | -0.375835 | 0.525735 | 5.521278 | 3.539662 | -0.387972 | 0.779623 |

## 5. 每年 ICIR

| 年份 | `all_windows` | `no_20d` | `short_5_10` |
| --- | ---: | ---: | ---: |
| 2021 | 0.591901 | 0.600987 | 0.548262 |
| 2022 | 0.756259 | 0.703520 | 0.697639 |
| 2023 | 0.687165 | 0.672960 | 0.671563 |
| 2024 | 0.560143 | 0.580271 | 0.543443 |
| 2025 | 0.587754 | 0.579865 | 0.680632 |
| 2026 | 0.323170 | 0.333231 | 0.344526 |

观察：

```text
三组每年 IC 均为正。
all_windows 平均 ICIR 最高。
short_5_10 的最低 ICIR 最高，但主要因为 2026 年略好。
no_20d 没有明显崩掉，说明 20 日窗口不是 label1d 的唯一核心。
```

## 6. 每年 topk20_drop3

| 年份 | `all_windows` Sharpe | `no_20d` Sharpe | `short_5_10` Sharpe |
| --- | ---: | ---: | ---: |
| 2021 | 4.790249 | 4.646029 | 6.435216 |
| 2022 | 5.010198 | 5.618534 | 5.139582 |
| 2023 | 5.036313 | 5.929016 | 5.646258 |
| 2024 | 2.825156 | 2.632250 | 2.762557 |
| 2025 | 6.436446 | 6.852634 | 7.982687 |
| 2026 | 5.627141 | 5.274503 | 5.908147 |

汇总：

```text
short_5_10 平均 Sharpe 最高：5.645741
all_windows 最低 Sharpe 最高：2.825156
no_20d 在 2022、2023、2025 较强，但 2024 较弱
```

## 7. 每年 topk20_drop5

| 年份 | `all_windows` Sharpe | `no_20d` Sharpe | `short_5_10` Sharpe |
| --- | ---: | ---: | ---: |
| 2021 | 5.240351 | 5.406922 | 5.785776 |
| 2022 | 5.529728 | 5.345322 | 5.175912 |
| 2023 | 5.010066 | 5.449594 | 5.452020 |
| 2024 | 3.179483 | 3.539662 | 3.182268 |
| 2025 | 7.348888 | 7.175082 | 8.450183 |
| 2026 | 6.380576 | 6.211087 | 6.460109 |

汇总：

```text
short_5_10 平均 Sharpe 最高：5.751045
no_20d 最低 Sharpe 最高：3.539662
all_windows 换手最低：0.745090
short_5_10 换手最高：0.800904
```

## 8. 2024 年压力样本

2024 年仍然是最需要重视的年份。

topk20_drop3：

| 变体 | Sharpe | 最大回撤 |
| --- | ---: | ---: |
| `all_windows` | 2.825156 | -0.359530 |
| `no_20d` | 2.632250 | -0.375835 |
| `short_5_10` | 2.762557 | -0.401597 |

topk20_drop5：

| 变体 | Sharpe | 最大回撤 |
| --- | ---: | ---: |
| `all_windows` | 3.179483 | -0.386410 |
| `no_20d` | 3.539662 | -0.387972 |
| `short_5_10` | 3.182268 | -0.405528 |

解释：

```text
2024 不是模型完全失效。
IC 仍然为正，ICIR 也不低。
问题主要出现在组合路径：topk 组合的最大回撤明显扩大。
```

这说明后续提升重点不应该只放在训练更强模型，也必须改策略约束：

```text
市场压力降仓
组合回撤控制
单票波动和流动性约束
换手和成本敏感性
行业或风格暴露控制
```

## 9. 结论

当前不建议直接替换 live 默认模型。

更稳妥的结论是：

```text
all_windows 仍然是默认 live 候选。
short_5_10 是进攻候选，平均 topk Sharpe 更高，但 2024 回撤更深、换手更高。
no_20d 是稳健研究候选，drop5 的最低 Sharpe 最好，但整体 ICIR 不如 all_windows。
```

如果按模型稳定性排序：

```text
1. all_windows
2. short_5_10
3. no_20d
```

如果按日频 topk 收益进攻性排序：

```text
1. short_5_10
2. no_20d
3. all_windows
```

如果按实盘可控性排序：

```text
1. all_windows
2. no_20d
3. short_5_10
```

因此当前 live 策略不应只按 Sharpe 最高切换到 `short_5_10`。下一步应该把三组预测接入真实交易约束版策略引擎，比较在同一套市场压力降仓和组合回撤控制下的收益、回撤、换手和成本敏感性。

## 10. 已生成文件

```text
outputs/models/20260609_172836__label1d_window_walkforward/walkforward_by_variant.csv
outputs/models/20260609_172836__label1d_window_walkforward/walkforward_fold_metrics.csv
outputs/models/20260609_172836__label1d_window_walkforward/walkforward_topk_step1_metrics.csv
outputs/models/20260609_172836__label1d_window_walkforward/walkforward_summary.json
outputs/models/20260609_172836__label1d_window_walkforward/walkforward_report.md
```

新增代码入口：

```text
python -m src.experiments label1d-window-walkforward
```

测试：

```text
tests/test_cli_registry.py 已通过。
```
