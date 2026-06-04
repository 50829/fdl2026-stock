# 05 Rolling Tranche Backtest

本文档记录 2026-05-30 对 `label_5d__cs_rank` GRU 信号做的 daily rolling tranche 回测。编号 `05` 用于和 daily GRU pilot 区分。

## 实验目的

实操约束是：

```text
每天必须产生交易列表
单笔可以持有多天
```

因此不一定要把预测目标改成 `label_1d__cs_rank`。更自然的方案是：

```text
继续用更强的 label_5d__cs_rank 信号
每天买入一批股票
每批持有 5 天
每天卖出到期 tranche
```

本实验检验这个策略口径是否比：

```text
label_1d GRU + 每日全换仓
```

更值得继续。

## 代码变更

新增/修改文件：

- `src/model_experiments/run_e0_e1.py`
- `src/model_experiments/run_rolling_tranche_eval.py`

新增回测模式：

```text
rolling_tranche
```

核心参数：

```text
tranche_size = 4
hold_days = 5
target_active = 20
daily_return_col = label_1d
transaction_cost_bps = 5.0
```

策略逻辑：

```text
每天根据 pred 排序
买入 top tranche_size 只未持仓股票
每个 tranche 持有 hold_days 天
每天用 label_1d 计算所有 active positions 的平均收益
扣除买入和到期卖出的交易成本
```

注意：这里没有把 `label_5d` 当作每日收益复利，而是使用 `label_1d` 做逐日净值更新。

## 使用的预测文件

本实验没有重新训练模型，直接复用已有最佳 5d layer1 GRU 的预测：

```text
valid: outputs/models/20260530_103415__sequence_ablation_full/layer1/valid/valid_pred.parquet
test:  outputs/models/20260530_103903__final_test_eval/layer1/test/test_pred.parquet
```

命令：

```bash
python -m src.model_experiments.run_rolling_tranche_eval \
  --out-root outputs/models/20260530_194341__rolling_tranche_eval
```

输出：

```text
outputs/models/20260530_194341__rolling_tranche_eval/summary.json
```

## 结果

### 5d 信号质量

| split | samples | days | IC | ICIR |
| --- | ---: | ---: | ---: | ---: |
| valid | 766,867 | 242 | 0.095154 | 0.511638 |
| test | 1,019,298 | 323 | 0.085123 | 0.648694 |

`label_5d__cs_rank` 的排序信号仍然稳定，明显强于上一轮 `label_1d__cs_rank` daily GRU。

### 回测对比

| split | 策略 | periods | total return | annual return | Sharpe | max drawdown | avg turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 5d non-overlap TopK | 49 | 0.126012 | 0.129836 | 0.488961 | -0.373277 | 0.510204 |
| valid | daily rolling tranche | 242 | -0.054890 | -0.057092 | 0.074668 | -0.438514 | 0.395868 |
| test | 5d non-overlap TopK | 65 | 0.358659 | 0.268270 | 1.248049 | -0.194366 | 0.507692 |
| test | daily rolling tranche | 323 | 0.875401 | 0.633293 | 2.305933 | -0.124192 | 0.396904 |

## 解读

这组结果有明显分化：

- test 上 daily rolling tranche 很强，收益、Sharpe、回撤都优于 5d non-overlap。
- valid 上 daily rolling tranche 是负收益，且最大回撤更大。
- 5d IC/ICIR 在 valid/test 都稳定，但把信号转成 daily rolling 策略后，valid 组合收益没有同步变好。

因此结论不能简单写成 rolling tranche 已经胜出。更准确的判断是：

```text
rolling tranche 策略口径是正确的，但当前选股/持仓规则还不稳定。
```

## 可能原因

1. `label_5d__cs_rank` 优化的是 5 日累计相对排序，不保证每天 mark-to-market 都平滑。
2. rolling tranche 使用 `label_1d` 做每日净值，可能暴露了 5 日信号在持有期内的路径波动。
3. 当前每天只买 4 只，组合分散度低，对单日噪声敏感。
4. valid/test 市场环境差异较大，test 阶段更适合当前 5d 信号。
5. 当前策略没有行业、市值、中性化约束，也没有风险预算。

## 下一步建议

不建议立刻把 daily GRU 或 rolling tranche 直接定为最终方案。建议先做 rolling tranche 参数敏感性：

```text
tranche_size = 4, 8, 10
hold_days = 3, 5, 10
transaction_cost_bps = 5, 10
```

重点看：

```text
valid 是否从负收益改善
test 是否仍保持优势
IC/ICIR 是否稳定
最大回撤是否下降
```

如果 rolling tranche 的 valid 表现能通过更分散的组合改善，则继续用：

```text
label_5d__cs_rank + daily rolling tranche
```

如果 valid 始终不稳定，则应转向：

```text
daily CorrLoss / LightGBM ensemble / 风险约束后的组合构建
```
