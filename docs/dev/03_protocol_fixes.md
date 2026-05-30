# 03 Protocol Fixes

本文档记录 2026-05-30 对训练、验证、评估和回测口径做的修复。编号 `03` 用于和已有实验结果文档区分。

## 修复背景

代码审查发现，前一阶段实验结果本身没有明显未来函数，但存在几个口径不一致问题：

- 当前最佳 GRU 只存在于 `src/models/sdd/run_ablation.py`，正式 `src/train.py` 仍然固定使用 MSE、无 early stopping。
- sequence model 的训练 valid loss 只在 valid split 内构造 lookback，少掉 valid 开头 59 个交易日；最终评估却使用 split 前历史 warmup。
- GRU 序列样本原先要求整个 lookback 窗口都在 universe，MLP 只要求预测日当天在 universe，二者 coverage 不完全可比。
- 回测调仓周期固定在代码里，不方便区分 5 日非重叠回测和 1 日 daily rebalance。

## 已完成代码修复

### 1. 新增 warmup-aware labeled sequence iterator

新增入口：

```text
src.data.iter_processed_sequence_labeled_feature_batches
```

用途：

```text
start_date: 用于构造历史窗口的起始日期
emit_start_date: 真正产生训练/验证样本的起始日期
end_date: 样本结束日期
```

这样 valid loss 可以和最终 evaluate 一样：

```text
使用 valid 前的历史特征构造窗口
只在 valid 日期内产生样本
```

### 2. universe 过滤改成预测日口径

旧口径：

```text
60 日 lookback 中每一天都必须在 universe
```

新口径：

```text
历史窗口只要求有特征
样本是否纳入训练/评估，只看预测日当天是否在 universe
```

这个口径更接近实盘：历史特征可以使用，是否交易看当天股票池。

### 3. `src/train.py` 支持 loss 和 early stopping

正式训练入口现在支持：

```yaml
train:
  loss: smooth_l1
  patience: 2
  min_delta: 0.0
```

支持的基础 loss：

```text
mse
smooth_l1
```

sequence model 的 train/valid 都改用新的 labeled sequence iterator。valid loss 使用 warmup 口径。

### 4. `sdd/run_ablation.py` 训练口径同步

`src/models/sdd/run_ablation.py` 也改为使用同一套 sequence iterator。

因此：

```text
early stopping 用的 valid loss
最终输出的 valid IC/ICIR
```

现在覆盖同一个 valid 日期范围。

### 5. 回测参数改为配置读取

`src/models/sdd/run_e0_e1.py` 现在支持：

```yaml
backtest:
  return_col: label_5d
  n_hold: 20
  k_rotate: 5
  step_days: 5
  transaction_cost_bps: 5.0
```

对于 daily rebalance 口径，新增配置使用：

```yaml
task:
  label: label_1d__cs_rank

backtest:
  return_col: label_1d
  n_hold: 20
  k_rotate: 20
  step_days: 1
```

注意：不能简单用 `label_5d` 配 `step_days=1` 当作 daily return，否则会把 5 日收益当成 1 日收益复利，回测会失真。

## 配置修复

原 E1 GRU 配置已更新为当前最佳 GRU 主线：

```text
layer1 GRU + 112 features + lookback=60 + attention + SmoothL1 + early stopping
```

涉及文件：

- `configs/exp_e1_gru_5d_rank.yaml`
- `configs/exp_e1_gru_5d_rank_pilot.yaml`

新增 daily 口径配置：

- `configs/exp_e1_gru_1d_rank_daily.yaml`
- `configs/exp_e1_gru_1d_rank_daily_pilot.yaml`

## 验证结果

修复前，pilot valid loss 的 sequence 样本覆盖：

```text
valid samples: 581,227
valid days: 183
first valid sample date: 2024-04-02
```

修复后，pilot valid loss 的 sequence 样本覆盖：

```text
valid samples: 766,867
valid days: 242
first valid sample date: 2024-01-02
```

这和最终 `evaluate_split` 的 valid 覆盖一致。

端到端 smoke run：

```bash
python -m src.models.sdd.run_ablation \
  --experiments layer1 \
  --out-root outputs/sdd_protocol_smoke \
  --epochs 1
```

结果：

| 实验 | epochs | valid samples | IC | ICIR | MSE | 回测 step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer1 | 1 | 766,867 | 0.092334 | 0.467922 | 0.328112 | 5 |

结果量级和旧 layer1 pilot 接近，说明修复后没有明显异常跳变。

## 当前仍未完成的口径

daily rebalance 的基础配置已经新增，但还没有重新训练 `label_1d__cs_rank` 模型。

如果要严格做“每天调仓但持有 5 天滚动组合”，还需要新增 rolling tranche backtest：

```text
每天买入一批
每批持有 5 天
每天卖出到期 tranche
每天补入新 tranche
```

当前实现支持的是：

- `label_5d` + `step_days=5` 的 5 日非重叠回测。
- `label_1d` + `step_days=1` 的每日全调仓回测。

## 下一步

建议先跑 daily 口径 pilot：

```bash
python -m src.models.sdd.run_e0_e1 \
  --experiments e1_daily \
  --stage train eval \
  --out-root outputs/sdd_daily_pilot
```

然后和当前 `label_5d` GRU 对比 daily IC/ICIR，再决定是走 1 日预测，还是实现 5 日 rolling tranche 回测。
