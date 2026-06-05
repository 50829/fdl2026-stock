# 04 每日调仓 GRU 小样本实验

本文档记录 2026-05-30 跑的每日调仓口径 GRU 小样本实验。编号 `04` 用于和口径修复文档区分。

## 实验目的

实操中如果每天调仓，训练目标和回测收益列需要对齐到 1 日周期。因此本实验测试：

```text
label_1d__cs_rank + label_1d 收益 + step_days=1
```

是否比原来的 5 日目标更适合每日调仓。

## 实验配置

命令：

```bash
python -m src.model_experiments.run_e0_e1 \
  --experiments e1_daily \
  --stage train eval \
  --out-root outputs/models/20260530_162345__sequence_daily_pilot
```

配置文件：

```text
configs/exp_e1_gru_1d_rank_daily_pilot.yaml
```

模型：

```text
layer1 GRU + 112 个特征 + 回看窗口=60 + attention
```

训练：

```text
target: label_1d__cs_rank
loss: SmoothL1
batch_size: 4096
epochs: 8
早停耐心轮数：2
最佳 epoch：2
```

回测：

```text
return_col: label_1d
n_hold: 20
k_rotate: 20
step_days: 1
transaction_cost_bps: 5.0
```

## 训练过程

| epoch | train 损失 | valid 损失 | 备注 |
| ---: | ---: | ---: | --- |
| 1 | 0.162300 | 0.163091 | 保存最佳模型 |
| 2 | 0.161592 | 0.163028 | 保存最佳模型 |
| 3 | 0.160959 | 0.163347 | valid 变差 |
| 4 | 0.160039 | 0.164430 | 触发早停 |

## 评估结果

| 数据集 | 样本数 | 天数 | MSE | IC | ICIR | 回测收益 | 年化收益 | Sharpe | 最大回撤 | 平均换手 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 766,867 | 242 | 0.326232 | 0.064111 | 0.355846 | 0.126550 | 0.132111 | 0.539897 | -0.334207 | 1.034298 |
| test | 308,885 | 97 | 0.324095 | 0.062969 | 0.349483 | 0.078522 | 0.216993 | 1.082123 | -0.135021 | 1.639175 |

## 结论

1 日目标存在稳定信号：

```text
valid IC = 0.064111
test IC = 0.062969
```

valid/test 方向一致，说明模型不是完全失效。

但相比之前 5 日目标的 GRU，daily GRU 的 IC/ICIR 明显更低：

```text
5d layer1 GRU valid IC 约 0.09+
1d daily GRU valid IC 约 0.064
```

这说明当前特征和 GRU 结构更擅长预测 5 日截面排序，而不是 1 日短周期排序。

## 风险与注意

当前每日回测是“每天全调仓”：

```text
k_rotate = 20
n_hold = 20
```

这会带来较高换手，交易成本敏感。valid 平均换手约 1.03，test 平均换手约 1.64。

此外，小样本 test 只有 97 个交易日，不能直接代表全量 test 稳定性。

## 下一步建议

暂时不建议直接扩全量 daily GRU。更合理的下一步是先实现：

```text
label_5d + daily rolling tranche backtest
```

也就是：

```text
每天买入一批
每批持有 5 天
每天卖出到期 tranche
每天补入新 tranche
```

这样可以同时保留 5 日目标较强的 IC/ICIR，又更贴近每天调仓的实操方式。
