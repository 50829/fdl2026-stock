# Model Experiment Log

本文档记录 2026-05-30 前后在 `src/models/sdd/` 下完成的模型实验、代码变更和阶段性结论。当前所有调参结论都以验证集为准，测试集只作为最终对比参考，不用于调参。

## 代码变更

新增或修改的主要代码：

- `src/models/sdd/run_e0_e1.py`：统一跑 E0/E1 的训练、验证/测试预测、IC/ICIR、Top-N 回测。
- `src/models/sdd/run_ablation.py`：用于 GRU/TCN 的消融实验，支持 early stopping、best checkpoint、特征组选择和不同 loss。
- `src/data/processed.py`：增加 `cache_in_memory` 路径，避免每个 epoch 反复按日期扫描 parquet；序列样本使用 `sliding_window_view` 批量构造。
- `src/models/lrk/alstm.py`：为 ALSTM/GRU 增加 `use_attention` 开关，用于比较 attention 和 no-attention。
- `src/models/lrk/tcn.py`：新增 TCN 模型，包含 causal dilated convolution、residual block 和可选 temporal attention pooling。
- `src/models/__init__.py`、`src/train.py`、`src/predict.py`：接入 TCN，并让训练/预测识别 TCN 为 sequence model。

注意：当前 `ALSTM`/GRU 还没有加入 LayerNorm。结构为：

```text
feature_proj: Linear -> Tanh
GRU
optional temporal attention
Linear head
```

因此后续仍可以尝试 `LayerNorm(input)` 或 `LayerNorm(hidden)`，这属于未完成的改进方向。

## 数据与评估口径

主目标：

```text
label_5d__cs_rank
```

主要指标：

- IC mean：模型分数与 `label_5d__cs_rank` 的逐日 Spearman 相关均值。
- ICIR：IC mean / IC std，用于衡量信号稳定性。
- MSE：辅助参考。
- Top-N 回测：辅助参考，当前策略依赖较强，不能作为唯一模型选择依据。

当前 Top-N 回测策略：

```text
n_hold = 20
k_rotate = 5
step_days = 5
transaction_cost_bps = 5
return_col = label_5d
```

GRU/TCN 评估时已修正 warmup：评估 split 内出分，但序列窗口允许使用 split 之前的历史特征，避免序列模型少掉 split 前 59 个交易日导致指标不可比。

## E0/E1 基线

E0 是 MLP baseline：

```text
112 features -> Linear(112, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 1)
```

E1 是 `src/models/lrk/alstm.py` 中的 ALSTM-GRU：

```text
112 features sequence -> Linear+Tanh -> GRU -> temporal attention -> score
```

全量测试集最终对比：

| 模型 | samples | MSE | IC | ICIR | 回测收益 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP baseline | 1,280,375 | 0.331642 | 0.082496 | 0.600163 | 2.806080 | 2.428451 | -0.133465 |
| layer1 GRU | 1,019,298 | 0.327920 | 0.085123 | 0.648694 | 0.358659 | 1.248049 | -0.194366 |
| layer1 core features | 1,019,298 | 0.327661 | 0.069877 | 0.492486 | 0.364097 | 1.102746 | -0.167735 |

结论：

- 如果以 IC/ICIR 为主，`layer1 GRU + 112 features` 是当前最好的 GRU 候选。
- MLP 回测收益很高，但 IC/ICIR 低于 `layer1 GRU`，说明当前简化回测策略可能更偏向 MLP 分数分布，不能只看回测收益。

## GRU 结构消融

pilot 设置：

```text
train: 2023
valid: 2024
loss: SmoothL1
early stopping: patience=2
```

| 实验 | 特征数 | hidden | layers | attention | best epoch | valid loss | IC | ICIR | 回测收益 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base_attn_h128_l2 | 112 | 128 | 2 | yes | 1 | 0.164106 | 0.082912 | 0.399819 | 0.253217 | 0.643299 | -0.413522 |
| hidden64 | 112 | 64 | 2 | yes | 2 | 0.163761 | 0.083066 | 0.384486 | 0.266423 | 0.692241 | -0.395093 |
| layer1 | 112 | 128 | 1 | yes | 1 | 0.162710 | 0.092944 | 0.468959 | 0.480024 | 0.922782 | -0.380244 |
| no_attention | 112 | 128 | 2 | no | 1 | 0.163257 | 0.087882 | 0.431210 | 0.147635 | 0.528733 | -0.419579 |
| core_features | 52 | 128 | 2 | yes | 2 | 0.163933 | 0.084944 | 0.441277 | 0.331567 | 0.782914 | -0.362593 |

结论：

- 1 层 GRU 明显优于 2 层 GRU，说明当前 2 层结构较容易过拟合。
- 去掉 attention 后 IC 略有改善但回测变差，attention 对头部排序可能仍有帮助。
- 手工核心特征子集在 pilot 稳定性尚可，但全量后 IC/ICIR 不如全特征。

## 全量 GRU 候选比较

全量 train/valid：

| 实验 | 特征数 | layers | best epoch | MSE | IC | ICIR | 回测收益 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP baseline | 112 | - | 3 | 0.329789 | 0.097258 | 0.494418 | -0.201582 | -0.047620 | -0.496278 |
| layer1 all features | 112 | 1 | 1 | 0.327678 | 0.095154 | 0.511638 | 0.126012 | 0.488961 | -0.373277 |
| core features 2-layer | 52 | 2 | 1 | 0.327513 | 0.084451 | 0.439894 | -0.031154 | 0.169488 | -0.343132 |
| layer1 core features | 52 | 1 | 1 | 0.327282 | 0.081162 | 0.416585 | 0.332457 | 0.794470 | -0.290193 |

结论：

- 如果以 IC/ICIR 为主，`layer1 all features` 仍是当前 GRU 主候选。
- `layer1 core features` 回撤更低、回测更好，但 IC/ICIR 下降明显，不适合作为主模型。

## Loss 对照

模型固定为：

```text
layer1 GRU
hidden=128
lookback=60
112 features
attention=true
```

pilot valid 结果：

| 实验 | loss | lambda | best epoch | MSE | IC | ICIR | 回测收益 | Sharpe | 最大回撤 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loss_layer1_smooth_l1 | SmoothL1 | - | 1 | 0.328197 | 0.092944 | 0.468959 | 0.480024 | 0.922782 | -0.380244 |
| loss_layer1_smooth_l1_corr | SmoothL1 + batch Corr | 0.05 | 1 | 0.329255 | 0.089665 | 0.447459 | 0.614289 | 1.064504 | -0.380975 |
| loss_layer1_mse_corr | MSE + batch Corr | 0.05 | 1 | 0.328137 | 0.089582 | 0.447845 | 0.517058 | 0.957968 | -0.388408 |

结论：

- 以 IC/ICIR 为主，SmoothL1 仍然最好。
- batch CorrLoss 提高了当前回测收益，但降低了 IC/ICIR。
- 主要原因可能是 batch 不是按交易日截面构造，batch correlation 不等价于 daily IC。
- 若要继续优化 IC，应考虑按日期构造 batch，再做 daily CorrLoss。

## Lookback 对照

模型固定为 `layer1 GRU + SmoothL1 + attention`。

| 实验 | lookback | samples | best epoch | MSE | IC | ICIR | 回测收益 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lb20 | 20 | 861,467 | 1 | 0.331110 | 0.083108 | 0.447625 | 0.487723 | 0.919401 | -0.358659 |
| lb30 | 30 | 831,465 | 1 | 0.330489 | 0.083941 | 0.465252 | 0.048113 | 0.369648 | -0.340580 |
| lb60 | 60 | 766,867 | 1 | 0.328197 | 0.092944 | 0.468959 | 0.480024 | 0.922782 | -0.380244 |

结论：

- IC 看，lookback=60 最好。
- ICIR 看，lookback=60 与 30 接近，但 60 仍略高。
- 当前不建议把 20/30 扩到全量。

## TCN 实验

新增 TCN 模型：

```text
Input [B, T, F]
-> causal dilated Conv1d residual blocks
-> last timestep or attention pooling
-> Linear head
```

参数选择参考 TCN 常见设置：

- causal convolution，避免未来信息。
- dilation 使用 `1,2,4,8`。
- kernel size 使用 3。
- 先用较小 channels，避免 GRU 已观察到的快速过拟合。

pilot 结果：

| 实验 | 模型 | seq | best epoch | MSE | IC | ICIR | 回测收益 | Sharpe | 最大回撤 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP pilot | MLP | - | - | 0.331217 | 0.090974 | 0.458085 | -0.090438 | 0.124551 | -0.444695 |
| GRU layer1 lb60 | GRU | 60 | 1 | 0.328197 | 0.092944 | 0.468959 | 0.480024 | 0.922782 | -0.380244 |
| GRU layer1 lb30 | GRU | 30 | 1 | 0.330489 | 0.083941 | 0.465252 | 0.048113 | 0.369648 | -0.340580 |
| TCN lb20 c64 | TCN | 20 | 1 | 0.337109 | 0.052631 | 0.262179 | 0.349237 | 0.766966 | -0.398329 |
| TCN lb30 c32 | TCN | 30 | 3 | 0.332273 | 0.074995 | 0.457244 | 0.063735 | 0.391533 | -0.376803 |
| TCN lb30 c32 attn | TCN | 30 | 3 | 0.330684 | 0.077673 | 0.417171 | 0.175830 | 0.560173 | -0.388820 |

结论：

- 当前 TCN 未超过 GRU layer1，也未超过 MLP。
- 轻量 TCN-30 的 ICIR 接近 baseline，但 IC 明显偏低。
- attention pooling 对 TCN 的 IC 有小幅提升，但 ICIR 下降。
- 暂不建议扩 TCN 到全量。TCN 可以作为报告中的纯深度卷积时序模型尝试，但不是主模型候选。

## 阶段性建议

当前最稳的深度模型候选：

```text
layer1 GRU + 112 features + lookback=60 + attention + SmoothL1 + early stopping
```

后续最值得尝试的方向：

1. 在 GRU 中加入 LayerNorm：
   - `LayerNorm(input features)` 后再进入 `feature_proj`。
   - 或对 GRU 输出 `h` 做 `LayerNorm(hidden)` 再 attention/head。
2. 做按日期构造 batch 的 daily CorrLoss，使训练目标更贴近 IC。
3. 做 LightGBM baseline 和 LightGBM/GRU rank ensemble。
4. 用 LightGBM importance 做 Top-K 特征选择，再训练 GRU，替代当前手工 core feature subset。

暂不建议继续扩大 TCN 搜索，除非先重构 TCN 输入归一化或 batch 组织方式。
