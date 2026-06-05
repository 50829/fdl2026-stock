# GRU 模型实验方案

本文档记录本项目中 GRU 方向的模型架构、实验流程和建议目录结构。目标是基于已经处理好的 A 股日频截面数据，训练一个可用于每日股票打分、IC 评估和 Top-N 回测的时序模型。

## 任务定义

本项目不建模为单只股票价格预测，而建模为每日截面排序任务：

```text
输入：某只股票过去若干个交易日的 112 维特征序列
输出：该股票在当前 trade_date 的预测 score
目标：score 在每日截面上尽量与未来收益排序一致
```

默认训练目标：

```text
label_5d__cs_rank
```

原因：

- `label_5d__cs_rank` 是未来 5 日收益的每日截面排名，适合选股排序。
- 相比 1 日收益，5 日收益噪声略低，更适合低频调仓。
- 与后续 IC、Top-N 回测的评价口径一致。

数据来源：

```text
data/processed/features.parquet
data/processed/labels.parquet
data/processed/feature_meta.json
data/processed/splits.json
```

训练时必须使用：

```python
feature_cols = meta["feature_columns"]
```

不要自行按字母序排列特征列。

## 业界和学界常见思路

GRU 在股票预测中通常作为时间编码器使用。常见结构包括：

1. Vanilla GRU

   直接输入过去若干交易日的多因子序列，用最后一个隐状态预测股票分数。

2. GRU + Temporal Attention

   GRU 负责提取时间状态，注意力层负责选择关键时间步。适合金融时间序列中“近期冲击”和“中期趋势”重要性不固定的情况。

3. CNN/TCN + GRU

   CNN 或 TCN 先提取局部时间模式，GRU 再建模较长依赖。高频订单簿模型中类似思路较常见。

4. Graph + GRU

   GRU 负责单只股票时间序列，GCN/GAT 负责股票间关系，例如行业、相关性、供应链等。效果潜力较大，但实现复杂度较高，不建议作为第一版主线。

本项目第一版建议采用 `GRU + Temporal Attention`，它比裸 GRU 更有表达力，也足够容易实现和解释。

## 推荐模型：AttentiveGRU

输入张量：

```text
X: [batch_size, lookback, n_features]
默认回看窗口 = 60
默认 n_features = 112
```

模型结构：

```text
Input [B, T, F]
  |
  |-- LayerNorm(F)
  |-- Linear(F -> d_model)
  |-- GELU
  |-- Dropout
  |
GRU(input_size=d_model, hidden_size=hidden, num_layers=2)
  |
  |-- temporal outputs: [B, T, H]
  |-- 最后隐状态:      [B, H]
  |
Temporal Attention
  |
  |-- score_t = v^T tanh(W h_t)
  |-- alpha = softmax(score_t)
  |-- context = sum(alpha_t * h_t)
  |
concat(context, last_hidden) -> [B, 2H]
  |
MLP Head
  |
score [B]
```

推荐超参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `lookback` | 60 | 使用过去 60 个交易日 |
| `d_model` | 64 | 输入特征投影维度 |
| `hidden_size` | 128 | GRU 隐状态维度 |
| `num_layers` | 2 | GRU 层数 |
| `dropout` | 0.1 | 防止过拟合 |
| `batch_size` | 512 或 1024 | 根据显存调整 |
| `lr` | 1e-3 | AdamW 初始学习率 |
| `weight_decay` | 1e-4 | L2 正则 |
| `epochs` | 20 | 配合早停 |

## 对照模型

为了报告完整，至少保留三类基线：

| 模型 | 输入 | 目的 |
| --- | --- | --- |
| MLP | 当日 112 维特征 | 检验时序信息是否有增益 |
| VanillaGRU | 60 日特征序列 | GRU 基线 |
| AttentiveGRU | 60 日特征序列 | 主模型 |

如果时间充足，可以增加：

| 模型 | 说明 |
| --- | --- |
| BiGRU | 双向 GRU，只在离线预测中使用，不能用于在线逐日滚动解释时要谨慎 |
| CNN-GRU | 1D 卷积提取局部时间模式后接 GRU |
| MultiTaskGRU | 同时预测 `label_1d__cs_rank` 和 `label_5d__cs_rank` |

## 损失函数

第一版推荐使用 Huber 损失：

```text
loss = SmoothL1Loss(score, label_5d__cs_rank)
```

原因：

- rank label 范围约为 `[-1, 1]`。
- Huber 损失比 MSE 对异常样本更稳。
- 实现简单，训练稳定。

第二版可以增加每日截面相关性损失：

```text
loss = SmoothL1Loss + lambda * CorrLoss
CorrLoss = 1 - PearsonCorr(score, label)
```

注意：`CorrLoss` 应按 `trade_date` 分组计算，工程复杂度更高。第一版可以只训练 Huber 损失，在验证阶段计算 IC。

## 数据集构造

处理步骤：

1. 读取 `features.parquet`、`labels.parquet`。
2. 读取 `feature_meta.json`，取得固定特征列顺序。
3. 按 `["trade_date", "ts_code"]` inner merge。
4. 按 `splits.json` 做时间切分。
5. 对每只股票按 `trade_date` 排序，构造滑动窗口样本。

样本定义：

```text
样本 i:
  X_i = stock s 在 [t-lookback+1, t] 的特征序列
  y_i = stock s 在 t 日的 label_5d__cs_rank
  meta_i = (trade_date=t, ts_code=s)
```

需要过滤：

- 窗口长度不足 `lookback` 的样本。
- 标签缺失的样本。
- 窗口中存在非有限值的样本。如果 processed features 已经填充为 0，通常不会出现。

时间切分：

```text
train: 20160101 - 20231231
valid: 20240101 - 20241231
test:  20250101 - 20260518
```

禁止随机切分日期。

## 评估指标

训练阶段记录：

- 训练损失
- 验证损失
- learning rate
- 最佳 epoch

验证和测试阶段记录：

- Daily Spearman IC
- IC mean
- IC std
- ICIR = IC mean / IC std
- Top-N 等权组合收益
- 年化收益
- 夏普比率
- 最大回撤
- 平均换手率

推荐回测组合：

| 组合 | 说明 |
| --- | --- |
| Top 20 | 集中组合，收益弹性大 |
| Top 50 | 更稳健 |
| Top 100 | 检查模型排序整体质量 |

调仓频率建议：

```text
每 5 个交易日调仓一次
```

这与 `label_5d` 的持有周期一致。

## 消融实验设计

建议实验表：

| 实验 | 改动 | 目的 |
| --- | --- | --- |
| E00 | MLP 基线 | 检验非时序基线 |
| E01 | VanillaGRU，回看窗口=60 | GRU 基线 |
| E02 | AttentiveGRU，回看窗口=60 | 主模型 |
| E03 | AttentiveGRU，回看窗口=20 | 检查短窗口 |
| E04 | AttentiveGRU，回看窗口=40 | 检查中窗口 |
| E05 | AttentiveGRU，隐状态维度=64 | 检查模型容量 |
| E06 | AttentiveGRU，隐状态维度=256 | 检查过拟合风险 |
| E07 | 目标改为 `label_1d__cs_rank` | 短周期对照 |
| E08 | 去掉 moneyflow 特征组 | 特征组消融 |
| E09 | 去掉 fundamental_size 特征组 | 特征组消融 |

最终报告建议主表包含：

```text
model, lookback, hidden, target, valid_ic, test_ic, icir, annual_return, sharpe, max_drawdown
```

## 建议目录结构

当前 `src/model_experiments/` 可以作为 GRU 方向的独立实现目录。建议结构如下：

```text
src/model_experiments/
  gru.md                 # 本设计文档
  __init__.py
  config.yaml            # GRU 默认实验配置
  dataset.py             # SequenceDataset / DataModule
  model.py               # MLP, VanillaGRU, AttentiveGRU
  losses.py              # Huber 损失，可选 CorrLoss
  train.py               # 训练入口
  predict.py             # 生成预测分数
  evaluate.py            # IC、ICIR、loss 评估
  backtest.py            # Top-N 回测入口，可复用 src/backtest
  utils.py               # seed、device、日志、checkpoint 工具
```

输出目录建议：

```text
outputs/models/20260530_013040__sequence_baselines/gru/
  E00_mlp/
    config.yaml
    best.pt
    train_log.csv
    valid_metrics.json
    test_metrics.json
  E01_vanilla_gru/
  E02_attentive_gru/

outputs/predictions/sequence_baselines/gru/
  E02_valid_pred.parquet
  E02_test_pred.parquet
  E02_latest_pred.csv

outputs/figures/sequence_baselines/gru/
  E02_ic_curve.png
  E02_equity_curve.png
  ablation_summary.png
```

日志建议：

```text
logs/sequence_baselines/gru/
  E02_attentive_gru.log
```

## 配置文件草案

建议 `src/model_experiments/config.yaml`：

```yaml
seed: 2026

data:
  processed_dir: data/processed
  feature_meta: data/processed/feature_meta.json
  features: data/processed/features.parquet
  labels: data/processed/labels.parquet
  splits: data/processed/splits.json
  target: label_5d__cs_rank
  raw_return_col: label_5d
  feature_mode: default
  selected_groups: null

model:
  name: attentive_gru
  lookback: 60
  d_model: 64
  hidden_size: 128
  num_layers: 2
  dropout: 0.1
  bidirectional: false

train:
  batch_size: 512
  epochs: 20
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  num_workers: 4
  early_stop_patience: 5
  loss: smooth_l1

eval:
  score_col: score
  ic_method: spearman
  topn: [20, 50, 100]
  rebalance_days: 5
  transaction_cost_bps: 5

output:
  run_name: E02_attentive_gru
  model_dir: outputs/models/20260530_013040__sequence_baselines/gru
  prediction_dir: outputs/predictions/sequence_baselines/gru
  figure_dir: outputs/figures/sequence_baselines/gru
  log_dir: logs/sequence_baselines/gru
```

## 实现顺序

建议按以下顺序实现，降低调试风险：

1. 实现 `dataset.py`，确认能构造 `[B, T, F]` 样本。
2. 实现 `MLP`，只用窗口最后一天特征训练，跑通完整训练和评估。
3. 实现 `VanillaGRU`。
4. 实现 `AttentiveGRU`。
5. 实现 `evaluate.py`，输出 daily IC、ICIR。
6. 实现 `predict.py`，保存 valid/test/latest 分数。
7. 实现 Top-N 回测和权益曲线。
8. 做消融实验，生成报告表格和图。

## 报告写作要点

报告中可以强调：

- 本任务是每日截面排序，不是单股票价格点预测。
- 特征只使用当日盘后及历史信息，标签从 `T+1` 买入开始，避免未来信息泄露。
- 训练、验证、测试严格按时间切分。
- GRU 用于捕捉单只股票的时间动态，注意力层用于学习不同历史交易日的重要性。
- 最终以 IC 和模拟交易指标评价模型，而不是只看损失。

# 下一步实验
1. 先做 GRU + LayerNorm 对照实验
    当前最好的模型是 layer1 + attention + SmoothL1 +
    回看窗口=60，但还没加 LayerNorm。
    我会加几组最小改动实验：
  - layer1_baseline：当前最优 GRU，不变
  - layer1_input_layernorm：对输入特征做 LayerNorm
  - layer1_hidden_layernorm：对 GRU 输出隐状态做
    LayerNorm
  - layer1_input_hidden_layernorm：输入和隐状态都
    做 LayerNorm

    先跑 pilot，看 IC/ICIR 趋势；如果提升稳定，再全量
    训练。
2. 保持实验变量干净
    这次只改 LayerNorm，不同时改损失函数、回看窗口、
    feature set。否则看不清楚到底是谁带来的变化。
3. 指标主要看 IC / ICIR
    排序优先级我建议是：
  - 第一优先：valid/test IC
  - 第二优先：valid/test ICIR
  - 第三优先：MSE
  - 第四优先：当前 Top-N 回测

    因为现在回测策略比较简单，不能直接代表模型真实上
    限。
4. 如果 LayerNorm 有提升，再做全量
    不是 pilot 最好就直接信，而是看它是不是：
    - IC 提升
    - ICIR 不下降
    - valid 和 test 方向一致
    - 没有明显只靠某一年表现拉高
5. LayerNorm 后再考虑两个方向
    如果 LayerNorm 有用：继续做 GRU 稳定性改进，比如
    dropout、weight decay、hidden size 小范围调参。
    如果 LayerNorm 没用：转向 LightGBM/融合方案，先做
    一个传统强基线，再考虑神经网络输出作为额外特
    征。

下一步优先改 src/models/sequence/alstm.py，给当前 GRU/ALSTM
增加可配置的 LayerNorm，然后在 src/model_experiments/run_ablation.py
里加这几组实验，跑 pilot 并生成对照表。
