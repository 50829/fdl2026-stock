# 02 GRU LayerNorm 实验

本文档记录 2026-05-30 进行的 GRU LayerNorm 对照实验。编号 `02` 用于和已有综合实验记录区分。

## 实验目的

前面 GRU 消融实验中，当前最优结构是：

```text
layer1 GRU + 112 个特征 + 回看窗口=60 + attention + SmoothL1 + 早停
```

由于 2 层 GRU、TCN、缩短回看窗口、手工核心特征都没有超过该基线，本轮实验只检查一个问题：

```text
在当前 layer1 GRU 上加入 LayerNorm，是否能提高 IC / ICIR？
```

## 代码改动

涉及文件：

- `src/models/sequence/alstm.py`
- `src/models/__init__.py`
- `src/model_experiments/run_ablation.py`

新增模型配置开关：

```yaml
input_layernorm: true
hidden_layernorm: true
```

含义：

```text
input_layernorm=true:
  LayerNorm(输入特征) -> Linear -> Tanh -> GRU -> attention/head

hidden_layernorm=true:
  输入特征 -> Linear -> Tanh -> GRU -> LayerNorm(隐状态) -> attention/head
```

## 实验设置

数据和训练设置沿用 pilot 口径：

```text
processed_dir: data/processed_pilot
train: 2023
valid: 2024
target: label_5d__cs_rank
loss: SmoothL1
batch_size: 4096
epochs: 8
早停耐心轮数：2
主要指标：IC、ICIR
```

固定模型参数：

```text
model: ALSTM/GRU
num_layers: 1
hidden_size: 128
seq_len: 60
use_attention: true
features: 全部 112 个特征
```

输出目录：

```text
outputs/models/20260530_154249__layernorm_ablation/
```

## 实验结果

| 实验 | 输入 LN | 隐状态 LN | 最佳 epoch | valid 损失 | MSE | IC | ICIR | 回测收益 | Sharpe | 最大回撤 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer1` | no | no | 1 | 0.162710 | 0.328197 | 0.092944 | 0.468959 | 0.480024 | 0.922782 | -0.380244 |
| `layer1_input_layernorm` | yes | no | 1 | 0.162855 | 0.329607 | 0.087430 | 0.447927 | 0.566943 | 1.034457 | -0.387209 |
| `layer1_hidden_layernorm` | no | yes | 3 | 0.165485 | 0.332257 | 0.070258 | 0.451512 | 0.453018 | 1.430329 | -0.114050 |
| `layer1_input_hidden_layernorm` | yes | yes | 4 | 0.166509 | 0.336350 | 0.065219 | 0.426890 | 0.436998 | 0.927140 | -0.354793 |

## 结果解读

以 IC / ICIR 为主，LayerNorm 没有带来提升：

- `layer1` 基线仍然最好，IC = 0.092944，ICIR = 0.468959。
- `input_layernorm` 的回测收益更高，但 IC 和 ICIR 都下降，说明它没有增强整体截面排序信号。
- `hidden_layernorm` 的最佳 epoch 从 1 推迟到 3，但 IC 明显下降到 0.070258。
- 输入和隐状态同时加 LayerNorm 最差，IC 下降到 0.065219。

这说明当前问题不太像是 GRU 隐状态数值不稳定导致的。LayerNorm 可能改变了已有工程特征的尺度结构，反而削弱了特征中的截面排序信息。

## 结论

本轮结论：

```text
不建议把 LayerNorm GRU 扩到全量训练。
```

当前 GRU 主候选仍然是：

```text
layer1 GRU + 112 个特征 + 回看窗口=60 + attention + SmoothL1 + 早停
```

GRU 结构层面的主要实验已经基本跑完：

- 隐状态维度：试过 128 和 64。
- 层数：1 层优于 2 层。
- 注意力：保留 attention 更稳。
- 特征子集：核心特征不如全特征。
- 损失函数：SmoothL1 的 IC/ICIR 最好。
- 回看窗口：60 优于 20/30。
- LayerNorm：不如原始 layer1。

## 下一步建议

继续硬调 GRU 结构的收益预计不高。更值得做的是：

1. 实现按交易日构造 batch 的每日 CorrLoss，让训练目标更贴近每日 IC。
2. 跑 LightGBM 基线，得到一个强传统机器学习对照。
3. 做 GRU + LightGBM 融合，例如 rank ensemble 或把 GRU 输出作为 LightGBM 的额外特征。
4. 用 LightGBM importance 选 Top-K 特征，再训练 GRU，替代当前手工核心特征。
