# 收益率目标模型重训方案

本文档记录下一阶段如果改为直接拟合未来收益率 `label_5d`，应该如何重新训练模型、训练哪些模型、如何评估，以及它和当前最终 rank 模型的关系。

## 1. 背景

当前最终模型是：

```text
Residual-rank deep_ln, alpha=1.5
```

当前模型的核心监督目标是：

```text
label_5d__cs_rank
```

也就是未来 5 日收益率在当日股票截面中的相对排序。最终输出 `pred` 是排序分数，不是收益率数值。

如果后续希望模型输出更接近“预测未来收益率”的含义，可以开一条独立实验线，把目标改为：

```text
target = label_5d
```

此时模型输出可以解释为：

```text
预测未来 5 日收益率
```

但即使输出是收益率，策略侧仍然可以按预测收益率排序选股。

## 2. 重要判断

可以用集成学习拟合收益率。

LightGBM / XGBoost 本身就是回归模型，完全可以把 target 从 `label_5d__cs_rank` 改成 `label_5d`。

但需要注意：

- 可以拟合收益率，不代表收益率目标一定更适合选股。
- 原始收益率噪声更大，极端值更多。
- MSE 容易被少数异常收益率主导。
- 策略通常更关心股票之间的相对排序，而不是精确预测未来涨跌幅。

因此收益率模型应作为补充实验线，而不是直接替换当前最终 rank 模型。

## 3. 不建议直接改 residual-rank

当前最终 residual-rank 模型公式是：

```text
pred = pred_lgb + alpha * residual_rank_pred
```

其中 `residual_rank_pred` 是 MLP 预测的残差排序修正项，不是收益率单位。

如果直接把这个结构用于收益率输出，会出现单位不一致：

```text
收益率预测 + rank 修正
```

这在解释上不够严谨。

如果目标是输出可解释为收益率的数值，更适合使用 residual-value 架构。

## 4. 推荐主线：Residual-value 收益率模型

收益率版本的 residual-value 思路：

```text
base_pred = LightGBM(label_5d)
residual = label_5d - base_pred
MLP 学 residual
final_pred = base_pred + alpha * residual_pred
```

含义：

- LightGBM 先预测未来 5 日收益率。
- MLP 再预测 LightGBM 没解释掉的收益率残差。
- 最终输出 `final_pred` 可以解释为预测未来 5 日收益率。

这条线最接近当前“树模型 + 深度学习融合”的最终思路。

## 5. 计划训练哪些模型

### 5.1 第一组：纯树模型收益率 baseline

先训练：

```text
LightGBM top40 -> label_5d
XGBoost top40 -> label_5d
```

目的：

- 验证 `label_5d` 是否可以直接拟合。
- 得到收益率目标下的强 baseline。
- 为后续 stacking / residual-value 提供 base prediction。
- 对比 raw return 目标和 rank target 的 IC/ICIR、MSE、回测收益。

输出：

```text
pred = 预测未来 5 日收益率
```

### 5.2 第二组：收益率 residual-value MLP

训练两版：

```text
LGB top40 + MLP top40 residual
LGB top40 + MLP meta residual
```

第一版输入：

```text
top40 原始因子 + base_pred
```

第二版输入：

```text
pred_lgb
pred_xgb
rank_lgb
rank_xgb
pred_mean
rank_mean
pred_diff
rank_diff
```

目的：

- 看深度学习网络能否补充树模型对收益率的残差。
- 和当前 residual-rank 方法对比。
- 形成一个输出单位更接近收益率的融合模型。

### 5.3 第三组：收益率 stacking

训练：

```text
Stacking Ridge(label_5d)
Stacking MLP deep_ln(label_5d)
```

输入：

```text
pred_lgb
pred_xgb
rank_lgb
rank_xgb
pred_mean
rank_mean
pred_diff
rank_diff
```

目的：

- 看直接融合 LGB/XGB 的收益率预测是否更稳定。
- 作为 residual-value 的对照。
- 判断 MLP 是否比线性 Ridge 更有增益。

## 6. 暂不优先训练的模型

暂时不建议第一轮就重新训练 GRU 拟合 `label_5d`。

原因：

- GRU 在 rank 目标上已经明显弱于树模型。
- `label_5d` 比 `label_5d__cs_rank` 噪声更大。
- 序列模型拟合 raw return 更容易不稳定。
- 先用树模型和轻量 MLP 判断收益率目标是否可行，成本更低。

如果收益率目标在树模型和 MLP 融合上显示出明显潜力，再考虑：

```text
GRU -> label_5d
GRU residual -> label_5d residual
```

## 7. 推荐训练顺序

建议按以下顺序推进：

1. `LightGBM top40 -> label_5d`
2. `XGBoost top40 -> label_5d`
3. `Stacking Ridge -> label_5d`
4. `Stacking MLP deep_ln -> label_5d`
5. `LGB residual-value MLP -> label_5d`
6. 如果收益率目标明显有希望，再考虑 GRU / 序列模型收益率版本。

这样做的好处：

- 第一阶段快速判断 raw return 目标是否有价值。
- 第二阶段再看融合模型是否能提升。
- 避免一开始就跑成本最高、最不稳定的深度序列模型。

## 8. 评估指标

收益率目标不能只看 MSE。

建议同时评估：

```text
MSE
MAE
IC with label_5d
ICIR with label_5d
IC with label_5d__cs_rank
ICIR with label_5d__cs_rank
top-k return
top-k Sharpe
top-k maxDD
rolling return
rolling Sharpe
rolling maxDD
```

原因：

- MSE/MAE 衡量收益率数值拟合。
- IC/ICIR 衡量排序能力。
- top-k / rolling 回测衡量策略可用性。
- 即使收益率数值不准，只要排序有效，策略仍然可用。

## 9. 训练命令草案

### 9.1 LightGBM 收益率 baseline

```bash
python -m src.model_experiments.run_gbdt \
  --model lightgbm \
  --processed-dir data/processed \
  --out-root outputs/models/return_gbdt_top40 \
  --target label_5d \
  --raw-return-col label_5d \
  --daily-return-col label_1d \
  --feature-list outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt \
  --num-threads 16 \
  --num-boost-round 800 \
  --early-stopping-rounds 80 \
  --log-period 200
```

### 9.2 XGBoost 收益率 baseline

```bash
python -m src.model_experiments.run_gbdt \
  --model xgboost \
  --processed-dir data/processed \
  --out-root outputs/models/return_gbdt_top40 \
  --target label_5d \
  --raw-return-col label_5d \
  --daily-return-col label_1d \
  --feature-list outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt \
  --num-threads 16 \
  --num-boost-round 800 \
  --early-stopping-rounds 80 \
  --log-period 200
```

### 9.3 Residual-value MLP 收益率模型

```bash
python -m src.model_experiments.run_residual_mlp \
  --mode oof \
  --processed-dir data/processed \
  --out-root outputs/models/return_residual_mlp_deep_ln \
  --target label_5d \
  --raw-return-col label_5d \
  --daily-return-col label_1d \
  --base-feature-list outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt \
  --mlp-feature-list outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt \
  --mlp-arch deep_ln \
  --mlp-loss smooth_l1 \
  --mlp-hidden 128 \
  --mlp-dropout 0.1 \
  --mlp-epochs 8 \
  --patience 2 \
  --mlp-batch-size 8192 \
  --num-boost-round 800 \
  --early-stopping-rounds 80 \
  --num-threads 16 \
  --log-period 200 \
  --alpha-grid 0 0.25 0.5 0.75 1.0
```

## 10. 实操注意事项

之前尝试直接启动过一版全量 OOF 收益率 residual-value 训练：

```text
outputs/models/return_residual_mlp_deep_ln
```

但该任务比 rank 目标明显更慢，几分钟内没有写出中间产物，因此已经中止。

后续正式跑之前建议：

- 先单独跑 LightGBM / XGBoost 收益率 baseline。
- 确认 raw return target 的训练耗时和结果。
- 对 `label_5d` 做分布检查。
- 必要时对 `label_5d` 做 winsorize / clip，再训练收益率模型。

建议不要直接用 pilot 跑当前 OOF residual 脚本，因为脚本内部固定使用 `2021/2022/2023` 做 OOF，而 pilot 数据时间覆盖不完整。

## 11. 如何决定是否采用收益率模型

如果收益率模型满足：

- MSE/MAE 表现合理。
- IC/ICIR 不低于当前 rank 模型。
- rolling return 或风险指标更好。
- 输出能稳定解释为预测收益率。

则可以考虑让收益率模型成为策略侧新主模型。

如果收益率模型：

- IC/ICIR 不如当前 rank 模型。
- 回测收益不稳定。
- MSE 被极端值主导。

则保留当前最终模型：

```text
Residual-rank deep_ln, alpha=1.5
```

收益率模型只作为补充实验和报告讨论。

## 12. 当前建议

当前不建议直接替换最终模型。

更稳妥的方案是：

1. 保留 `Residual-rank deep_ln, alpha=1.5` 作为交接主模型。
2. 新开收益率目标实验线。
3. 先跑树模型收益率 baseline。
4. 再跑收益率 residual-value MLP。
5. 最后用 IC/ICIR 和策略回测共同决定是否切换。
