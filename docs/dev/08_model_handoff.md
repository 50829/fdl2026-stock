# 模型使用交接文档

本文档面向后续策略和回测流程，说明当前有哪些模型、最终推荐模型怎么用、输入输出字段是什么，以及预测结果如何接入策略。

## 1. 最终推荐模型

最终推荐使用：

```text
Residual-rank deep_ln, alpha=1.5
```

它是一个“树模型 + 深度学习残差修正”的融合模型：

```text
final_pred = pred_lgb + 1.5 * residual_rank_pred
```

其中：

- `pred_lgb` 是 LightGBM top40 的预测分数。
- `residual_rank_pred` 是 MLP 对 LightGBM 残差排序的修正项。
- `alpha=1.5` 是最终选择的残差修正权重。
- `final_pred` / `pred` 是策略侧应该使用的最终打分。

模型文件：

```text
outputs/models/20260531_121653__fusion_rank_tune/alpha_ext_h128_d010_wd1e4/residual_rank_mlp/residual_rank_mlp.pt
```

最终预测交接文件：

```text
outputs/models/20260531_162154__final_model_handoff/valid/valid_pred.parquet
outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet
outputs/models/20260531_162154__final_model_handoff/summary.json
```

策略端优先使用 `outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet`。

## 2. 数据和时间范围

当前最终模型不是 pilot 版本，使用的是正式 `data/processed` 口径。

训练和评估切分：

| 数据集 | 日期范围 | 用途 |
| --- | --- | --- |
| residual-rank MLP train | `20210101-20231231` | 使用 OOF 的 LGB/XGB meta prediction 训练残差排序网络 |
| valid | `20240102-20241231` | 选 alpha 和调参 |
| test | `20250102-20260508` | 最终测试和策略交接 |

说明：

- 原始 test 配置到 `20260518`，但 `label_5d` / `label_5d__cs_rank` 需要未来 5 日收益，所以最终可评估预测文件截至 `20260508`。
- 如果策略侧只需要打分，不需要标签，未来可以生成更靠后的 live prediction；但当前交接文件包含标签，因此遵循可回测日期范围。

## 3. 最终预测文件格式

`valid_pred.parquet` 和 `test_pred.parquet` 字段一致。

| 字段 | 类型 | 含义 | 策略是否需要 |
| --- | --- | --- | --- |
| `trade_date` | str | 交易日期 | 需要 |
| `ts_code` | str | 股票代码 | 需要 |
| `pred` | float32 | 最终模型打分，等于 `final_pred` | 需要 |
| `final_pred` | float32 | 最终模型打分，保留解释字段 | 可选 |
| `pred_lgb` | float32 | LightGBM top40 原始预测 | 可选 |
| `pred_xgb` | float32 | XGBoost top40 原始预测 | 可选 |
| `residual_rank_pred` | float32 | MLP 预测的残差排序修正项 | 可选 |
| `alpha` | float32 | 残差权重，当前固定为 `1.5` | 可选 |
| `label_5d__cs_rank` | float64 | 5 日未来收益的截面 rank 标签 | 只用于回测评估 |
| `label_5d` | float32 | 未来 5 日原始收益 | 只用于回测评估 |
| `label_1d` | float32 | 未来 1 日原始收益 | 只用于回测评估 |

策略侧最小输入只需要：

```text
trade_date, ts_code, pred
```

建议策略使用方式：

```python
import pandas as pd

pred = pd.read_parquet("outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet")

# 每天按 pred 从高到低排序，选前 K 只股票。
daily_list = (
    pred.sort_values(["trade_date", "pred"], ascending=[True, False])
        .groupby("trade_date")
        .head(20)
        [["trade_date", "ts_code", "pred"]]
)
```

注意：

- `pred` 越大，模型越看好。
- `pred` 不是交易指令，也不是未来收益率；它只是当日股票截面的排序分数。
- 策略侧不要使用 `label_5d__cs_rank`、`label_5d`、`label_1d` 做选股；这些字段只用于回测后验评估。
- 如果做 live 策略，输出文件里不应该包含 label 字段。

策略侧最直接的转换方式：

```text
模型输出 pred -> 按 trade_date 分组 -> 每天按 pred 降序排序 -> 选 top K -> 生成买入列表
```

如果采用“每天调仓一次、单笔持有多天”的策略：

```text
target_positions = 20
hold_days = 5
每天新买入数量 = target_positions / hold_days = 4
```

即每天从 `pred` 最高的股票里选 4 只买入，每笔持有 5 天，到期卖出。组合中同时持有过去 5 个交易日买入的股票。

## 4. 最终模型输入

最终 residual-rank MLP 不直接吃 112 个原始因子，而是吃 LGB/XGB 的 meta prediction。

MLP 输入列：

| 输入列 | 含义 |
| --- | --- |
| `pred_lgb` | LightGBM top40 预测值 |
| `pred_xgb` | XGBoost top40 预测值 |
| `rank_lgb` | 当日 `pred_lgb` 的截面百分位 rank |
| `rank_xgb` | 当日 `pred_xgb` 的截面百分位 rank |
| `pred_mean` | `(pred_lgb + pred_xgb) / 2` |
| `rank_mean` | `(rank_lgb + rank_xgb) / 2` |
| `pred_diff` | `pred_lgb - pred_xgb` |
| `rank_diff` | `rank_lgb - rank_xgb` |

MLP 输出：

```text
residual_rank_pred
```

最终融合输出：

```text
pred = final_pred = pred_lgb + 1.5 * residual_rank_pred
```

## 5. 现有模型清单

当前仓库里主要有这些模型结果：

| 模型 | 路径 | 用途 | 是否最终推荐 |
| --- | --- | --- | --- |
| GRU layer1 基线 | `outputs/models/20260530_103903__final_test_eval/layer1/test/test_pred.parquet` | 深度学习基线 | 否 |
| MLP 基线 | `outputs/models/20260530_103903__final_test_eval/mlp_baseline/test/test_pred.parquet` | 非序列深度学习基线 | 否 |
| LightGBM top40 | `outputs/models/20260530_205006__feature_selection/lightgbm_top40/lightgbm/test/test_pred.parquet` | 最强纯树模型基线 | 作为强基线 |
| XGBoost top40 | `outputs/models/20260530_205006__feature_selection/xgboost_top40/xgboost/test/test_pred.parquet` | 纯树模型基线 | 否 |
| Residual-rank deep_ln alpha=1.5 | `outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet` | 最终融合模型 | 是 |

对应模型文件：

| 模型 | 模型文件 |
| --- | --- |
| LightGBM top40 | `outputs/models/20260530_205006__feature_selection/lightgbm_top40/lightgbm/model.txt` |
| XGBoost top40 | `outputs/models/20260530_205006__feature_selection/xgboost_top40/xgboost/model.json` |
| Residual-rank MLP | `outputs/models/20260531_121653__fusion_rank_tune/alpha_ext_h128_d010_wd1e4/residual_rank_mlp/residual_rank_mlp.pt` |

## 6. 模型结果对比

最终模型和两个关键基线的 test 结果：

| 模型 | 是否包含深度网络 | test IC | test ICIR | top-k 收益 | top-k Sharpe | top-k 最大回撤 | rolling 收益 | rolling Sharpe | rolling 最大回撤 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU layer1 基线 | 是 | 0.085123 | 0.648694 | 0.358659 | - | - | 0.875401 | - | - |
| LightGBM top40 基线 | 否 | 0.106769 | 0.817390 | 6.346443 | 2.830258 | -0.142791 | 5.413240 | 6.594812 | -0.070443 |
| Residual-rank deep_ln, alpha=1.5 | 是 | 0.106807 | 0.821452 | 4.424243 | 4.261833 | -0.083557 | 4.330470 | 6.013328 | -0.069576 |

解读：

- 最终模型相比 GRU 基线明显更强。
- 最终模型相比 LightGBM top40，IC/ICIR 小幅更高。
- LightGBM top40 的简单 rolling 收益仍更高。
- 最终模型的 top-k Sharpe 和 top-k maxDD 更好，风险特征更稳。

## 7. 如何重新生成最终预测

如果只是做策略回测，不需要重新跑模型，直接读取：

```text
outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet
```

如果要重新生成，需要先有 LGB/XGB top40 的预测文件：

```text
outputs/models/20260530_205006__feature_selection/lightgbm_top40/lightgbm/valid/valid_pred.parquet
outputs/models/20260530_205006__feature_selection/lightgbm_top40/lightgbm/test/test_pred.parquet
outputs/models/20260530_205006__feature_selection/xgboost_top40/xgboost/valid/valid_pred.parquet
outputs/models/20260530_205006__feature_selection/xgboost_top40/xgboost/test/test_pred.parquet
```

这些文件由 `src/model_experiments/run_gbdt.py` 生成。典型命令：

```bash
python -m src.model_experiments.run_gbdt \
  --model lightgbm \
  --processed-dir data/processed \
  --out-root outputs/models/20260530_205006__feature_selection/lightgbm_top40 \
  --feature-list outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt \
  --num-threads 16 \
  --num-boost-round 800 \
  --early-stopping-rounds 80

python -m src.model_experiments.run_gbdt \
  --model xgboost \
  --processed-dir data/processed \
  --out-root outputs/models/20260530_205006__feature_selection/xgboost_top40 \
  --feature-list outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt \
  --num-threads 16 \
  --num-boost-round 800 \
  --early-stopping-rounds 80
```

然后用 residual-rank MLP 的 checkpoint 生成最终分数。核心逻辑如下：

```python
final_pred = pred_lgb + 1.5 * residual_rank_pred
```

其中 `residual_rank_pred` 来自：

```text
outputs/models/20260531_121653__fusion_rank_tune/alpha_ext_h128_d010_wd1e4/residual_rank_mlp/residual_rank_mlp.pt
```

当前已经生成好的最终预测文件保存在：

```text
outputs/models/20260531_162154__final_model_handoff/
```

## 8. 策略侧建议

策略侧第一版可以直接做：

1. 每天读取当日全部股票的 `pred`。
2. 按 `pred` 降序排序。
3. 选 top K 股票。
4. 每日生成交易列表。
5. 单笔持有多天，例如持有 5 天。

建议先对这些参数做敏感性测试：

| 参数 | 候选值 |
| --- | --- |
| top K | `10 / 20 / 50` |
| 持有天数 | `3 / 5 / 10` |
| 调仓频率 | 每日 |
| 交易成本 | `5 bps` 起步 |

当前简单回测说明：

- LightGBM top40 在当前 rolling 收益上更高。
- Final fusion model 在 ICIR 和 top-k 风险指标上更好。
- 因此策略侧不要只用一个固定 topK 策略定结论，应该做 topK 和持有天数的敏感性测试。

## 9. 常见问题

Q: 策略侧应该用哪个字段排序？

A: 用 `pred`。它和 `final_pred` 相同，保留 `final_pred` 只是为了可解释。

Q: `pred` 是收益率吗？

A: 不是。`pred` 是模型打分，主要用于同一天股票之间排序。不要把它当作未来收益率的数值预测。

Q: `pred` 越大越好吗？

A: 是。越大表示模型越看好。

Q: label 字段能不能用于选股？

A: 不能。`label_5d__cs_rank`、`label_5d`、`label_1d` 是未来信息，只能用于回测评估。

Q: 如果未来要实盘或新日期预测怎么办？

A: 需要先用同样的特征工程生成新日期的 processed features，再用保存的 LightGBM/XGBoost 模型得到 `pred_lgb/pred_xgb`，最后用 residual-rank MLP 生成 `residual_rank_pred` 并计算 `final_pred`。

Q: 当前最终模型是否已经包含深度学习网络？

A: 是。最终模型中的 residual-rank MLP 是 `deep_ln` 网络，结构为 `Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm -> GELU -> Dropout -> Linear`。

## 10. 交接 zip 包内容

建议交接包文件名：

```text
model_handoff_alpha15.zip
```

交接包中应包含以下内容。

### 10.1 文档

```text
docs/dev/08_model_handoff.md
docs/dev/07_后续模型实验记录.md
```

用途：

- `08_model_handoff.md` 是策略端优先阅读的交接文档。
- `07_后续模型实验记录.md` 用于追溯模型选择、alpha 调参、基线对比和实验过程。

### 10.2 最终模型和最终预测

```text
outputs/models/20260531_162154__final_model_handoff/
outputs/models/20260531_121653__fusion_rank_tune/
```

其中最关键的是：

```text
outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet
outputs/models/20260531_162154__final_model_handoff/valid/valid_pred.parquet
outputs/models/20260531_162154__final_model_handoff/summary.json
outputs/models/20260531_121653__fusion_rank_tune/alpha_ext_h128_d010_wd1e4/residual_rank_mlp/residual_rank_mlp.pt
outputs/models/20260531_121653__fusion_rank_tune/alpha_ext_h128_d010_wd1e4/residual_rank_mlp/summary.json
```

用途：

- `test_pred.parquet` 是策略侧可直接读取的最终预测结果。
- `residual_rank_mlp.pt` 是最终深度融合模型参数。
- `outputs/models/20260531_121653__fusion_rank_tune/` 保留了 alpha 和结构调参结果，方便复核为什么最终选 `alpha=1.5`。

### 10.3 关键基线

```text
outputs/models/20260530_205006__feature_selection/lightgbm_top40/
outputs/models/20260530_205006__feature_selection/xgboost_top40/
outputs/models/20260530_103903__final_test_eval/layer1/
outputs/models/20260530_103903__final_test_eval/mlp_baseline/
outputs/models/20260530_015421__sequence_full/e1_gru_5d_rank/
outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt
```

用途：

- `lightgbm_top40` 是最强纯树模型基线，也是最终融合模型的主基模型。
- `xgboost_top40` 是最终融合模型的另一个 meta input。
- `layer1` / `e1_gru_5d_rank` 是 GRU 基线，用于体现深度学习基线。
- `mlp_baseline` 是非序列深度学习基线。
- `lightgbm_top40.txt` 是 top40 特征列表，用于复现实验。

### 10.4 代码和配置

```text
src/
configs/
requirements.txt
```

用途：

- `src/model_experiments/run_fusion_methods.py`：最终 residual-rank fusion 训练和评估逻辑。
- `src/model_experiments/run_gbdt.py`：LightGBM / XGBoost 训练和预测逻辑。
- `src/model_experiments/run_prediction_ensemble.py`：早期 ensemble 对照。
- `src/model_experiments/run_rolling_tranche_eval.py`：rolling tranche 回测相关逻辑。
- `configs/` 和 `requirements.txt` 用于复现实验环境。

### 10.5 不放入交接包的内容

不建议放入：

```text
data/
outputs/ 下所有无关实验目录
__pycache__/
.git/
```

原因：

- `data/` 体积大且可能涉及原始数据分发问题。
- 无关实验目录会让策略端难以判断应该使用哪个模型。
- Python cache 和 git metadata 没有交接价值。

## 11. 交接包使用顺序

策略端拿到 zip 后，建议按这个顺序使用：

1. 先读：

```text
docs/dev/08_model_handoff.md
```

2. 直接读取最终测试预测：

```text
outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet
```

3. 用以下字段生成策略：

```text
trade_date, ts_code, pred
```

4. 每天按 `pred` 降序排序，生成候选股票列表。

5. 如果要对比基线，再读取：

```text
outputs/models/20260530_205006__feature_selection/lightgbm_top40/lightgbm/test/test_pred.parquet
outputs/models/20260530_103903__final_test_eval/layer1/test/test_pred.parquet
```

## 12. 交接时需要强调的话

给策略端时建议明确说明：

```text
最终模型输出是每日-股票级别的排序分数 pred，不是收益率，也不是买卖指令。
策略侧需要按 trade_date 分组，用 pred 降序排序，再自行决定 topK、持仓天数、调仓和风控规则。
label_5d__cs_rank、label_5d、label_1d 是未来信息，只能用于回测评估，不能用于生成交易。
```
