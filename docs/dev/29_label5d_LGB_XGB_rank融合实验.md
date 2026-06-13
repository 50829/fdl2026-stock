# label5d LGB/XGB rank 融合实验

日期：2026 年 6 月 13 日

## 目的

本实验回答一个问题：`label5d` 是否也应该像 `label1d` 一样使用 LightGBM 和 XGBoost 的 rank 融合结果，而不是单独使用 XGBoost 或 LightGBM。

本实验不重训模型，只使用已经训练好的 `label5d` LightGBM 和 XGBoost 预测文件做预测层融合，然后用同一套策略回测协议评估。

## 输出目录

| 内容 | 路径 |
| --- | --- |
| 融合预测与模型注册表 | `outputs/models/20260613_212541__report_label5d_lgb_xgb_rank_fusion` |
| 策略回测 | `outputs/strategy/20260613_212612__report_label5d_lgb_xgb_rank_fusion_strategy` |

## 融合方式

| 融合模型 | 说明 |
| --- | --- |
| `fusion_rank_equal_all` | LGB 和 XGB 每日截面 rank 等权平均 |
| `fusion_rank_valid_ic_weighted_all` | 按验证集 IC 加权 |
| `fusion_rank_valid_icir_weighted_all` | 按验证集 ICIR 加权 |

## 预测指标

| 模型 | Valid IC | Valid ICIR | Test IC | Test ICIR |
| --- | ---: | ---: | ---: | ---: |
| label5d LightGBM | 0.1288 | 0.7890 | 0.1068 | 0.8174 |
| label5d XGBoost | 0.1282 | 0.7934 | 0.1055 | 0.8049 |
| 等权 rank 融合 | 0.1289 | 0.7907 | 0.1064 | 0.8107 |
| Valid IC 加权 rank 融合 | 0.1289 | 0.7907 | 0.1064 | 0.8107 |
| Valid ICIR 加权 rank 融合 | 0.1289 | 0.7907 | 0.1064 | 0.8107 |

预测层结论：rank 融合的 Test IC/ICIR 介于 LGB 和 XGB 附近，没有明显超过单独 LightGBM。说明两个树模型在 `label5d` 上的信息重叠较高，简单 rank 融合不能稳定产生新的 IC 增益。

## 策略结果

统一设置：

- 策略：`topk10_drop2`、`topk20_drop3`、`rolling_p10_h5`、`rankbuf_p20_b50_s100_min2_max10`
- 交易成本：5 bps
- 交易约束：使用 `data/processed/universe.parquet`
- 流动性阈值：本轮设为 `0.0`，用于和前一轮报告策略实验保持可比

核心测试集结果如下：

| 模型 | 策略 | Test 总收益 | Test 夏普 | Test 最大回撤 | Test 平均换手 |
| --- | --- | ---: | ---: | ---: | ---: |
| label5d LightGBM | TopK10 Drop2 | 6.4341 | 6.6127 | -6.27% | 71.45% |
| label5d XGBoost | TopK10 Drop2 | 12.8943 | 7.9611 | -11.32% | 78.02% |
| 等权 rank 融合 | TopK10 Drop2 | 9.2292 | 7.3738 | -7.52% | 74.98% |
| Valid IC 加权 rank 融合 | TopK10 Drop2 | 8.9024 | 7.2597 | -7.53% | 74.93% |
| Valid ICIR 加权 rank 融合 | TopK10 Drop2 | 9.3915 | 7.2915 | -7.53% | 75.37% |
| label5d XGBoost | Rolling P10 H5 | 4.1556 | 6.8487 | -6.56% | 38.89% |
| 等权 rank 融合 | Rolling P10 H5 | 4.1931 | 6.9072 | -7.25% | 38.95% |

## 结论

1. 这个实验值得保留在报告里，但不应该改变最终主模型。
2. `label5d` 的 LGB/XGB rank 融合没有显著提升预测 IC，也没有在测试集 TopK10 Drop2 上超过单独 XGBoost。
3. TopK10 Drop2 下，融合的主要作用是平滑排名并降低回撤：等权融合最大回撤约 -7.52%，低于 XGBoost 的 -11.32%，但总收益从 12.8943 降到 9.2292。
4. Rolling P10 H5 下，等权融合的收益和夏普略高于 XGBoost，但回撤略差，优势很小。
5. 因此，`label5d LGB/XGB rank 融合` 适合作为稳健性消融和低回撤备选，不作为最终主模型；最终主线仍是 `label1d LGB+XGB rank 融合 + TopK10 Drop2`。
