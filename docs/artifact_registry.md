# 产物注册表

产物引用统一写在 `configs/registry/artifacts.yaml`。

单个文件或目录写在 `artifacts` 下：

```yaml
artifacts:
  prediction.final.test:
    kind: prediction
    split: test
    path: outputs/models/YYYYMMDD_HHMMSS__final_model_handoff/test/test_pred.parquet
```

需要成组传入命令的输入写在 `bundles` 下：

```yaml
bundles:
  final_handoff_inputs:
    fusion_model: model.residual_rank_alpha_1_5
    valid_lgb: prediction.lightgbm_top40.valid
    test_lgb: prediction.lightgbm_top40.test
```

`configs/registry/models.yaml` 的范围更窄：它只登记策略回测中可以按名称选择的预测文件。`configs/registry/experiments.yaml` 和 `configs/registry/strategies.yaml` 保存命令默认值；除非引用产物键，否则不应直接写具体产物路径。

当新的模型成为选定的交接模型时，应更新注册表条目，而不是修改 Python 默认路径。
