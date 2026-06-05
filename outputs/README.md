# 输出目录

除 Markdown 说明外，本地产物默认不纳入 git。新的运行应使用以下根目录：

- `outputs/models/<YYYYMMDD_HHMMSS>__<experiment_name>/`：模型训练、预测和评估产物。
- `outputs/strategy/<YYYYMMDD_HHMMSS>__<run_name>/`：策略回测产物。
- `outputs/live/<trade_date>__<model_or_strategy>__from_<decision_date>/`：每日交易计划产物。

使用 `configs/registry/models.yaml` 登记策略回测可选的模型预测文件；每日交易模型等通用产物登记在 `configs/registry/artifacts.yaml`。优先更新注册表，不要在脚本中新增硬编码路径。

检查本地旧命名目录：

```bash
python -m src.experiments normalize-outputs --dry-run
```

迁移本地旧命名目录：

```bash
python -m src.experiments normalize-outputs --apply
```
