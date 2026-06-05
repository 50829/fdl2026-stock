# 架构说明

本仓库按稳定边界组织代码：

- `src/data/`：原始数据到处理后数据的构建逻辑，以及特征元数据。
- `src/models/`：只放可复用的模型类。
- `src/model_experiments/`：训练、融合、模型评估等实验命令实现。
- `src/evaluation/`：预测指标和兼容适配层。
- `src/strategy/`：统一的策略回测引擎。
- `src/pipelines/`：可复现的模型交接、每日排序和产物管理流程。
- `configs/registry/`：模型、产物、实验和策略注册表。

所有面向使用者的命令都应通过统一入口运行：

```bash
python -m src.experiments <command>
```

实验输出不要依赖会变化的别名目录。每次运行目录应保持不可变，并使用时间戳前缀：

```text
outputs/models/YYYYMMDD_HHMMSS__experiment_name/
outputs/strategy/YYYYMMDD_HHMMSS__strategy_name/
outputs/live/YYYYMMDD__model_or_strategy__from_YYYYMMDD/
```

被选中的稳定产物引用应写在注册表里，不要硬编码在代码中。
