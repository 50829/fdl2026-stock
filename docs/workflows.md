# 工作流

## 训练或评估模型

优先使用命名实验默认配置：

```bash
python -m src.experiments gbdt --experiment gbdt_full
python -m src.experiments fusion --experiment fusion_methods
python -m src.experiments final-handoff --experiment final_model_handoff
```

每次运行都会写出 `run_meta.json`，记录命令参数、输入产物、注册表快照和 git 状态。

## 运行策略回测

使用命名策略运行配置：

```bash
python -m src.experiments strategy-backtest --strategy-run default_grid
```

策略运行配置定义在 `configs/registry/strategies.yaml`；模型预测文件路径来自 `configs/registry/models.yaml`。

## 生成每日排序

使用选定的每日交易产物包：

```bash
python -m src.experiments live-rank \
  --decision-date 20260603 \
  --trade-date 20260604 \
  --artifact-registry configs/registry/artifacts.yaml \
  --out-dir outputs/live/20260604__final__from_20260603
```

每日交易使用的模型文件来自 `configs/registry/artifacts.yaml`。

## 规范化本地输出目录

检查本地输出目录是否仍有不符合命名规则的旧目录：

```bash
python -m src.experiments normalize-outputs --dry-run
```
