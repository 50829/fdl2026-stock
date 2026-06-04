# FDL2026 Stock Trend Prediction

深度学习基础大作业：基于 A 股日频数据的趋势预测、模型融合与模拟交易评估。

## 环境配置

推荐 Python 3.12。CPU 环境可直接安装：

```bash
pip install -r requirements.txt
```

如需 CUDA 版本 PyTorch，先按本机 CUDA 版本安装官方 wheel，再安装其余依赖：

```bash
pip install -r requirements.txt
```

## 目录结构

```text
configs/        # MLP/GRU 主实验配置
data/           # 原始数据和 processed parquet，默认不提交
src/data/       # 数据读取、特征、标签、缓存数据集
src/models/     # 可复用模型定义
src/model_experiments/ # 实验实现模块，不再作为推荐命令入口
src/evaluation/ # IC/ICIR 和模型评测适配层，回测执行复用 src/strategy
src/strategy/   # 唯一策略回测引擎，输出 curve/trades/holdings/metrics
src/pipelines/  # 可复现实验和交接产物生成实现
outputs/        # 模型、预测、指标产物，默认不提交
docs/           # 实验记录与报告
```

## 数据预处理

```bash
python -m src.experiments preprocess --config configs/config.yaml
```

主实验默认读取：

- `data/processed/feature_meta.json`
- `data/processed/splits.json`
- `data/processed/features.parquet`
- `data/processed/labels.parquet`
- `data/processed/universe.parquet`

## 统一入口

所有实验命令统一从 `src.experiments` 进入：

```bash
python -m src.experiments --help
```

## 基线模型

通用 Torch 模型可以直接用 YAML 配置训练/预测：

```bash
python -m src.experiments train --config configs/exp_e0_mlp_5d_rank.yaml
python -m src.experiments predict --config configs/exp_e0_mlp_5d_rank.yaml
```

MLP/GRU 主入口：

```bash
python -m src.experiments gru --experiments e0_full e1_full --stage train eval predict --out-root outputs/models --run-name sequence_e0_e1
```

GRU 消融实验入口：

```bash
python -m src.experiments gru-ablation --processed-dir data/processed --out-root outputs/models --run-name sequence_ablation_full
```

## 树模型

LightGBM/XGBoost 训练、评测、预测统一入口：

```bash
python -m src.experiments gbdt --model lightgbm --processed-dir data/processed --out-root outputs/models --run-name gbdt_full
python -m src.experiments gbdt --model xgboost --processed-dir data/processed --out-root outputs/models --run-name gbdt_full
```

使用已经筛好的 top40 特征：

```bash
python -m src.experiments gbdt --model lightgbm --processed-dir data/processed --feature-list outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt --out-root outputs/models --run-name feature_selection_lightgbm_top40
python -m src.experiments gbdt --model xgboost --processed-dir data/processed --feature-list outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt --out-root outputs/models --run-name feature_selection_xgboost_top40
```

## 融合模型

Residual-rank / stacking / leaf embedding 等融合实验入口：

```bash
python -m src.experiments fusion --processed-dir data/processed --out-root outputs/models --run-name fusion_methods --experiments residual_rank --mlp-arch deep_ln --alpha-grid 0.0 0.25 0.5 0.75 1.0 1.5
```

最终交接模型使用 residual-rank deep_ln，默认 `alpha=1.5`。用已保存的 residual-rank MLP checkpoint 和 LightGBM/XGBoost top40 预测复现最终交接文件：

```bash
python -m src.experiments final-handoff --alpha 1.5 --out-root outputs/models --run-name final_model_handoff
```

输出：

- `outputs/models/<YYYYMMDD_HHMMSS>__final_model_handoff/valid/valid_pred.parquet`
- `outputs/models/<YYYYMMDD_HHMMSS>__final_model_handoff/test/test_pred.parquet`
- `outputs/models/<YYYYMMDD_HHMMSS>__final_model_handoff/summary.json`

预测文件核心字段：

- `pred` / `final_pred`：策略流程使用的最终排序分数
- `pred_lgb`、`pred_xgb`：树模型基础分数
- `residual_rank_pred`：深度网络预测的 residual-rank 修正项
- `alpha`：修正项权重

## 评测与回测

统一策略回测引擎位于 `src/strategy/`；`src/evaluation/` 只保留模型评测适配层。对已有预测文件做回测敏感性分析：

```bash
python -m src.experiments backtest-sensitivity --pred final valid outputs/models/20260531_162154__final_model_handoff/valid/valid_pred.parquet --pred final test outputs/models/20260531_162154__final_model_handoff/test/test_pred.parquet
```

策略网格回测入口：

```bash
python -m src.experiments strategy-backtest --models final lgb_top40 --splits valid test --out-root outputs/strategy --run-name strategy_backtest
```

默认模型路径来自 `configs/registry/models.yaml`。新增模型或替换最终模型时，优先改注册表，而不是在脚本里改路径。

## Live 交易计划

每日收盘后生成下一交易日排序：

```bash
python -m src.experiments live-rank --decision-date 20260603 --trade-date 20260604 --out-dir outputs/live/20260604__final__from_20260603
```

可选 `--watchlist watchlist.csv`，文件包含 `ts_code` 和可选 `stock_name`/`name` 列。

## 输出目录命名

新实验统一使用：

- `outputs/models/<YYYYMMDD_HHMMSS>__<experiment_name>/`：训练、预测、模型评测产物
- `outputs/strategy/<YYYYMMDD_HHMMSS>__<run_name>/`：策略回测产物
- `outputs/live/<trade_date>__<model_or_strategy>__from_<decision_date>/`：每日交易计划

本地历史产物可用命令检查或迁移命名：

```bash
python -m src.experiments normalize-outputs --dry-run
```
