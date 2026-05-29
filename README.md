# FDL2026 Stock Trend Prediction

深度学习基础大作业：基于 A 股日频数据的趋势预测与模拟交易。

## 环境配置

```bash
pip install -r requirements.txt
```

```bash
pip install -r requirements.txt
```

## 目录结构

```text
configs/            # 统一配置
data/raw/           # 原始数据，不提交
data/processed/     # 预处理产物，不提交
src/data/           # 数据读取、特征、标签、Dataset
src/models/         # 模型定义
src/backtest/       # 指标、策略、回测
tests/smoke_pipeline/ # 跑通流程用的临时 smoke pipeline
tests/configs/      # smoke pipeline 配置
notebooks/          # 探索分析
outputs/            # 模型、预测、图表产出，不提交
logs/               # 训练日志，不提交
docs/dev/           # 开发日志
docs/report/        # 实验报告与插图
```

## 小组分工

- A：数据预处理、特征工程、标签构造、Dataset
- B：模型实现、训练流程、每日预测
- C：指标评估、历史回测、结果可视化

## 当前目标

先跑通最小闭环：`raw csv -> processed parquet -> features/labels -> baseline model -> IC/backtest -> daily prediction`。

## 环境安装

使用 python 3.12 和 cuda 13.0

```bash
conda create -n fdl python=3.12
conda activate fdl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## 数据预处理

```bash
conda run -n fdl python -m src.data.preprocess --config configs/config.yaml
```

处理方案见 `docs/dev/data_preprocessing.md`，产物说明见 `data/processed/README.md`。

## Smoke 流程测试

远端合入的流程测试代码已收在 `tests/smoke_pipeline/`，避免和主线 `src/` 模块重名冲突。

```bash
python tests/smoke_pipeline/train.py --config tests/configs/smoke.yaml
python tests/smoke_pipeline/eval.py --config tests/configs/smoke.yaml
```

Smoke pipeline 会把产物输出到 `outputs/smoke/`（默认）：

- `ckpt.pt`：checkpoint
- `val_pred.csv`：逐样本预测与标签（可选）
- `ic_by_day.csv`：逐日 IC 表（可选）
- `eval_metrics.csv`：评估/回测汇总（可选）
- `bt_curve.csv`：回测资金曲线（可选）

`eval.py` 终端输出也会打印：

- `val_mse/ic_mean/icir/ic_days/samples`
- 回测开启时打印 `{"backtest": {...}}`
