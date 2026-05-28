# Processed Data Handoff

这是已经预处理好的建模数据。接手训练/建模的人可以从这里直接开始，不需要再读 `data/raw/` 里的几千个 CSV。

本目录由下面命令生成：

```bash
/home/mirawind/miniforge3/bin/conda run -n fdl python -m src.data.preprocess --config configs/config.yaml
```

## Parquet 是什么

`.parquet` 是一种表格数据文件格式，可以理解成“更快、更省空间的 CSV”。它适合存几百万行数据。

读取方式不是 `pd.read_csv()`，而是：

```python
import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")
```

如果报错缺少 `pyarrow`，安装：

```bash
pip install pyarrow
```

本项目推荐直接用 `fdl` 环境：

```bash
/home/mirawind/miniforge3/bin/conda run -n fdl python your_script.py
```

## 数据范围

- 开始日期：`20190102`
- 结束日期：`20260518`
- 交易日数量：`1785`
- 股票数量：`5745`
- 面板行数：`8460388`
- 默认特征数量：`104`
- 完整候选特征池：`110`
- 特征样本：`7856700` 行，日期 `20190402` 到 `20260518`
- 标签样本：`7824281` 行，日期 `20190402` 到 `20260508`

## 文件说明

- `features.parquet`：模型输入特征。建模主要读这个。
- `labels.parquet`：训练目标。建模主要读这个。
- `feature_meta.json`：默认特征列、完整候选特征池、特征分组。训练和预测必须按这里的顺序取列。
- `splits.json`：训练、验证、测试的日期切分。
- `panel.parquet`：合并后的原始大表。只有需要重新造特征时才读。
- `universe.parquet`：每日股票池过滤结果。通常已经体现在 `features.parquet` 和 `labels.parquet` 里。
- `data_quality.json`：数据质量检查摘要。只用于了解数据是否异常。

每张主要表都有两个主键：

```text
trade_date, ts_code
```

含义：

- `trade_date`：交易日，例如 `20240528`
- `ts_code`：股票代码，例如 `000001.SZ`

## 最短使用方式

如果你只是要训练模型，从 `features.parquet + labels.parquet` 开始。

```python
import json
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")

features = pd.read_parquet(data_dir / "features.parquet")
labels = pd.read_parquet(data_dir / "labels.parquet")

with open(data_dir / "feature_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# 默认推荐特征列。当前默认包含 baseline + RSI/KDJ + 量价配合，
# 不包含 MACD；MACD 保存在候选特征池里，适合做对照实验。
feature_cols = meta["feature_columns"]

df = features.merge(labels, on=["trade_date", "ts_code"], how="inner")

X = df[feature_cols]
y = df["label_5d__cs_rank"]

print(X.shape, y.shape)
```

推荐第一版训练目标：

```text
label_5d__cs_rank
```

它表示未来 5 日收益在当天所有股票中的相对排名，范围大致是 `[-1, 1]`。越大表示越看好。

可以做对照实验的目标：

```text
label_1d__cs_rank
```

## 时间切分

不要随机切分数据。金融时序必须按时间切。

```python
train = df[(df["trade_date"] >= "20190101") & (df["trade_date"] <= "20231231")]
valid = df[(df["trade_date"] >= "20240101") & (df["trade_date"] <= "20241231")]
test = df[(df["trade_date"] >= "20250101")]
```

也可以读取 `splits.json`：

```python
import json

with open("data/processed/splits.json", "r", encoding="utf-8") as f:
    splits = json.load(f)

print(splits)
```

## 字段说明

`features.parquet` 包含：

- `trade_date`
- `ts_code`
- 110 个候选模型特征，其中 `feature_meta.json` 默认推荐 104 个

特征名大致分几类：

- `*_cs_rank`：当天截面排名特征，已经缩放到 `[-1, 1]`
- `*_ts_z60`：按单只股票历史 60 日计算的 rolling z-score
- `*_missing`：缺失标记，1 表示原始值缺失过
- `rsi_*`、`kdj_*`：超买超卖/震荡指标
- `corr_ret_logvol_chg_*`、`ret_x_volume_ratio_*`、`turnover_shock_20`：量价配合特征
- `macd_*`：MACD 候选特征，默认不启用

`labels.parquet` 包含：

- `label_1d`：从 T+1 买入到 T+2 卖出的原始收益率，计算方式是 `close[T+2] / close[T+1] - 1`
- `label_5d`：从 T+1 买入到 T+6 卖出的原始收益率，计算方式是 `close[T+6] / close[T+1] - 1`
- `label_1d__cs_rank`：`label_1d` 的每日截面排名
- `label_5d__cs_rank`：`label_5d` 的每日截面排名

## 注意事项

- 特征只使用当日盘后及以前可获得的数据。
- 标签从 `T+1` 买入价开始计算，避免使用不可交易收益。
- `label_5d` 需要未来 6 个交易点，因此标签结束日期会早于特征结束日期。
- 训练、验证、测试按时间切分，禁止随机日期切分。
- 训练时必须使用 `feature_meta.json` 里的 `feature_columns` 或按 `feature_groups` 展开列，不要自己按字母排序。
- `features.parquet` 日期到 `20260518`，但 `labels.parquet` 到 `20260508`，这是正常的。
- 预测最新日期时只需要 `features.parquet`，不需要 label。

## 常见任务

### 训练 MLP

直接使用一行一个样本：

```text
X = 默认 104 个特征
y = label_5d__cs_rank
```

### 做特征消融实验

`feature_meta.json` 里有特征分组：

```text
core_price
volume_liquidity
momentum_ma
volatility
moneyflow
fundamental_size
ts_zscore
oscillator
volume_price_interaction
macd
```

默认组为：

```text
baseline + oscillator + volume_price_interaction
```

其中 `oscillator` 是 RSI/KDJ，`volume_price_interaction` 是量价配合，`macd` 已存入候选池但默认不启用。做 MACD 对照实验时，可以从 `feature_groups["macd"]` 额外取列。

### 训练 LSTM/Transformer

需要自己按股票构造滑动窗口：

```text
同一只股票过去 30 或 60 个交易日的特征序列 -> 当前日期的 label_5d__cs_rank
```

排序前必须按：

```python
df = df.sort_values(["ts_code", "trade_date"])
```

### 每日选股

模型输出每个 `(trade_date, ts_code)` 的 `score` 后，在同一天内部排序：

```python
pred["rank"] = pred.groupby("trade_date")["score"].rank(ascending=False)
top = pred[pred["rank"] <= 10]
```

然后取 Top-N 股票作为候选买入池。

## 重新生成数据

如果原始数据更新了，重新运行：

```bash
/home/mirawind/miniforge3/bin/conda run -n fdl python -m src.data.preprocess --config configs/config.yaml
```

会覆盖本目录下的 processed 文件。
