# Processed Data Handoff

这份目录是已经预处理好的建模数据。训练模型的人一般只需要读：

- `features.parquet`
- `labels.parquet`
- `feature_meta.json`
- `splits.json`

不需要重新扫 `data/raw/` 里的逐日 CSV。

## 最短训练用法

```python
import json
from pathlib import Path

import pandas as pd

data_dir = Path("data/processed")

features = pd.read_parquet(data_dir / "features.parquet")
labels = pd.read_parquet(data_dir / "labels.parquet")

with open(data_dir / "feature_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

with open(data_dir / "splits.json", "r", encoding="utf-8") as f:
    splits = json.load(f)

feature_cols = meta["feature_columns"]  # 112 列，顺序已经固定
label_col = "label_5d__cs_rank"         # 推荐 baseline 训练目标

df = features.merge(labels, on=["trade_date", "ts_code"], how="inner")

train_start, train_end = splits["train"]
valid_start, valid_end = splits["valid"]

train = df[(df["trade_date"] >= train_start) & (df["trade_date"] <= train_end)]
valid = df[(df["trade_date"] >= valid_start) & (df["trade_date"] <= valid_end)]

X_train = train[feature_cols].astype("float32")
y_train = train[label_col].astype("float32")

X_valid = valid[feature_cols].astype("float32")
y_valid = valid[label_col].astype("float32")

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
```

如果要先检查元数据和 parquet 是否一致：

```bash
python -m src.data.feature_meta \
  --meta data/processed/feature_meta.json \
  --features data/processed/features.parquet
```

正常输出里会有：

```text
"status": "ok"
```

## 数据范围

本目录由以下命令生成：

```bash
python -m src.data.preprocess --config configs/config.yaml
```

当前数据范围：

| 文件 | 行数 | 列数 | 日期范围 | 股票数 |
| --- | ---: | ---: | --- | ---: |
| `panel.parquet` | 10,692,159 | 45 | 20160104 - 20260518 | 5,758 |
| `universe.parquet` | 10,692,159 | 9 | 20160104 - 20260518 | 5,758 |
| `features.parquet` | 7,927,579 | 114 | 20160108 - 20260518 | 5,184 |
| `labels.parquet` | 7,632,257 | 13 | 20160108 - 20260508 | 5,184 |

`features.parquet` 的 114 列中有 2 个键列：

```text
trade_date, ts_code
```

其余 112 列是模型输入特征。

## 主键

所有主要表都使用同一组主键：

```text
trade_date, ts_code
```

含义：

- `trade_date`：交易日，字符串格式，例如 `20240528`
- `ts_code`：股票代码，例如 `000001.SZ`

训练时通常这样合并：

```python
df = features.merge(labels, on=["trade_date", "ts_code"], how="inner")
```

`features` 的最后日期比 `labels` 晚，这是正常的。最新几天可以做预测，但还没有未来收益标签。

## 文件说明

### `features.parquet`

模型输入特征表。

键列：

```text
trade_date, ts_code
```

特征列数量：

```text
112
```

这些特征已经完成：

- ST、北交所、新股、低流动性股票过滤
- 在可交易股票池内部做截面标准化
- 缺失值填充
- 缺失标记列生成
- 技术指标、资金流、基本面、市值、行业相对特征等构造

特征名后缀含义：

| 后缀 | 含义 |
| --- | --- |
| `__cs_rank` | 当天可交易股票池内截面排名，范围约为 `[-1, 1]` |
| `__missing` | 原始特征是否缺失，1 表示缺失 |
| `__ts_z60` | 单只股票自身 60 日 rolling z-score |
| `__cs_robust_z` | 当天截面 robust z-score，按 median/MAD 计算并 clip |

### `labels.parquet`

训练标签表。

重要列：

| 列名 | 含义 |
| --- | --- |
| `buy_date` | 对 `trade_date` 盘后预测后，下一交易日买入日期 |
| `sell_date_1d` | 1 日持有标签对应卖出日期 |
| `sell_date_5d` | 5 日持有标签对应卖出日期 |
| `label_1d` | `T+1` 买入到 `T+2` 卖出的原始收益 |
| `label_5d` | `T+1` 买入到 `T+6` 卖出的原始收益 |
| `label_1d_excess` | `label_1d` 减去当天股票池等权平均收益 |
| `label_5d_excess` | `label_5d` 减去当天股票池等权平均收益 |
| `label_1d__cs_rank` | `label_1d` 的当天截面排名 |
| `label_5d__cs_rank` | `label_5d` 的当天截面排名 |
| `label_1d_excess__cs_rank` | `label_1d_excess` 的当天截面排名 |
| `label_5d_excess__cs_rank` | `label_5d_excess` 的当天截面排名 |

推荐 baseline 训练目标：

```text
label_5d__cs_rank
```

原因：5 日收益比 1 日收益噪声更低，也更适合低频调仓。

注意：`label_5d_excess__cs_rank` 和 `label_5d__cs_rank` 排名通常相同，因为同一天所有股票都减去同一个市场均值，不改变截面排序。`excess` 更适合做回归目标对照。

### `feature_meta.json`

特征元数据。

最重要的字段：

```text
feature_columns
feature_groups
features
config
```

`feature_columns` 是训练时应该使用的完整特征列，已经按 `features.parquet` 的真实列顺序排列。

训练时不要自己按字母排序特征列。直接使用：

```python
feature_cols = meta["feature_columns"]
```

`feature_groups` 用于做消融实验。当前分组包括：

| 组名 | 特征数 | 含义 |
| --- | ---: | --- |
| `core_price` | 7 | 基础价格/收益特征 |
| `volume_liquidity` | 18 | 成交量、成交额、换手率、量比 |
| `momentum_ma` | 18 | 动量和均线偏离 |
| `volatility` | 9 | 波动率 |
| `moneyflow` | 16 | 资金流比例和滚动资金流 |
| `fundamental_size` | 9 | 估值、市值、基本面缺失标记 |
| `oscillator` | 12 | RSI、KDJ |
| `macd` | 6 | MACD 相关特征 |
| `industry_relative` | 5 | 行业动量、个股相对行业强弱 |
| `candlestick` | 2 | 收盘价在日内区间的位置 |
| `volume_price_interaction` | 10 | 量价相关、量价交互、换手冲击 |
| `ts_zscore` | 5 | 个股自身时间序列 z-score |
| `robust_z` | 4 | 核心特征截面 robust z-score |

如果想只用某些组训练，需要保持 parquet 列顺序：

```python
selected_groups = ["core_price", "momentum_ma", "industry_relative"]
selected = set()
for group in selected_groups:
    selected.update(meta["feature_groups"][group])

feature_cols = [c for c in meta["feature_columns"] if c in selected]
```

### `splits.json`

时间切分：

```json
{
  "train": ["20160101", "20231231"],
  "valid": ["20240101", "20241231"],
  "test": ["20250101", "20260518"]
}
```

不要随机切分。金融时间序列必须按时间切分。

### `universe.parquet`

每日股票池过滤结果。

重要列：

| 列名 | 含义 |
| --- | --- |
| `in_universe` | 当日是否进入可交易股票池 |
| `is_st` | 当日是否 ST / 风险警示 |
| `market` | 所属市场，例如主板、创业板、北交所 |
| `industry` | 行业 |
| `listed_days_in_data` | 按真实上市日期计算的上市天数 |
| `amount_mean_20` | 20 日平均成交额 |
| `passes_liquidity` | 是否通过流动性过滤 |

训练一般不需要再读 `universe.parquet`，因为 `features.parquet` 和 `labels.parquet` 已经只保留了可交易股票池内样本。

### `panel.parquet`

合并后的原始大表，主要用于重新造特征或排查问题。普通训练不需要读它。

它包含：

- 基础日频量价：`open/high/low/close/pre_close/vol/amount/vwap`
- 每日指标：换手率、估值、市值等
- 资金流：大单、超大单、小单资金流等

## 推荐训练目标

baseline 建议：

```text
X = features[meta["feature_columns"]]
y = labels["label_5d__cs_rank"]
```

可以做的对照：

| 目标 | 用途 |
| --- | --- |
| `label_1d__cs_rank` | 1 日短周期对照，噪声更高 |
| `label_5d__cs_rank` | 推荐主目标 |
| `label_5d` | 原始 5 日收益回归 |
| `label_5d_excess` | 相对市场超额收益回归 |

## 评估建议

模型输出每个 `(trade_date, ts_code)` 的 `score` 后，建议至少算：

- 验证集 loss
- 每日 Spearman IC：当天 `score` 和 `label_5d` 或 `label_5d__cs_rank` 的相关性
- IC 均值
- ICIR：IC 均值 / IC 标准差
- Top-N 回测收益、年化收益、夏普比率、最大回撤

每日 IC 示例：

```python
def daily_ic(pred_df, score_col="score", label_col="label_5d__cs_rank"):
    out = []
    for date, g in pred_df.groupby("trade_date"):
        if len(g) < 3:
            continue
        ic = g[score_col].corr(g[label_col], method="spearman")
        out.append((date, ic))
    return pd.DataFrame(out, columns=["trade_date", "ic"])
```

## 每日预测用法

比赛或模拟交易时，只需要最新日期的 `features.parquet`，不需要 label：

```python
latest_date = features["trade_date"].max()
today = features[features["trade_date"] == latest_date].copy()

X_today = today[feature_cols].astype("float32")
scores = model.predict(X_today)  # 按你们模型接口替换

pred = today[["trade_date", "ts_code"]].copy()
pred["score"] = scores
pred = pred.sort_values("score", ascending=False)

print(pred.head(30))
```

## 关键口径

这份数据已经遵守以下约束：

- 只使用 `2016-01-04` 到 `2026-05-18` 的本地原始数据。
- 股票池排除 ST、北交所、上市不足 60 天、成交量/成交额无效、20 日平均成交额截面最低 20% 的股票。
- 先过滤到可交易股票池，再做截面 rank / robust z / 行业相对特征。
- 标签按全市场交易日历对齐，不按单只股票的下一条记录跳过停牌日期。
- 标签从 `T+1` 买入价开始计算，避免把 `T` 日盘后才知道的信息用于 `T` 日交易。
- 训练、验证、测试按时间切分，不随机打乱日期。

## 常见坑

1. 不要随机划分训练集和验证集。
2. 不要自己按字母排序特征列，必须用 `meta["feature_columns"]`。
3. 不要用 `panel.parquet` 直接训练，除非你清楚自己在重新做标准化和股票池过滤。
4. 最新日期有 features 但没有 labels 是正常的。
5. 如果只用部分 feature group，仍然要按 `meta["feature_columns"]` 的顺序筛列。
6. `label_5d__cs_rank` 是排序任务目标，不是原始收益率；回测收益要用 `label_5d`。
