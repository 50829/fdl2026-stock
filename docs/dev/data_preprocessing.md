# Data Preprocessing Plan

本文档记录本项目第一版数据预处理主流程。目标是先跑通一个严格无未来信息泄露、可训练、可回测、可每日预测的 A 股截面选股数据管线。

## 对现有方案的评价

整体方向是正确的：本任务应被建模为“每日截面排序”，而不是单只股票价格预测。作业中的 IC、Top-N 调仓和模拟交易都要求模型在每个交易日给全市场股票打分，因此按天做截面处理是主线。

优点：

- 按日期 CSV 合并成 `(trade_date, ts_code)` 面板是必要地基，后续实验不能每次重新扫几千个 CSV。
- ST 使用 `stock_st/` 按日过滤、北交所使用 `basic.csv` 过滤，这一点符合 point-in-time 思路。
- 时序特征按 `ts_code` 分组计算，截面标准化按 `trade_date` 分组计算，方向清楚。
- 标签使用 `T+1 -> T+2` 收益，符合 A 股盘后数据和 T+1 交易约束。
- 严格按时间切分训练、验证、测试，不随机打乱，这是金融时序任务的底线。

需要修正：

- 不建议对停牌日量价数据随意 forward fill 后继续生成样本。若当日无行情、`vol <= 0` 或 `amount <= 0`，该日该股票不可作为可交易样本。
- Qlib 已提供 `CSZScoreNorm`、`CSRankNorm` 等截面处理器；如果本项目不直接使用 Qlib，则需要自己用 pandas 实现类似逻辑。
- `ProcessInf -> 列均值 -> Fillna 0` 不能机械照搬。神经网络版本更稳的做法是：`inf -> NaN`，标准化后缺失填 0，并保留 missing mask。
- 特征筛选、IC 分析只能使用训练期，不能用验证期、测试期或比赛期共同筛因子。
- 16 日回看窗口偏短。第一版建议使用 30 或 60 个交易日；16 可作为对照实验。
- 新闻数据暂不进入第一版主流程，避免文本时间戳、实体匹配和覆盖区间带来额外复杂度。

## 最终主流程

当前实现入口：

```bash
/home/mirawind/miniforge3/bin/conda run -n fdl python -m src.data.preprocess --config configs/config.yaml
```

处理后的数据说明见 `data/processed/README.md`。

### 1. 构建原始面板

输入：

- `data/raw/daily/*.csv`
- `data/raw/metric/*.csv`
- `data/raw/moneyflow/*.csv`
- `data/raw/basic.csv`
- `data/raw/trade_cal.csv`
- `data/raw/stock_st/*.csv`

处理：

1. 逐日读取 `daily/`，纵向拼接。
2. 按 `(trade_date, ts_code)` merge `metric/` 和 `moneyflow/`。
3. 将数值列尽量转为 `float32`。
4. 按 `trade_date, ts_code` 排序。

输出：

- `data/processed/panel.parquet`

### 2. 数据质量校验

在进入特征工程前，先生成一份轻量质量报告，避免把数据源问题带进训练。

检查项：

- 每日股票数量，若相邻交易日变化异常需要标记。
- `open/high/low/close/pre_close/vwap` 是否存在小于等于 0 的值。
- `vol/amount` 是否存在负值。
- 各字段按日统计缺失率。
- `daily/metric/moneyflow` 在 `(trade_date, ts_code)` 上的覆盖率。
- 是否存在重复 `(trade_date, ts_code)`。

输出：

- `data/processed/data_quality.json`

### 3. 构造每日股票池

每个交易日动态过滤：

- 排除 `basic.market == "北交所"`。
- 排除当日出现在 `stock_st/{date}.csv` 中的股票。
- 排除上市不足 `min_list_days` 的股票，默认先用 60 个交易日，120 个交易日作为对照实验。
- 排除当日无有效行情的股票：缺少 `daily`、`vol <= 0`、`amount <= 0`。
- 可选：排除过去 20 日平均成交额处于每日截面最低 20% 的股票。

注意：

- 股票池应以“当日实际存在且可交易”为准，不能只用当前仍上市股票回溯历史，否则会产生幸存者偏差。
- 若 `daily/` 中包含已退市股票历史，应保留其历史可交易样本；若不包含，需要在报告中说明数据限制。

输出：

- `data/processed/universe.parquet`

### 4. 构造时序特征

蓝色步骤，按股票纵向计算：

```python
df = df.sort_values(["ts_code", "trade_date"])
df.groupby("ts_code")
```

基础特征：

- `ret_1 = close / pre_close - 1`
- `open_gap = open / pre_close - 1`
- `intraday_ret = close / open - 1`
- `high_low_range = high / low - 1`
- `close_vwap_gap = close / vwap - 1`
- `log_vol = log1p(vol)`
- `log_amount = log1p(amount)`

滚动窗口：`5, 10, 20, 60`

- `momentum_w = close / close.shift(w) - 1`
- `volatility_w = rolling_std(ret_1, w)`
- `ma_gap_w = close / rolling_mean(close, w) - 1`
- `volume_ratio_w = vol / rolling_mean(vol, w)`
- `turnover_mean_w`
- `moneyflow_ratio_w`

资金流特征先比例化：

- `net_mf_amount / amount`
- `(buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount) / amount`
- `buy_lg_amount / amount`
- `buy_elg_amount / amount`

基本面特征：

- `turnover_rate`
- `pb`
- `ps_ttm`
- `log_total_mv`
- `log_circ_mv`
- `pe_ttm_missing`
- `dv_ttm_missing`

输出：

- 未标准化特征暂存在内存或 `data/processed/features_raw.parquet`。

### 5. 构造标签

标签基于原始收盘价构造，不使用标准化后的价格。

主标签候选：

```text
label_1d[t] = close[t+2] / close[t+1] - 1
label_5d[t] = close[t+6] / close[t+1] - 1
```

含义：在 `t` 日盘后得到数据和预测，`t+1` 买入，之后按持有期卖出。

第一版同时生成 `label_1d` 和 `label_5d`：

- `label_1d` 用作高换手对照实验。
- `label_5d` 作为主试标签候选，通常信噪比更高，也更贴合作业“不应过高调仓频率”的建议。

可选增强：

```text
excess_label = stock_return - benchmark_return
```

注意：

- 必须基于交易日顺序 shift，不能用自然日加减。
- 标签缺失的样本直接丢弃。
- 训练用标签建议再按每日截面做 rank 或 z-score。

输出：

- `data/processed/labels.parquet`

### 6. 去极值与标准化

第一版不把所有特征都强制处理成同一种尺度，而是保留两类视角。

红色步骤，按交易日横向计算，保留“相对强弱”：

```python
df.groupby("trade_date")
```

推荐第一版使用截面 rank，鲁棒且实现简单：

```text
rank_pct = rank(pct=True)
feature = rank_pct * 2 - 1
```

蓝色步骤，按股票纵向或窗口内计算，保留“自身形态”：

```text
ts_zscore = (x - rolling_mean(x, w)) / rolling_std(x, w)
```

若使用 z-score，则处理顺序为：

1. `inf -> NaN`
2. 每日截面 MAD 或分位数 winsorize
3. 每日截面 z-score
4. 缺失填 0
5. 保留 missing mask

第一版默认：

- 对估值、市值、资金流比例、滚动统计等连续特征生成截面 rank 版本，范围 `[-1, 1]`。
- 对收益率、动量、波动率等时间形态特征保留一组 rolling z-score 版本。
- 对缺失较多的特征添加 `*_missing`。
- 标准化后仍缺失的值填 0。
- 若某特征当日缺失率超过 50%，该日该特征不做 rank，直接填 0 并保留缺失标记。

输出：

- `data/processed/features.parquet`
- `data/processed/feature_meta.json`

`feature_meta.json` 至少包含：

- 特征列名及顺序。
- 每个特征的来源：`daily`、`metric`、`moneyflow`、`basic` 或衍生特征。
- 每个特征的处理方式：`cross_section_rank`、`rolling_zscore`、`binary_mask` 等。
- 训练期缺失率和是否启用。
- 生成配置：窗口长度、标签 horizon、股票池过滤参数。
- 生成时间和代码版本信息。

### 7. 时间切分

严格按交易日切分：

```text
train: 2019-01-01 到 2023-12-31
valid: 2024-01-01 到 2024-12-31
test/backtest: 2025-01-01 到最新可用日期
```

禁止：

- 随机打乱日期。
- 在全量数据上 fit 标准化参数。
- 用验证期、测试期、比赛期做特征筛选。

输出：

- `data/processed/splits.json`

### 8. 构造滑动窗口样本

默认参数：

```text
lookback = 60
horizon = 1 和 5 都生成
```

样本定义：

```text
X[i, t] = stock i 在 t-lookback+1 到 t 的特征序列
y[i, t] = label_h[i, t]，h 可取 1 或 5
```

要求：

- 窗口内特征不能包含 `t+1` 及之后的信息。
- 窗口不足、标签缺失、当日不在股票池内的样本丢弃。
- 模型训练时可以打乱样本顺序，但验证、预测和回测必须按日期恢复截面。

### 9. 每日预测数据

在比赛或模拟交易日 `D` 交易前，最多只能使用前一交易日 `T` 的盘后数据。

流程：

1. 找到 `D` 的前一交易日 `T`。
2. 使用 `T-lookback+1` 到 `T` 的窗口构造特征。
3. 过滤 `T` 日可交易股票池。
4. 输出所有股票预测分数。
5. 取分数最高的 Top-N 作为候选买入池。

预测前必须校验：

- 当前特征列名和 `feature_meta.json` 完全一致。
- 当前特征列顺序和训练时完全一致。
- 最新窗口长度足够。
- 当日股票池不为空。
- 输出分数无 NaN 或 inf。

输出：

- `outputs/predictions/pred_{trade_date}.csv`

字段：

```text
trade_date, ts_code, score, rank
```

## 第一版产物清单

```text
data/processed/panel.parquet
data/processed/data_quality.json
data/processed/universe.parquet
data/processed/features.parquet
data/processed/labels.parquet
data/processed/feature_meta.json
data/processed/splits.json
outputs/predictions/pred_YYYYMMDD.csv
```

## 实现优先级

1. 跑通 `daily + basic + stock_st` 的最小面板、质量检查和股票池。
2. 实现基础量价特征和 `T+1 -> T+2`、`T+1 -> T+6` 标签。
3. 实现截面 rank 与 rolling z-score 两套标准化视角。
4. 实现 60 日滑动窗口 Dataset，并保留 30 日窗口作为对照。
5. 接入 MLP baseline、IC 评估和简单 Top-N 回测。
6. 再加入 `metric`、`moneyflow`、中性化或更复杂模型。

## 最终方案如何确定

最终清洗方案不要靠直觉拍板，应通过“硬约束 + 小规模对照实验”确定。

必须固定的硬约束：

- 不能使用未来信息。
- 股票池必须按日过滤 ST、北交所和无有效行情样本。
- 标签必须从可交易时点开始计算，即至少使用 `T+1` 之后的收益。
- 训练、验证、测试必须按时间切分。
- 每日预测必须和训练时使用相同特征列、相同顺序、相同处理逻辑。

需要用实验决定的选项：

- 标签 horizon：`1d` vs `5d`。
- lookback：`30` vs `60`。
- 上市天数过滤：`60` vs `120`。
- 标准化：纯截面 rank vs 截面 rank + rolling z-score。
- 股票池流动性过滤：不过滤 vs 过滤过去 20 日成交额后 20%。

第一轮只做 4 组最小对照：

```text
A: label_1d, lookback=60, cross_section_rank
B: label_5d, lookback=60, cross_section_rank
C: label_5d, lookback=30, cross_section_rank
D: label_5d, lookback=60, cross_section_rank + rolling_zscore
```

选择标准：

- 验证期 IC 均值更高。
- ICIR 更稳定。
- Top-N 回测最大回撤不过分恶化。
- 换手率与作业交易要求匹配。
- 每日预测在最新日期能稳定产出。

如果指标冲突，优先选择验证期 ICIR 更稳定、回测更稳、工程更简单的方案，而不是只看单次收益最高的方案。
