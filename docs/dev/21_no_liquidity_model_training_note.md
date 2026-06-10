# 21 无流动性过滤模型训练记录

日期：2026-06-10

## 1. 背景

用户提出一个合理问题：

```text
能否重新训练一个不含流动性过滤的 label1d 模型，并跑回测看看？
```

这个问题值得做，因为 2026-06-10 的每日预测里出现了明显现象：

```text
部分股票在正式流动性过滤口径下没有排名。
关闭流动性过滤后，部分股票排名很靠前。
```

例如：

| 股票 | 无流动性过滤排名 |
| --- | ---: |
| 燕塘乳业 | 15 |
| 汇得科技 | 37 |
| 奥福科技 | 86 |

因此需要判断：

```text
当前模型是不是因为训练时只看流动性过滤股票，而对低流动性股票泛化不足？
如果训练时放开流动性过滤，IC、ICIR 和策略回测会不会更好？
```

## 2. 已尝试操作

先尝试直接在现有 processed 数据上训练：

```bash
conda run -n fdl python -m src.experiments gbdt \
  --model lightgbm \
  --processed-dir data/processed \
  --run-name label1d_lgb_top40_no_liquidity_filter \
  --feature-list outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt \
  --target label_1d__cs_rank \
  --raw-return-col label_1d \
  --daily-return-col label_1d \
  --no-filter-in-universe \
  --num-threads 8 \
  --num-boost-round 800 \
  --early-stopping-rounds 80 \
  --log-period 100 \
  --step-days 1 \
  --k-rotate 3 \
  --transaction-cost-bps 5.0
```

输出目录：

```text
outputs/models/20260610_133711__label1d_lgb_top40_no_liquidity_filter
```

这次训练耗时约 90 秒，但结果不能作为无流动性过滤模型结论。

原因是样本数和正式流动性过滤模型完全一致：

| 口径 | train rows | valid rows | test samples |
| --- | ---: | ---: | ---: |
| 正式旧模型 | 5,408,072 | 943,810 | 1,280,375 |
| 本次尝试 | 5,408,072 | 943,810 | 1,280,375 |

指标也完全一致：

| split | IC | ICIR | topk step1 Sharpe | topk 最大回撤 |
| --- | ---: | ---: | ---: | ---: |
| valid | 0.083365 | 0.560143 | 2.825156 | -0.359530 |
| test | 0.081799 | 0.617292 | 6.748844 | -0.124631 |

这说明这不是一个真正的新模型，只是在已经过滤后的 processed 数据上重复训练。

## 3. 为什么当前不能直接训练真实无流动性模型

当前 `data/processed/features.parquet` 在预处理阶段已经过滤过股票池。

代码位置：

```text
src/data/preprocess.py
```

关键逻辑：

```python
df = df.merge(universe[["trade_date", "ts_code", "in_universe", "industry"]], ...)
df = df[df["in_universe"].fillna(False)].drop(columns=["in_universe"]).copy()
```

也就是说：

```text
features.parquet 只包含 in_universe=True 的股票。
如果当时 in_universe 包含流动性过滤，那么低流动性股票的历史特征已经被丢掉。
```

现有数据规模也印证了这一点：

```text
universe.parquet rows = 10,692,159
features.parquet rows = 7,927,579
```

因此在训练入口使用：

```text
--no-filter-in-universe
```

只能避免训练时再次过滤，不能恢复已经在预处理阶段丢掉的样本。

## 4. 当前仓库还缺什么

要训练真正的无流动性过滤模型，需要以下二选一：

### 方案 A：补齐完整历史原始数据

需要完整 2016-2026 历史原始数据，目录结构类似：

```text
data/raw_full/
  basic.csv
  daily/YYYYMMDD.csv
  metric/YYYYMMDD.csv
  moneyflow/YYYYMMDD.csv
  stock_st/YYYYMMDD.csv
```

然后重新预处理：

```text
processed_dir = data/processed_no_liquidity
liquidity_filter.enabled = false
```

### 方案 B：提供未过滤的历史 processed

需要已经生成好的：

```text
data/processed_no_liquidity/features.parquet
data/processed_no_liquidity/labels.parquet
data/processed_no_liquidity/universe.parquet
data/processed_no_liquidity/feature_meta.json
data/processed_no_liquidity/splits.json
```

其中 `features.parquet` 必须是在关闭流动性过滤后生成的。

## 5. 当前仓库里的 raw 数据不够

当前仓库有最近每日 zip，例如：

```text
data/raw/0609.zip
data/raw/20260608.zip
data/raw/20260605.zip
```

这些足够做每日 live 预测，但不够训练历史模型。

`data/raw/new.zip` 和 `data/raw/0601.zip` 也只是 2026 年 4 月到 6 月附近的数据，不是完整 2016-2026 历史数据。

所以当前无法完成真实意义上的：

```text
重新训练一个完整历史的无流动性过滤 label1d 模型
```

## 6. 正确实验流程

补齐完整历史 raw 后，应该这样做：

### 6.1 生成无流动性 processed

新建或临时覆盖配置：

```yaml
data:
  raw_dir: data/raw_full
  processed_dir: data/processed_no_liquidity

universe:
  min_list_days: 60
  liquidity_filter:
    enabled: false
    window: 20
    bottom_pct: 0.2
```

运行：

```bash
conda run -n fdl python -m src.experiments preprocess \
  --config configs/config_no_liquidity.yaml
```

### 6.2 训练无流动性模型

```bash
conda run -n fdl python -m src.experiments gbdt \
  --model lightgbm \
  --processed-dir data/processed_no_liquidity \
  --run-name label1d_lgb_top40_no_liquidity_filter \
  --feature-list outputs/models/20260530_205006__feature_selection/features/lightgbm_top40.txt \
  --target label_1d__cs_rank \
  --raw-return-col label_1d \
  --daily-return-col label_1d \
  --filter-in-universe \
  --num-threads 8 \
  --num-boost-round 800 \
  --early-stopping-rounds 80 \
  --log-period 100 \
  --step-days 1 \
  --k-rotate 3 \
  --transaction-cost-bps 5.0
```

注意：

```text
这里可以保留 --filter-in-universe。
因为 data/processed_no_liquidity/universe.parquet 里的 in_universe 已经是不含流动性过滤的基础股票池。
```

### 6.3 回测比较

至少比较：

| 模型 | processed | 训练股票池 | 评估股票池 |
| --- | --- | --- | --- |
| 旧正式模型 | data/processed | 含流动性过滤 | 含流动性过滤 |
| 新模型 A | data/processed_no_liquidity | 不含流动性过滤 | 不含流动性过滤 |
| 新模型 B | data/processed_no_liquidity | 不含流动性过滤 | 再叠加正式流动性过滤 |

重点指标：

```text
valid/test IC
valid/test ICIR
topk20_drop3 step_days=1 Sharpe
最大回撤
平均换手
2024 年压力样本回撤
每日 live 指定股票排名稳定性
```

## 7. 当前结论

当前不能说“无流动性过滤训练更好”或“更差”。

能确定的是：

```text
现有正式模型是在预处理阶段就过滤过的股票池上训练的。
关闭 live 预测的流动性过滤，只是诊断用途。
真正无流动性训练需要重新生成 processed 数据。
```

从实盘角度，暂时不建议直接采用无流动性 live 排名。

下一步应该先补齐完整历史原始数据或未过滤 processed，再进行正式训练和 walk-forward 回测。
