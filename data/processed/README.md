# Processed Data

本目录由 `python -m src.data.preprocess --config configs/config.yaml` 生成。

## 数据范围

- 开始日期：`20160104`
- 结束日期：`20260518`
- 交易日数量：`2516`
- 股票数量：`5758`
- 面板行数：`10692159`
- 特征数量：`112`

## 文件说明

- `panel.parquet`：合并后的原始日频面板，键为 `trade_date, ts_code`。
- `data_quality.json`：数据质量检查摘要。
- `universe.parquet`：每日股票池过滤结果。
- `features.parquet`：模型输入特征。
- `labels.parquet`：`label_1d`、`label_5d`、市场超额标签及其截面 rank 标签。
- `feature_meta.json`：默认特征列、特征分组、处理方式和生成配置。
- `splits.json`：训练、验证、测试时间切分。

## 关键约束

- 特征只使用当日盘后及以前可获得的数据。
- 截面标准化只在当日 `in_universe=True` 的可交易股票池内部计算。
- 标签从 `T+1` 买入价开始计算，避免使用不可交易收益。
- 标签使用全市场交易日历对齐，不按单只股票记录跳过停牌/缺失日期。
- 训练、验证、测试按时间切分，禁止随机日期切分。
