# outputs 目录说明

本目录用于存放训练/验证过程中生成的**产物文件**（checkpoint、预测结果等）。一般不需要手动改动其中的产物文件；你只需要通过修改 `config.yaml`，然后运行 `train.py` / `eval.py` 来更新这些产物。

> 注意：仓库中 `train.py` / `eval.py` 的代码文件在项目根目录；`outputs/` 仅保存运行结果。

---

## 1. 目录结构与文件含义

运行默认配置后，`outputs/` 通常包含：

### 1.1 `ckpt.pt`

`train.py` 训练结束后保存的 PyTorch checkpoint 文件。

它包含（核心字段）：

- `model_state`：模型参数（`state_dict`）
- `feature_cols`：本次训练实际使用的特征列名列表
- `label_col`：标签列名（例如 `label_ret_1d`）
- `seq_len`：滑动窗口长度（时间步数）
- `normalize`：Dataset 的标准化方式（目前默认 `window`）
- `cfg`：保存一份当次训练时使用的配置内容，方便复现实验

用途：

- 用于 `eval.py` 加载并在验证集上计算指标
- 后续你替换模型结构时（修改 `models.py` 的 `build_model`），仍建议保持 checkpoint 保存逻辑不变，便于复现

### 1.2 `val_pred.csv`

`eval.py` 在验证集上推理后导出的逐样本结果文件（可在 `config.yaml` 中通过 `eval.pred_path` 控制是否导出以及导出路径）。

字段含义：

- `ts_code`：股票代码
- `trade_date`：样本对应的交易日（该日作为输入窗口的最后一天）
- `pred`：模型输出的预测分数（回归场景下可理解为对未来收益的估计/打分）
- `label`：真实标签（默认是未来 `horizon` 日收益率）

用途：

- 调试模型是否能在给定日期正常输出预测（避免“未来信息缺失导致无法预测”的问题）
- 作为后续回测/交易策略模块的输入（例如每天对全市场排序选股）

### 1.3 其他可能出现的文件

- 你可以在 `train.save_path` / `eval.pred_path` 指定任意文件名；因此 `outputs/` 下的文件名并非固定。
- 建议把不同实验用不同文件名保存（例如 `ckpt_transformer.pt`、`val_pred_2025.csv`），方便对比。

### 1.4 `bt_curve.csv`（历史回测资金曲线，可选）

当 `config.yaml` 里开启 `backtest.enabled: true` 且设置了 `backtest.curve_path`，`eval.py` 会在验证集区间做一次简单历史回测，并导出资金曲线：

- `trade_date`：调仓/评估日期（与预测日期对齐）
- `gross_ret`：组合当期毛收益（持仓股票 `label` 的均值）
- `net_ret`：扣除交易成本后的组合收益
- `turnover`：当期换手率（按等权近似）
- `holdings`：当期持仓数量
- `equity`：累计净值

---

## 2. 如何生成这些 outputs 产物

### 2.1 训练并生成 checkpoint

在项目根目录运行：

```bash
python train.py --config config.yaml
```

也可以用轻量化的 smoke 配置快速验证流程是否跑通：

```bash
python train.py --config config_smoke.yaml
```

关键配置项（`config.yaml`）：

- `data.*`：数据读取范围、数据目录
- `split.train_end / split.val_end`：按日期切分训练/验证（严格时间切分，不随机打乱）
- `task.horizon`：预测未来 `n` 日收益（标签由 `shift(-horizon)` 构造）
- `model.seq_len`：滑动窗口长度（每个样本包含最近 `seq_len` 个交易日特征）
- `train.save_path`：checkpoint 保存路径，默认 `outputs/ckpt.pt`

训练日志：

- `train.py` 每个 epoch 会输出一行 JSON，包含 `epoch/train_loss/val_loss`，便于你重定向到文件或用脚本解析。

进度显示：

- `train.use_tqdm: true` 时，会显示训练/验证的进度条，并在读取大量日频 CSV 时显示文件读取进度。
- 如果你感觉“迟迟没有反应”，通常是在做数据读取或特征工程（会先耗时，之后才进入 epoch 循环）。

### 2.2 验证并导出预测

在项目根目录运行：

```bash
python eval.py --config config.yaml
```

对应的 smoke 验证：

```bash
python eval.py --config config_smoke.yaml
```

关键配置项：

- `eval.ckpt`：要加载的 checkpoint 路径
- `eval.pred_path`：是否导出预测结果 CSV（为空/不填则不导出）

`eval.py` 输出：

- `val_mse`：验证集均方误差（回归损失）
- `ic_mean`：按天计算 Spearman IC 后的均值
- `icir`：`ic_mean / ic_std`（IC 的均值除以波动）
- `ic_days`：有效交易日数量（当日可用股票数太少会跳过）

如果开启回测（`backtest.enabled: true`），还会额外输出一段 `{"backtest": ...}`，包含：

- `annual_return`：年化收益率
- `sharpe`：夏普比率（按 252 交易日年化）
- `max_drawdown`：最大回撤
- `avg_turnover`：平均换手率

回测策略可外接配置：

- `backtest.strategy.name`：策略名称（当前内置 `topk_rotate`）
- `backtest.strategy.params`：策略参数（例如 `n_hold/k_rotate/transaction_cost_bps`）

提示：

- 在全量数据上跑验证时，建议把 `eval.pred_path` 设为 `null`，避免导出超大的逐样本 CSV。

---

## 3. outputs 与各代码文件的对应关系（你关心的“每段代码做什么”）

下面按“谁生成 outputs 里哪个文件”的角度解释：

### 3.1 `train.py` → `outputs/ckpt.pt`

`train.py` 做的事情：

1. 读取数据并拼接成面板（默认从 `documents-export-2026-5-18/daily/` 按日期加载）
2. 可选过滤股票池：北交所、ST（在 `config.yaml` 的 `data.pool` 配置）
3. 在每只股票内部做特征提取（技术指标/滚动特征等），并构造未来收益标签
4. 按日期切分训练/验证集（严格时间切分，防止未来信息泄露）
5. 构造滑动窗口样本并训练模型
6. 保存 checkpoint 到 `train.save_path`

你替换模型时最常修改：

- `models.py` 的 `build_model(cfg, in_dim)`（`train.py` 会调用它来创建模型）

### 3.2 `eval.py` → `outputs/val_pred.csv`

`eval.py` 做的事情：

1. 加载 `eval.ckpt` 指定的 checkpoint
2. 用与训练一致的方式重新读取数据、提取特征、构造标签、切出验证集
3. 用 checkpoint 中的 `feature_cols/seq_len/normalize` 构造 Dataset（确保和训练一致）
4. 在验证集上推理，计算 `MSE`、按天 IC/ICIR
5. 若配置了 `eval.pred_path`，导出逐样本 `ts_code/trade_date/pred/label`

---

## 4. 常见问题（排错）

### 4.1 提示“训练集/验证集样本数为 0”

通常原因：

- `data.start_date~end_date` 太短，无法满足 `model.seq_len` 的窗口长度
- 过滤条件太强（比如 ST 过滤 + 北交所过滤 + 日期范围短）
- `split.train_end/val_end` 配置导致验证集日期段没有足够数据

解决方式（优先级从高到低）：

- 扩大 `data.end_date`（或整体日期范围）
- 减小 `model.seq_len`
- 调整 `split.train_end/val_end`
- 暂时关闭过滤：`data.pool.exclude_st: false` / `exclude_bj: false`（仅用于调试）

### 4.2 训练/验证能跑但 IC 很低

这是正常现象（尤其是用占位模型 `DummyModel` 时）。IC/ICIR 主要用于衡量排序能力；后续替换为更合适的时序模型、增加特征工程、改损失函数/采样方式等，才会逐步改善。

---

## 5. 建议的实验产物管理方式

- 不同实验保存到不同文件名：
  - `train.save_path: outputs/ckpt_exp1.pt`
  - `eval.pred_path: outputs/val_pred_exp1.csv`
- 每次训练把 `config.yaml` 也复制一份到 `outputs/`（手动或脚本），方便复现实验。
