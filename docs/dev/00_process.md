
  我建议下一步优先做 LightGBM baseline，然后再做融合。

  具体顺序：

  1. 先跑 LightGBM baseline
      - 用同一套 label_5d__cs_rank
      - 用全量 112 features
      - valid/test 都评估 IC、ICIR、MSE、当前回测
      - 这样我们有一个强传统机器学习模型作为对照
  2. 做 LightGBM 特征重要性分析
      - 看哪些特征真正有效
      - 用 importance 选 Top-K 特征
      - 再训练一版 GRU，看是否比手工 core features 更合
        理
  3. 做 GRU + LightGBM 融合
     三个方向里我建议先做最简单、最稳的：
      - rank ensemble：final_score = alpha * rank(GRU)
        + (1-alpha) * rank(LightGBM)
      - alpha 先扫 0.2 / 0.4 / 0.5 / 0.6 / 0.8
      - 看 valid IC/ICIR，最后只在 test 上评一次
  4. 之后再考虑 daily CorrLoss
     这个更贴近 IC，但需要改 batch 组织方式，工程量更
     大。它值得做，但我建议排在 LightGBM baseline 后
     面，因为 LightGBM 能快速告诉我们传统模型上限在哪
     里。