# Smoke Pipeline

This folder keeps the temporary end-to-end pipeline that was added to validate the training, evaluation, and backtest flow without mixing those prototype modules into the main `src/` package.

Run from the project root:

```bash
python tests/smoke_pipeline/train.py --config tests/configs/smoke.yaml
python tests/smoke_pipeline/eval.py --config tests/configs/smoke.yaml
```
