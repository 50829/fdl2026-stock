# Outputs

Local artifacts are ignored by git except Markdown notes. New runs should use these roots:

- `outputs/models/<YYYYMMDD_HHMMSS>__<experiment_name>/` for model training, prediction, and evaluation artifacts.
- `outputs/strategy/<YYYYMMDD_HHMMSS>__<run_name>/` for strategy backtests.
- `outputs/live/<trade_date>__<model_or_strategy>__from_<decision_date>/` for daily trading plans.

Use `configs/registry/models.yaml` to register model prediction files and live model artifacts. Prefer updating the registry over adding hardcoded paths to scripts.

To inspect local legacy names:

```bash
python -m src.experiments normalize-outputs --dry-run
```

To migrate local legacy names:

```bash
python -m src.experiments normalize-outputs --apply
```
