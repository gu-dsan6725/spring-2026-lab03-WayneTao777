# Demo Reference Material

This folder contains reference implementations and the original exercises.

## Contents

- **solved/**: Pre-built Python scripts showing one possible implementation of the California Housing ML pipeline
- **exercises/**: The original three separate exercises (now consolidated into the single unified lab in the parent README)

## Solved Pipeline

| Script | Description |
|--------|-------------|
| `01_eda.py` | EDA: statistics, distributions, correlations, outliers |
| `02_feature_engineering.py` | Derived features, scaling, train/test split |
| `03_xgboost_model.py` | XGBoost training with CV and hyperparameter tuning |
| `04_generate_report.py` | Report generation from model artifacts |

Run them in order from the repo root:

```bash
uv run python part1_claude_code/demo/solved/01_eda.py
uv run python part1_claude_code/demo/solved/02_feature_engineering.py
uv run python part1_claude_code/demo/solved/03_xgboost_model.py
uv run python part1_claude_code/demo/solved/04_generate_report.py
```
