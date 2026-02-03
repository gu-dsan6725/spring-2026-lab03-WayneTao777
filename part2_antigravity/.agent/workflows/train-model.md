# Train Model Workflow

Train an XGBoost model on the prepared dataset.

## Prerequisites

Run the EDA and feature engineering steps first to generate the train/test splits.

## Steps

1. Load the prepared train/test splits from `output/` (parquet files)
2. Train an XGBoost model with reasonable hyperparameters
3. Generate predictions on the test set
4. Compute evaluation metrics appropriate for the task (classification or regression)
5. Create diagnostic plots (confusion matrix for classification, residual plots for regression)
6. Create a feature importance bar chart
7. Save the trained model as `output/xgboost_model.joblib`
8. Write an evaluation report to `output/evaluation_report.md`

## Requirements

- Use polars for data loading
- Follow the coding standards in `.agent/rules/code-style-guide.md`
- Save the training script as `part2_antigravity/src/03_xgboost_model.py`
- After writing the file, run `uv run ruff check --fix` and `uv run python -m py_compile`
