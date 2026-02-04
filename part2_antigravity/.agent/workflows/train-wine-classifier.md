# Workflow: train-wine-classifier

## Goal
Train and evaluate an XGBoost classifier for Wine classification.

## Steps
1. Load train/test splits from parquet files in `output/antigravity/`.
2. Train XGBoost classifier with fixed hyperparameters.
3. Run 5-fold stratified cross-validation and record metrics.
4. Evaluate on the test set with accuracy, precision, recall, F1-score.
5. Save confusion matrix and feature importance plot.
6. Save model artifact and metrics JSON files.

## Quality Checks
- Verify metrics JSON is written.
- Confirm confusion matrix and feature importance plots exist.
