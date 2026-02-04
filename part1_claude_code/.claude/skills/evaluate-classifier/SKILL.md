# Skill: Evaluate Wine Classifier

## Purpose
Evaluate a trained Wine classifier and generate a concise performance report.

## Steps
1. Load train/test splits from parquet files.
2. Load the trained model artifact.
3. Generate predictions on the test set.
4. Compute accuracy, precision, recall, F1-score, and confusion matrix.
5. Save metrics to JSON and plot the confusion matrix.
6. Summarize results into a markdown report section.

## Output Artifacts
- `metrics.json`
- `confusion_matrix.png`
- `evaluation_summary.md`
