# Data Quality Rules

## Wine Dataset Requirements
- Use `sklearn.datasets.load_wine()` as the source.
- Verify dataset shape and class balance before modeling.
- Check for missing values and non-finite values.
- Log outlier counts using the IQR method.

## Preprocessing
- Add at least 3 derived features.
- Standardize features with `StandardScaler`.
- Use stratified train/test split.

## Evaluation Metrics
- Report accuracy, precision, recall, F1-score, and confusion matrix.
- Use 5-fold stratified cross-validation.
